"""
Amazon ML Worker — Entity Resolution Pipeline.

Business entity matching across multiple sources using:
  TSV loading → normalization → multi-pass blocking → pair features
  → LightGBM model → GroupKFold by source1_entity_id
  → Macro F0.5 threshold tuning → test inference → matching_results.tsv
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_sources(dataset_dir: str, split: str = "train") -> Dict[str, pd.DataFrame]:
    """Load source TSV files for the given split (train or test)."""
    base = Path(dataset_dir) / split
    sources = {}
    for i in range(1, 4):
        fname = f"{split}_source{i}.tsv"
        path = base / fname
        if path.exists():
            df = pd.read_csv(path, sep="\t", dtype=str)
            df = df.fillna("")
            sources[f"source{i}"] = df
            log.info("Loaded %s: %d rows, columns=%s", fname, len(df), list(df.columns))
        else:
            log.warning("File not found: %s", path)
    return sources


def load_ground_truth(dataset_dir: str) -> pd.DataFrame:
    """Load train ground truth."""
    path = Path(dataset_dir) / "train" / "train_ground_truth.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")
    gt = pd.read_csv(path, sep="\t", dtype=str)
    gt = gt.fillna("")
    log.info("Loaded ground truth: %d rows, columns=%s", len(gt), list(gt.columns))
    return gt


# ---------------------------------------------------------------------------
# 2. Normalization
# ---------------------------------------------------------------------------

def normalize_field(text: str) -> str:
    """Normalize a text field: NFKC, lowercase, collapse whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_entity(row: pd.Series) -> pd.Series:
    """Normalize all fields of an entity row."""
    result = row.copy()
    for col in ["business_name", "business_address", "country"]:
        if col in result.index:
            result[f"{col}_norm"] = normalize_field(str(result.get(col, "")))
    return result


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all entities in a DataFrame."""
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[f"{col}_norm"] = df[col].fillna("").apply(normalize_field)
    return df


def extract_tokens(text: str) -> Set[str]:
    """Extract word tokens from normalized text."""
    if not text:
        return set()
    return set(text.split())


def extract_numeric_tokens(text: str) -> Set[str]:
    """Extract numeric tokens (postal codes, house numbers, etc.)."""
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def extract_alpha_tokens(text: str) -> Set[str]:
    """Extract alphabetic tokens only."""
    if not text:
        return set()
    return {t for t in text.split() if t.isalpha()}


# ---------------------------------------------------------------------------
# 3. Multi-Pass Blocking
# ---------------------------------------------------------------------------

def _block_name_country(s1: pd.DataFrame, s2: pd.DataFrame) -> Set[Tuple[str, str]]:
    """Block on first word of normalized name + country."""
    pairs = set()
    # Build index from s2
    index: Dict[str, List[str]] = {}
    for _, row in s2.iterrows():
        name = row.get("business_name_norm", "")
        country = row.get("country_norm", "")
        if not name:
            continue
        first_word = name.split()[0] if name.split() else ""
        if first_word and len(first_word) >= 2:
            key = f"{first_word}|{country}"
            index.setdefault(key, []).append(row["entity_id"])

    for _, row in s1.iterrows():
        name = row.get("business_name_norm", "")
        country = row.get("country_norm", "")
        if not name:
            continue
        first_word = name.split()[0] if name.split() else ""
        if first_word and len(first_word) >= 2:
            key = f"{first_word}|{country}"
            for s2_id in index.get(key, []):
                pairs.add((row["entity_id"], s2_id))
    return pairs


def _block_numeric_address(s1: pd.DataFrame, s2: pd.DataFrame) -> Set[Tuple[str, str]]:
    """Block on numeric tokens in address + country."""
    pairs = set()
    index: Dict[str, List[str]] = {}
    for _, row in s2.iterrows():
        addr = row.get("business_address_norm", "")
        country = row.get("country_norm", "")
        nums = extract_numeric_tokens(addr)
        if nums:
            key = f"{'|'.join(sorted(nums)[:3])}|{country}"
            index.setdefault(key, []).append(row["entity_id"])

    for _, row in s1.iterrows():
        addr = row.get("business_address_norm", "")
        country = row.get("country_norm", "")
        nums = extract_numeric_tokens(addr)
        if nums:
            key = f"{'|'.join(sorted(nums)[:3])}|{country}"
            for s2_id in index.get(key, []):
                pairs.add((row["entity_id"], s2_id))
    return pairs


def _block_name_trigram(s1: pd.DataFrame, s2: pd.DataFrame, top_k: int = 10) -> Set[Tuple[str, str]]:
    """Block using character 3-gram TF-IDF on business_name."""
    s1_names = s1["business_name_norm"].fillna("").tolist()
    s2_names = s2["business_name_norm"].fillna("").tolist()
    s1_ids = s1["entity_id"].tolist()
    s2_ids = s2["entity_id"].tolist()

    if not s1_names or not s2_names:
        return set()

    # Filter empty
    valid_s1 = [(i, n) for i, n in enumerate(s1_names) if len(n) >= 2]
    valid_s2 = [(i, n) for i, n in enumerate(s2_names) if len(n) >= 2]

    if not valid_s1 or not valid_s2:
        return set()

    s1_idx, s1_texts = zip(*valid_s1)
    s2_idx, s2_texts = zip(*valid_s2)

    all_texts = list(s1_texts) + list(s2_texts)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_features=50000)
    try:
        mat = tfidf.fit_transform(all_texts)
    except ValueError:
        return set()

    s1_mat = mat[:len(s1_texts)]
    s2_mat = mat[len(s1_texts):]

    pairs = set()
    # Process in chunks to limit memory
    chunk_size = 500
    for start in range(0, s1_mat.shape[0], chunk_size):
        end = min(start + chunk_size, s1_mat.shape[0])
        sims = sk_cosine(s1_mat[start:end], s2_mat)
        for local_i in range(sims.shape[0]):
            global_i = start + local_i
            top_indices = np.argsort(sims[local_i])[-top_k:]
            for j in top_indices:
                if sims[local_i, j] > 0.2:
                    pairs.add((s1_ids[s1_idx[global_i]], s2_ids[s2_idx[j]]))

    return pairs


def generate_candidates(
    s1: pd.DataFrame,
    other_sources: Dict[str, pd.DataFrame],
    worker_id: int = 0,
    total_workers: int = 1,
) -> pd.DataFrame:
    """
    Generate candidate pairs using multi-pass blocking.

    Supports worker sharding: each worker handles a shard of S1 entities.
    """
    # Shard S1 for this worker
    s1_ids = s1["entity_id"].unique()
    if total_workers > 1:
        all_ids = sorted(s1_ids)
        worker_ids = [aid for aid in all_ids if hash(aid) % total_workers == worker_id]
        s1 = s1[s1["entity_id"].isin(worker_ids)].reset_index(drop=True)
        log.info("Worker %d/%d: processing %d S1 entities", worker_id, total_workers, len(s1))

    all_pairs: Set[Tuple[str, str]] = set()

    for src_name, s2 in other_sources.items():
        log.info("Blocking S1 vs %s...", src_name)

        # Pass 1: name + country
        p1 = _block_name_country(s1, s2)
        log.info("  name+country block: %d pairs", len(p1))
        all_pairs.update(p1)

        # Pass 2: numeric address
        p2 = _block_numeric_address(s1, s2)
        log.info("  numeric+address block: %d pairs", len(p2))
        all_pairs.update(p2)

        # Pass 3: name trigram TF-IDF
        p3 = _block_name_trigram(s1, s2, top_k=10)
        log.info("  name trigram block: %d pairs", len(p3))
        all_pairs.update(p3)

    log.info("Total candidate pairs (union): %d", len(all_pairs))

    if not all_pairs:
        return pd.DataFrame(columns=["source1_entity_id", "source2_entity_id"])

    pairs_df = pd.DataFrame(list(all_pairs), columns=["source1_entity_id", "source2_entity_id"])
    pairs_df = pairs_df.drop_duplicates().reset_index(drop=True)
    return pairs_df


# ---------------------------------------------------------------------------
# 4. Pair Feature Engineering
# ---------------------------------------------------------------------------

def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _edit_similarity(a: str, b: str) -> float:
    """Normalized edit distance similarity (1 - edit_distance / max_len)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Use simple DP
    la, lb = len(a), len(b)
    if la > 500 or lb > 500:
        # Truncate for performance
        a, b = a[:500], b[:500]
        la, lb = len(a), len(b)
    # Optimized: use numpy-based approach for longer strings
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def _length_ratio(a: str, b: str) -> float:
    """Length ratio (shorter/longer)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    return min(la, lb) / max(la, lb)


def compute_pair_features(
    row1: Dict,
    row2: Dict,
) -> Dict[str, float]:
    """Compute features for a single entity pair."""
    features = {}

    # Name features
    name1 = row1.get("business_name_norm", "")
    name2 = row2.get("business_name_norm", "")

    features["name_exact"] = float(name1 == name2 and name1 != "")
    features["name_jaccard"] = _jaccard(extract_tokens(name1), extract_tokens(name2))
    features["name_edit_sim"] = _edit_similarity(name1, name2)
    features["name_len_ratio"] = _length_ratio(name1, name2)
    features["name_len1"] = float(len(name1))
    features["name_len2"] = float(len(name2))

    # Name first-word match
    w1 = name1.split() if name1 else []
    w2 = name2.split() if name2 else []
    features["name_first_word_match"] = 1.0 if (w1 and w2 and w1[0] == w2[0]) else 0.0

    # Name alpha-token Jaccard
    features["name_alpha_jaccard"] = _jaccard(extract_alpha_tokens(name1), extract_alpha_tokens(name2))

    # Address features
    addr1 = row1.get("business_address_norm", "")
    addr2 = row2.get("business_address_norm", "")

    features["addr_exact"] = float(addr1 == addr2 and addr1 != "")
    features["addr_jaccard"] = _jaccard(extract_tokens(addr1), extract_tokens(addr2))
    features["addr_edit_sim"] = _edit_similarity(addr1, addr2)
    features["addr_len_ratio"] = _length_ratio(addr1, addr2)

    # Numeric overlap in address (postal, house numbers)
    nums1 = extract_numeric_tokens(addr1)
    nums2 = extract_numeric_tokens(addr2)
    features["addr_numeric_jaccard"] = _jaccard(nums1, nums2)
    features["addr_numeric_overlap"] = float(len(nums1 & nums2)) if nums1 or nums2 else 0.0

    # Country match
    c1 = row1.get("country_norm", "")
    c2 = row2.get("country_norm", "")
    features["country_match"] = float(c1 == c2 and c1 != "")
    features["country_both_empty"] = float(c1 == "" and c2 == "")

    # Combined features
    features["name_addr_concat_jaccard"] = _jaccard(
        extract_tokens(f"{name1} {addr1}"),
        extract_tokens(f"{name2} {addr2}")
    )

    return features


def compute_tfidf_cosine_features(
    pairs_df: pd.DataFrame,
    entities: Dict[str, Dict],
) -> np.ndarray:
    """Compute TF-IDF character cosine similarity for all pairs at once."""
    # Collect all unique entity IDs
    all_ids = set(pairs_df["source1_entity_id"]) | set(pairs_df["source2_entity_id"])
    id_list = sorted(all_ids)
    id_to_idx = {eid: i for i, eid in enumerate(id_list)}

    texts = [entities.get(eid, {}).get("business_name_norm", "") for eid in id_list]

    if not any(texts):
        return np.zeros(len(pairs_df))

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_features=30000)
    try:
        mat = tfidf.fit_transform(texts)
    except ValueError:
        return np.zeros(len(pairs_df))

    cosines = np.zeros(len(pairs_df))
    for i, (_, row) in enumerate(pairs_df.iterrows()):
        idx1 = id_to_idx.get(row["source1_entity_id"])
        idx2 = id_to_idx.get(row["source2_entity_id"])
        if idx1 is not None and idx2 is not None:
            sim = sk_cosine(mat[idx1:idx1+1], mat[idx2:idx2+1])
            cosines[i] = float(sim[0, 0])

    return cosines


def build_pair_features(
    candidates: pd.DataFrame,
    entities: Dict[str, Dict],
    worker_id: int = 0,
    total_workers: int = 1,
) -> pd.DataFrame:
    """
    Build feature matrix for candidate pairs.

    Supports worker sharding by distributing pairs across workers.
    """
    if total_workers > 1:
        n = len(candidates)
        indices = [i for i in range(n) if i % total_workers == worker_id]
        candidates = candidates.iloc[indices].reset_index(drop=True)
        log.info("Worker %d/%d: computing features for %d pairs", worker_id, total_workers, len(candidates))

    features_list = []
    for _, row in candidates.iterrows():
        e1 = entities.get(row["source1_entity_id"], {})
        e2 = entities.get(row["source2_entity_id"], {})
        feats = compute_pair_features(e1, e2)
        feats["source1_entity_id"] = row["source1_entity_id"]
        feats["source2_entity_id"] = row["source2_entity_id"]
        features_list.append(feats)

    if not features_list:
        return pd.DataFrame()

    feat_df = pd.DataFrame(features_list)

    # Add TF-IDF cosine as a batch feature
    tfidf_cos = compute_tfidf_cosine_features(candidates, entities)
    feat_df["name_tfidf_cosine"] = tfidf_cos

    log.info("Built %d pair features for %d candidates", len(feat_df.columns) - 2, len(feat_df))
    return feat_df


# ---------------------------------------------------------------------------
# 5. Ground Truth Label Assignment
# ---------------------------------------------------------------------------

def assign_labels(
    pairs_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign binary match labels to candidate pairs using ground truth.

    Ground truth has columns: source1_entity_id, source2_entity_id (or matched_entity_ids).
    """
    # Detect ground truth format
    if "source1_entity_id" in ground_truth.columns and "source2_entity_id" in ground_truth.columns:
        # Direct pair format
        gt_set = set()
        for _, row in ground_truth.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            s2_ids = str(row.get("source2_entity_id", row.get("matched_entity_ids", ""))).strip()
            for s2_id in s2_ids.split(","):
                s2_id = s2_id.strip()
                if s2_id:
                    gt_set.add((s1, s2_id))
    elif "source1_entity_id" in ground_truth.columns and "matched_entity_ids" in ground_truth.columns:
        gt_set = set()
        for _, row in ground_truth.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            matched = str(row.get("matched_entity_ids", "")).strip()
            for s2_id in matched.split(","):
                s2_id = s2_id.strip()
                if s2_id:
                    gt_set.add((s1, s2_id))
    else:
        raise ValueError(f"Unexpected ground truth columns: {list(ground_truth.columns)}")

    labels = []
    for _, row in pairs_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        s2 = str(row["source2_entity_id"]).strip()
        labels.append(1 if (s1, s2) in gt_set else 0)

    result = pairs_df.copy()
    result["label"] = labels
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    log.info("Labels assigned: %d positive, %d negative (%.2f%% positive)",
             n_pos, n_neg, 100.0 * n_pos / max(len(labels), 1))
    return result


# ---------------------------------------------------------------------------
# 6. Blocking Recall
# ---------------------------------------------------------------------------

def compute_blocking_recall(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, float]:
    """Compute what fraction of true pairs are in the candidate set."""
    # Build GT pair set
    gt_pairs = set()
    if "matched_entity_ids" in ground_truth.columns:
        for _, row in ground_truth.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            matched = str(row.get("matched_entity_ids", "")).strip()
            for s2 in matched.split(","):
                s2 = s2.strip()
                if s2:
                    gt_pairs.add((s1, s2))
    elif "source2_entity_id" in ground_truth.columns:
        for _, row in ground_truth.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            s2 = str(row["source2_entity_id"]).strip()
            if s2:
                gt_pairs.add((s1, s2))

    if not gt_pairs:
        log.warning("No ground truth pairs found for blocking recall")
        return {"blocking_recall": 0.0, "gt_pairs": 0, "found": 0}

    cand_set = set(zip(
        candidates["source1_entity_id"].astype(str),
        candidates["source2_entity_id"].astype(str)
    ))

    found = len(gt_pairs & cand_set)
    recall = found / len(gt_pairs)
    log.info("Blocking recall: %.4f (%d/%d GT pairs found in candidates)",
             recall, found, len(gt_pairs))
    return {"blocking_recall": recall, "gt_pairs": len(gt_pairs), "found": found}


# ---------------------------------------------------------------------------
# 7. Model Training & Threshold Tuning
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature column names (exclude ID and label columns)."""
    exclude = {"source1_entity_id", "source2_entity_id", "label"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in (
        np.float64, np.float32, np.int64, np.int32, float, int
    )]


def train_er_model(
    train_features: pd.DataFrame,
    model_type: str = "lightgbm",
    model_params: Optional[Dict] = None,
    seed: int = 42,
):
    """Train entity resolution model."""
    from src.models.baseline import BaselineModel

    feat_cols = get_feature_columns(train_features)
    X = train_features[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = train_features["label"].values.astype(int)

    model = BaselineModel(model_type=model_type, model_params=model_params or {})
    model.fit(X, y)

    return model, feat_cols


def predict_proba_er(model, features_df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    """Get match probability predictions."""
    X = features_df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    proba = model.predict_proba(X)
    if proba is not None and proba.ndim == 2:
        return proba[:, 1]
    # Fallback: use predictions directly
    return model.predict(X).astype(float)


def macro_f05_score(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """
    Compute entity-level Macro F0.5 score.

    Groups are source1_entity_ids. For each group, compute F0.5,
    then average across groups.
    """
    unique_groups = np.unique(groups)
    scores = []
    beta = 0.5
    beta_sq = beta ** 2

    for g in unique_groups:
        mask = groups == g
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        if tp + fp + fn == 0:
            scores.append(1.0)  # No positives and no predictions: perfect
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
            else:
                f05 = 0.0
            scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0


def tune_threshold_f05(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: np.ndarray,
    n_thresholds: int = 200,
) -> Dict[str, float]:
    """Search for the best threshold optimizing entity-level Macro F0.5."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -1.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        score = macro_f05_score(y_true, preds, groups)
        if score > best_score:
            best_score = score
            best_threshold = thresh
            # Overall precision/recall
            tp = np.sum((y_true == 1) & (preds == 1))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            best_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    log.info("Best threshold: %.4f (Macro F0.5=%.6f, P=%.4f, R=%.4f)",
             best_threshold, best_score, best_precision, best_recall)
    return {
        "best_threshold": round(float(best_threshold), 4),
        "best_f05_macro": round(float(best_score), 6),
        "precision": round(float(best_precision), 6),
        "recall": round(float(best_recall), 6),
    }


# ---------------------------------------------------------------------------
# 8. GroupKFold Validation
# ---------------------------------------------------------------------------

def group_kfold_validate(
    features_df: pd.DataFrame,
    model_type: str = "lightgbm",
    model_params: Optional[Dict] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """GroupKFold cross-validation grouped by source1_entity_id."""
    from sklearn.model_selection import GroupKFold

    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = features_df["label"].values.astype(int)
    groups = features_df["source1_entity_id"].values

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    fold_scores = []
    all_probas = np.zeros(len(y))
    all_groups = groups.copy()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        from src.models.baseline import BaselineModel
        model = BaselineModel(model_type=model_type, model_params=model_params or {})
        model.fit(X[train_idx], y[train_idx])

        proba = model.predict_proba(X[val_idx])
        if proba is not None and proba.ndim == 2:
            all_probas[val_idx] = proba[:, 1]
        else:
            all_probas[val_idx] = model.predict(X[val_idx]).astype(float)

        # Per-fold score at 0.5
        preds = (all_probas[val_idx] >= 0.5).astype(int)
        fold_f05 = macro_f05_score(y[val_idx], preds, groups[val_idx])
        fold_scores.append(fold_f05)
        log.info("Fold %d: Macro F0.5 = %.4f", fold, fold_f05)

    # Overall threshold tuning on OOF predictions
    threshold_result = tune_threshold_f05(y, all_probas, groups)

    # Error analysis
    best_preds = (all_probas >= threshold_result["best_threshold"]).astype(int)
    false_merges = np.sum((y == 0) & (best_preds == 1))
    singleton_errors = 0
    for g in np.unique(groups):
        mask = groups == g
        gt_has_match = np.any(y[mask] == 1)
        pred_has_match = np.any(best_preds[mask] == 1)
        if gt_has_match and not pred_has_match:
            singleton_errors += 1

    result = {
        "mean_fold_f05": round(float(np.mean(fold_scores)), 6),
        "std_fold_f05": round(float(np.std(fold_scores)), 6),
        **threshold_result,
        "false_merges": int(false_merges),
        "singleton_errors": int(singleton_errors),
    }
    log.info("GroupKFold results: %s", result)
    return result


# ---------------------------------------------------------------------------
# 9. Test Inference & Output
# ---------------------------------------------------------------------------

def generate_matching_results(
    test_candidates: pd.DataFrame,
    test_features: pd.DataFrame,
    model,
    feat_cols: List[str],
    threshold: float,
    test_s1_ids: List[str],
) -> pd.DataFrame:
    """
    Generate matching_results.tsv from test predictions.

    One row per test S1 entity, with matched S2/S3 IDs.
    """
    if len(test_features) > 0:
        probas = predict_proba_er(model, test_features, feat_cols)
        test_features = test_features.copy()
        test_features["proba"] = probas
        matches = test_features[test_features["proba"] >= threshold]
    else:
        matches = pd.DataFrame(columns=["source1_entity_id", "source2_entity_id"])

    # Group by S1 entity
    match_dict: Dict[str, List[str]] = {}
    for _, row in matches.iterrows():
        s1 = str(row["source1_entity_id"])
        s2 = str(row["source2_entity_id"])
        match_dict.setdefault(s1, []).append(s2)

    rows = []
    for s1_id in test_s1_ids:
        s1_id = str(s1_id)
        matched = match_dict.get(s1_id, [])
        # Deduplicate
        matched = list(dict.fromkeys(matched))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched) if matched else "",
        })

    result = pd.DataFrame(rows)
    n_matched = sum(1 for r in rows if r["matched_entity_ids"])
    n_singletons = sum(1 for r in rows if not r["matched_entity_ids"])
    log.info("Matching results: %d entities, %d matched, %d singletons",
             len(result), n_matched, n_singletons)
    return result


# ---------------------------------------------------------------------------
# 10. Full Pipeline
# ---------------------------------------------------------------------------

def run_entity_resolution(
    dataset_dir: str = "dataset",
    output_dir: str = "output",
    model_type: str = "lightgbm",
    model_params: Optional[Dict] = None,
    worker_id: int = 0,
    total_workers: int = 1,
    seed: int = 42,
) -> Dict:
    """
    Run the full entity resolution pipeline.

    Returns metrics dict.
    """
    from src.utils.timing import Timer
    timer = Timer()
    timer.start("total")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load Data ──────────────────────────────────────────────────────────
    with timer.section("load"):
        train_sources = load_sources(dataset_dir, "train")
        test_sources = load_sources(dataset_dir, "test")
        ground_truth = load_ground_truth(dataset_dir)

    # ── Normalize ──────────────────────────────────────────────────────────
    with timer.section("normalize"):
        for name in train_sources:
            train_sources[name] = normalize_df(train_sources[name])
        for name in test_sources:
            test_sources[name] = normalize_df(test_sources[name])
        log.info("Normalization complete")

    # ── Build Entity Lookup ────────────────────────────────────────────────
    entities: Dict[str, Dict] = {}
    for name, df in {**train_sources, **test_sources}.items():
        for _, row in df.iterrows():
            entities[str(row["entity_id"])] = row.to_dict()

    # ── Training: Candidate Generation ─────────────────────────────────────
    with timer.section("train_blocking"):
        s1_train = train_sources["source1"]
        other_train = {k: v for k, v in train_sources.items() if k != "source1"}
        train_candidates = generate_candidates(s1_train, other_train, worker_id, total_workers)

        # Save train candidates
        train_candidates.to_csv(
            Path(output_dir) / "train_candidate_pairs.tsv",
            sep="\t", index=False
        )

    # ── Blocking Recall ────────────────────────────────────────────────────
    blocking_metrics = compute_blocking_recall(train_candidates, ground_truth)

    # ── Training: Feature Generation ───────────────────────────────────────
    with timer.section("train_features"):
        train_feat = build_pair_features(train_candidates, entities, worker_id, total_workers)

    # ── Label Assignment ───────────────────────────────────────────────────
    if len(train_feat) > 0:
        train_feat = assign_labels(train_feat, ground_truth)
    else:
        log.error("No training features generated!")
        return {"error": "No training features"}

    # ── GroupKFold Validation ──────────────────────────────────────────────
    with timer.section("validation"):
        cv_results = group_kfold_validate(
            train_feat, model_type=model_type,
            model_params=model_params, seed=seed
        )

    # ── Train Final Model ──────────────────────────────────────────────────
    with timer.section("train_final"):
        final_model, feat_cols = train_er_model(
            train_feat, model_type=model_type,
            model_params=model_params, seed=seed
        )
        threshold = cv_results["best_threshold"]

    # ── Test: Candidate Generation ─────────────────────────────────────────
    with timer.section("test_blocking"):
        s1_test = test_sources.get("source1")
        if s1_test is None:
            log.error("No test source1 found")
            return {"error": "No test source1"}

        other_test = {k: v for k, v in test_sources.items() if k != "source1"}
        test_candidates = generate_candidates(s1_test, other_test, worker_id, total_workers)

    # ── Test: Feature Generation ───────────────────────────────────────────
    with timer.section("test_features"):
        test_feat = build_pair_features(test_candidates, entities, worker_id, total_workers)

    # ── Save candidate_pairs.tsv ───────────────────────────────────────────
    test_candidates.to_csv(
        Path(output_dir) / "candidate_pairs.tsv",
        sep="\t", index=False
    )
    log.info("Saved candidate_pairs.tsv: %d pairs", len(test_candidates))

    # ── Test Inference ─────────────────────────────────────────────────────
    with timer.section("inference"):
        test_s1_ids = s1_test["entity_id"].unique().tolist()
        matching_results = generate_matching_results(
            test_candidates, test_feat, final_model,
            feat_cols, threshold, test_s1_ids
        )

    # ── Save matching_results.tsv ──────────────────────────────────────────
    matching_results.to_csv(
        Path(output_dir) / "matching_results.tsv",
        sep="\t", index=False
    )
    log.info("Saved matching_results.tsv: %d rows", len(matching_results))

    # ── Save model ─────────────────────────────────────────────────────────
    final_model.save(str(Path(output_dir) / "er_model.joblib"))

    timer.stop("total")

    # ── Metrics ────────────────────────────────────────────────────────────
    all_metrics = {
        **blocking_metrics,
        **cv_results,
        "threshold": threshold,
        "train_candidates": len(train_candidates),
        "test_candidates": len(test_candidates),
        "test_s1_entities": len(test_s1_ids),
        "runtime": timer.summary(),
    }

    # Save metrics
    import json
    with open(Path(output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("  Entity Resolution Complete")
    log.info("=" * 60)
    log.info("  Blocking recall:     %.4f", blocking_metrics["blocking_recall"])
    log.info("  CV Macro F0.5:       %.4f ± %.4f", cv_results["mean_fold_f05"], cv_results["std_fold_f05"])
    log.info("  Best threshold:      %.4f", threshold)
    log.info("  Best F0.5:           %.4f", cv_results["best_f05_macro"])
    log.info("  Precision:           %.4f", cv_results["precision"])
    log.info("  Recall:              %.4f", cv_results["recall"])
    log.info("  False merges:        %d", cv_results["false_merges"])
    log.info("  Singleton errors:    %d", cv_results["singleton_errors"])
    log.info("  Train candidates:    %d", len(train_candidates))
    log.info("  Test candidates:     %d", len(test_candidates))
    log.info("=" * 60)

    return all_metrics
