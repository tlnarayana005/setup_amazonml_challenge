"""
Create a ground-truth-aware smoke dataset for ER quality benchmarking.

Unlike naive head(N), this script:
1. Selects N Source1 entities
2. Includes their TRUE matching S2/S3 from ground truth
3. Adds negative (non-matching) S2/S3 records
4. Creates a proper train/test split

This ensures blocking recall can actually be measured.

Usage:
    python scripts/create_smoke_dataset.py --dataset-dir dataset --n-entities 2000
    python scripts/create_smoke_dataset.py --dataset-dir dataset --n-entities 2000 --output-dir smoke_dataset
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_smoke_dataset(
    dataset_dir: str = "dataset",
    output_dir: str = "smoke_dataset",
    n_entities: int = 2000,
    neg_multiplier: float = 2.5,
    test_fraction: float = 0.3,
    seed: int = 42,
):
    """
    Create a ground-truth-aware smoke dataset.

    Args:
        dataset_dir: Path to the full dataset
        output_dir: Where to write the smoke dataset
        n_entities: Number of Source1 entities to sample
        neg_multiplier: Ratio of negative S2/S3 to positive (per source)
        test_fraction: Fraction of S1 entities for test split
        seed: Random seed
    """
    rng = np.random.RandomState(seed)

    print("=" * 60)
    print("  Creating Ground-Truth-Aware Smoke Dataset")
    print("=" * 60)

    # ── Load full data ─────────────────────────────────────────────────────
    print("\nLoading full dataset...")
    s1 = pd.read_csv(Path(dataset_dir) / "train" / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(Path(dataset_dir) / "train" / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(Path(dataset_dir) / "train" / "train_source3.tsv", sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(Path(dataset_dir) / "train" / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    print(f"  Full S1: {len(s1):,}")
    print(f"  Full S2: {len(s2):,}")
    print(f"  Full S3: {len(s3):,}")
    print(f"  Full GT: {len(gt):,}")

    # ── Sample S1 entities ─────────────────────────────────────────────────
    n_entities = min(n_entities, len(s1))
    sampled_indices = rng.choice(len(s1), size=n_entities, replace=False)
    s1_sample = s1.iloc[sampled_indices].reset_index(drop=True)
    s1_ids = set(s1_sample["entity_id"])

    print(f"\n  Sampled S1: {len(s1_sample):,}")

    # ── Get ground truth for sampled S1 ────────────────────────────────────
    gt_sample = gt[gt["source1_entity_id"].isin(s1_ids)].reset_index(drop=True)

    # ── Collect TRUE matching S2/S3 IDs ────────────────────────────────────
    true_s2_ids = set()
    true_s3_ids = set()

    for _, row in gt_sample.iterrows():
        matched = str(row.get("matched_entity_ids", "")).strip()
        if not matched:
            continue
        for mid in matched.split(","):
            mid = mid.strip()
            if not mid:
                continue
            if mid.startswith("S2-") or mid.startswith("s2_"):
                true_s2_ids.add(mid)
            elif mid.startswith("S3-") or mid.startswith("s3_"):
                true_s3_ids.add(mid)
            else:
                # Try to find in S2 or S3
                if mid in set(s2["entity_id"]):
                    true_s2_ids.add(mid)
                elif mid in set(s3["entity_id"]):
                    true_s3_ids.add(mid)

    print(f"  True S2 matches: {len(true_s2_ids):,}")
    print(f"  True S3 matches: {len(true_s3_ids):,}")

    # ── Build S2 smoke: true matches + negative samples ────────────────────
    s2_true = s2[s2["entity_id"].isin(true_s2_ids)]
    s2_remaining = s2[~s2["entity_id"].isin(true_s2_ids)]
    n_neg_s2 = min(int(len(true_s2_ids) * neg_multiplier), len(s2_remaining))
    if n_neg_s2 > 0:
        neg_idx = rng.choice(len(s2_remaining), size=n_neg_s2, replace=False)
        s2_neg = s2_remaining.iloc[neg_idx]
    else:
        s2_neg = pd.DataFrame(columns=s2.columns)
    s2_smoke = pd.concat([s2_true, s2_neg], ignore_index=True)

    # ── Build S3 smoke: true matches + negative samples ────────────────────
    s3_true = s3[s3["entity_id"].isin(true_s3_ids)]
    s3_remaining = s3[~s3["entity_id"].isin(true_s3_ids)]
    n_neg_s3 = min(int(len(true_s3_ids) * neg_multiplier), len(s3_remaining))
    if n_neg_s3 > 0:
        neg_idx = rng.choice(len(s3_remaining), size=n_neg_s3, replace=False)
        s3_neg = s3_remaining.iloc[neg_idx]
    else:
        s3_neg = pd.DataFrame(columns=s3.columns)
    s3_smoke = pd.concat([s3_true, s3_neg], ignore_index=True)

    print(f"  Smoke S2: {len(s2_smoke):,} ({len(s2_true):,} true + {len(s2_neg):,} negative)")
    print(f"  Smoke S3: {len(s3_smoke):,} ({len(s3_true):,} true + {len(s3_neg):,} negative)")

    # ── Split into train/test ──────────────────────────────────────────────
    n_test = max(1, int(n_entities * test_fraction))
    n_train = n_entities - n_test

    s1_train = s1_sample.iloc[:n_train].reset_index(drop=True)
    s1_test = s1_sample.iloc[n_train:].reset_index(drop=True)

    gt_train = gt_sample[gt_sample["source1_entity_id"].isin(set(s1_train["entity_id"]))].reset_index(drop=True)

    print(f"\n  Train S1: {len(s1_train):,}")
    print(f"  Test  S1: {len(s1_test):,}")
    print(f"  Train GT: {len(gt_train):,}")

    # ── Save ───────────────────────────────────────────────────────────────
    out = Path(output_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    s1_train.to_csv(out / "train" / "train_source1.tsv", sep="\t", index=False)
    s2_smoke.to_csv(out / "train" / "train_source2.tsv", sep="\t", index=False)
    s3_smoke.to_csv(out / "train" / "train_source3.tsv", sep="\t", index=False)
    gt_train.to_csv(out / "train" / "train_ground_truth.tsv", sep="\t", index=False)

    s1_test.to_csv(out / "test" / "test_source1.tsv", sep="\t", index=False)
    s2_smoke.to_csv(out / "test" / "test_source2.tsv", sep="\t", index=False)
    s3_smoke.to_csv(out / "test" / "test_source3.tsv", sep="\t", index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    # Count non-empty GT rows
    n_with_match = sum(1 for _, r in gt_train.iterrows()
                       if str(r.get("matched_entity_ids", "")).strip())

    print(f"\n  Smoke dataset saved to: {output_dir}/")
    print(f"  GT rows with matches: {n_with_match:,} / {len(gt_train):,}")
    print("=" * 60)

    return {
        "train_s1": len(s1_train),
        "train_s2": len(s2_smoke),
        "train_s3": len(s3_smoke),
        "train_gt": len(gt_train),
        "test_s1": len(s1_test),
        "true_s2": len(true_s2_ids),
        "true_s3": len(true_s3_ids),
        "gt_with_match": n_with_match,
    }


def main():
    parser = argparse.ArgumentParser(description="Create ground-truth-aware smoke dataset")
    parser.add_argument("--dataset-dir", default="dataset", help="Full dataset directory")
    parser.add_argument("--output-dir", default="smoke_dataset", help="Output smoke dataset directory")
    parser.add_argument("--n-entities", type=int, default=2000, help="Number of S1 entities to sample")
    parser.add_argument("--neg-multiplier", type=float, default=2.5, help="Ratio of negative to positive S2/S3")
    parser.add_argument("--test-fraction", type=float, default=0.3, help="Fraction of S1 for test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    create_smoke_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n_entities=args.n_entities,
        neg_multiplier=args.neg_multiplier,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
