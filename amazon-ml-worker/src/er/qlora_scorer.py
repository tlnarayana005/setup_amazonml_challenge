"""
Amazon ML Worker — QLoRA Entity Resolution Scorer.

Uses Qwen2-0.5B fine-tuned with QLoRA to classify entity pairs as match/no-match.
Designed to replace LightGBM scoring while keeping the same blocking pipeline.

Usage:
    from src.er.qlora_scorer import train_qlora_er, predict_qlora_er
"""

import gc
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Text Formatting
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Basic normalization for entity fields."""
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def format_pair_prompt(entity1: Dict, entity2: Dict) -> str:
    """Format an entity pair as a classification prompt."""
    n1 = normalize_text(entity1.get("business_name", ""))
    a1 = normalize_text(entity1.get("business_address", ""))
    c1 = normalize_text(entity1.get("country", ""))

    n2 = normalize_text(entity2.get("business_name", ""))
    a2 = normalize_text(entity2.get("business_address", ""))
    c2 = normalize_text(entity2.get("country", ""))

    prompt = (
        f"Do these two business records refer to the same real-world company?\n\n"
        f"Business A:\n"
        f"  Name: {n1}\n"
        f"  Address: {a1}\n"
        f"  Country: {c1}\n\n"
        f"Business B:\n"
        f"  Name: {n2}\n"
        f"  Address: {a2}\n"
        f"  Country: {c2}\n\n"
        f"Answer (Yes or No):"
    )
    return prompt


# ---------------------------------------------------------------------------
# 2. Dataset Preparation
# ---------------------------------------------------------------------------

def prepare_training_data(
    candidates: pd.DataFrame,
    entities: Dict[str, Dict],
    ground_truth: pd.DataFrame,
    max_samples: int = 20000,
    neg_ratio: float = 3.0,
    seed: int = 42,
) -> List[Dict]:
    """
    Create training examples from candidates + ground truth.

    Returns list of {"prompt": str, "label": "Yes"/"No"}
    """
    rng = np.random.RandomState(seed)

    # Build GT set
    gt_set = set()
    for _, row in ground_truth.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched = str(row.get("matched_entity_ids", "")).strip()
        if matched:
            for mid in matched.split(","):
                mid = mid.strip()
                if mid:
                    gt_set.add((s1_id, mid))

    # Create positive and negative examples
    positives = []
    negatives = []

    for _, row in candidates.iterrows():
        s1_id = str(row.iloc[0])
        s2_id = str(row.iloc[1])

        if s1_id not in entities or s2_id not in entities:
            continue

        e1 = entities[s1_id]
        e2 = entities[s2_id]
        prompt = format_pair_prompt(e1, e2)

        if (s1_id, s2_id) in gt_set:
            positives.append({"prompt": prompt, "label": "Yes"})
        else:
            negatives.append({"prompt": prompt, "label": "No"})

    log.info("Raw examples: %d positive, %d negative", len(positives), len(negatives))

    # Balance dataset
    max_pos = min(len(positives), max_samples // 2)
    max_neg = min(len(negatives), int(max_pos * neg_ratio))

    if max_pos < len(positives):
        idx = rng.choice(len(positives), size=max_pos, replace=False)
        positives = [positives[i] for i in idx]

    if max_neg < len(negatives):
        idx = rng.choice(len(negatives), size=max_neg, replace=False)
        negatives = [negatives[i] for i in idx]

    data = positives + negatives
    rng.shuffle(data)

    log.info("Training set: %d positive, %d negative, %d total",
             len(positives), len(negatives), len(data))

    return data


# ---------------------------------------------------------------------------
# 3. QLoRA Training
# ---------------------------------------------------------------------------

def train_qlora_er(
    training_data: List[Dict],
    model_name: str = "Qwen/Qwen2-0.5B",
    output_dir: str = "output/qlora_model",
    max_seq_len: int = 256,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lr: float = 2e-4,
    epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    seed: int = 42,
):
    """
    Fine-tune a model with QLoRA for entity match classification.

    Returns (model, tokenizer).
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset

    log.info("Loading model: %s", model_name)

    # ── Quantization config ───────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model + tokenizer ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Prepare dataset ───────────────────────────────────────────────────
    def format_for_training(example):
        text = example["prompt"] + " " + example["label"]
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors=None,
        )
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    dataset = Dataset.from_list(training_data)
    tokenized = dataset.map(format_for_training, remove_columns=["prompt", "label"])

    # ── Split train/eval ──────────────────────────────────────────────────
    split = tokenized.train_test_split(test_size=0.1, seed=seed)

    # ── Training args ─────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        fp16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        seed=seed,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    log.info("Starting QLoRA training: %d samples, %d epochs", len(split["train"]), epochs)
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved to %s", output_dir)

    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------

def predict_qlora_er(
    model,
    tokenizer,
    candidates: pd.DataFrame,
    entities: Dict[str, Dict],
    batch_size: int = 32,
    max_seq_len: int = 256,
) -> pd.DataFrame:
    """
    Score candidate pairs using the fine-tuned QLoRA model.

    Returns DataFrame with columns: [source1_entity_id, candidate_entity_id, score]
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    # Get token IDs for "Yes" and "No"
    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)
    yes_id = yes_tokens[0] if yes_tokens else tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = no_tokens[0] if no_tokens else tokenizer.encode("No", add_special_tokens=False)[0]

    results = []
    total = len(candidates)

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = candidates.iloc[start_idx:end_idx]

        prompts = []
        pair_ids = []
        for _, row in batch.iterrows():
            s1_id = str(row.iloc[0])
            s2_id = str(row.iloc[1])

            if s1_id in entities and s2_id in entities:
                prompt = format_pair_prompt(entities[s1_id], entities[s2_id])
                prompts.append(prompt)
                pair_ids.append((s1_id, s2_id))

        if not prompts:
            continue

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # last token logits

            # Get P(Yes) vs P(No)
            yes_logits = logits[:, yes_id]
            no_logits = logits[:, no_id]

            # Softmax over Yes/No
            pair_logits = torch.stack([no_logits, yes_logits], dim=-1)
            probs = torch.softmax(pair_logits, dim=-1)
            yes_probs = probs[:, 1].cpu().numpy()

        for (s1_id, s2_id), score in zip(pair_ids, yes_probs):
            results.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": s2_id,
                "score": float(score),
            })

        if (start_idx // batch_size) % 20 == 0:
            log.info("Inference: %d / %d pairs (%.1f%%)", start_idx, total, 100 * start_idx / total)

    log.info("Inference complete: %d pairs scored", len(results))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Full QLoRA Pipeline
# ---------------------------------------------------------------------------

def run_qlora_er_pipeline(
    dataset_dir: str = "dataset",
    output_dir: str = "output",
    model_name: str = "Qwen/Qwen2-0.5B",
    max_train_samples: int = 20000,
    max_rows: int = -1,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict:
    """
    Full QLoRA ER pipeline:
    1. Load data
    2. Blocking (reuse classical pipeline)
    3. Prepare training pairs
    4. QLoRA fine-tune
    5. Score all candidates
    6. Threshold + output
    """
    from src.er.pipeline import (
        load_sources, load_ground_truth, normalize_df,
        generate_candidates, compute_blocking_recall,
        assign_labels, macro_f05_score,
    )
    from src.utils.timing import Timer

    timer = Timer()
    timer.start("total")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    with timer.section("load"):
        train_sources = load_sources(dataset_dir, "train", max_rows=max_rows)
        test_sources = load_sources(dataset_dir, "test", max_rows=max_rows)
        ground_truth = load_ground_truth(dataset_dir)

    # ── Normalize ─────────────────────────────────────────────────────────
    with timer.section("normalize"):
        for name in train_sources:
            train_sources[name] = normalize_df(train_sources[name])
        for name in test_sources:
            test_sources[name] = normalize_df(test_sources[name])

    # ── Entity lookup ─────────────────────────────────────────────────────
    entities = {}
    for name, df in {**train_sources, **test_sources}.items():
        for _, row in df.iterrows():
            entities[str(row["entity_id"])] = row.to_dict()

    # ── Train blocking ────────────────────────────────────────────────────
    with timer.section("train_blocking"):
        s1_train = train_sources["source1"]
        other_train = {k: v for k, v in train_sources.items() if k != "source1"}
        train_candidates = generate_candidates(s1_train, other_train, 0, 1)

    blocking_metrics = compute_blocking_recall(train_candidates, ground_truth)
    log.info("Blocking recall: %.4f", blocking_metrics["blocking_recall"])

    # ── Prepare QLoRA training data ───────────────────────────────────────
    with timer.section("prepare_data"):
        training_data = prepare_training_data(
            train_candidates, entities, ground_truth,
            max_samples=max_train_samples, seed=seed,
        )

    # ── QLoRA Training ────────────────────────────────────────────────────
    with timer.section("qlora_training"):
        model, tokenizer = train_qlora_er(
            training_data,
            model_name=model_name,
            output_dir=str(Path(output_dir) / "qlora_model"),
            seed=seed,
        )

    # ── Test blocking ─────────────────────────────────────────────────────
    with timer.section("test_blocking"):
        s1_test = test_sources.get("source1")
        other_test = {k: v for k, v in test_sources.items() if k != "source1"}
        test_candidates = generate_candidates(s1_test, other_test, 0, 1)

    # Save candidate_pairs.tsv
    test_candidates.to_csv(Path(output_dir) / "candidate_pairs.tsv", sep="\t", index=False)

    # ── QLoRA Inference ───────────────────────────────────────────────────
    with timer.section("qlora_inference"):
        scored = predict_qlora_er(model, tokenizer, test_candidates, entities)

    # ── Threshold + Output ────────────────────────────────────────────────
    with timer.section("output"):
        test_s1_ids = s1_test["entity_id"].unique().tolist()
        matched = scored[scored["score"] >= threshold]

        rows = []
        for s1_id in test_s1_ids:
            s1_matches = matched[matched["source1_entity_id"] == s1_id]
            match_ids = s1_matches["candidate_entity_id"].tolist()
            rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(match_ids) if match_ids else "",
            })

        result_df = pd.DataFrame(rows)
        result_df.to_csv(Path(output_dir) / "matching_results.tsv", sep="\t", index=False)

    timer.stop("total")

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = {
        **blocking_metrics,
        "train_candidates": len(train_candidates),
        "test_candidates": len(test_candidates),
        "test_s1_entities": len(test_s1_ids),
        "threshold": threshold,
        "scored_pairs": len(scored),
        "matched_pairs": len(matched),
        "runtime": timer.summary(),
    }

    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("QLoRA ER pipeline complete")
    return metrics
