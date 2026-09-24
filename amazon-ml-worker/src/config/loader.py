"""
Amazon ML Worker — Configuration Loader.

Hierarchical config: default.yaml → tasks/{task}.yaml → worker.yaml → CLI args → env vars.
"""

import argparse
import copy
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclass – canonical worker configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """Complete configuration for a single worker run."""

    # ── Identity ──────────────────────────────────────────────────────────
    worker_id: int = 0
    total_workers: int = 1
    task: str = "etl"          # etl | ocr | cv | nlp | embeddings | baseline | inference
    subtask: str = ""

    # ── Paths ─────────────────────────────────────────────────────────────
    input_path: str = "data/raw"
    output_path: str = "outputs/worker_results"
    cache_path: str = "cache"
    data_path: str = "data"
    features_path: str = "features"
    experiments_path: str = "experiments"

    # ── Seed / reproducibility ────────────────────────────────────────────
    seed: int = 42

    # ── Module toggles ────────────────────────────────────────────────────
    enable_ocr: bool = False
    enable_cv: bool = False
    enable_nlp: bool = False
    enable_embeddings: bool = False
    enable_baseline: bool = False
    enable_lightweight_model: bool = False
    enable_qlora_smoke_test: bool = False

    # ── Resource limits ───────────────────────────────────────────────────
    max_rows: int = -1          # -1 = unlimited
    max_steps: int = -1
    max_runtime: int = -1       # seconds, -1 = unlimited
    batch_size: int = 32
    num_workers: int = 0        # dataloader workers

    # ── Schema (set on Day-1) ─────────────────────────────────────────────
    id_column: str = "id"
    target_column: str = "target"
    group_column: str = ""
    text_column: str = "text"
    image_column: str = "image"
    metric: str = "f1_macro"

    # ── Split ─────────────────────────────────────────────────────────────
    split_method: str = "stratified"   # random | stratified | group
    val_fraction: float = 0.2

    # ── Model ─────────────────────────────────────────────────────────────
    model_type: str = "lightgbm"       # logistic | random_forest | lightgbm | xgboost | catboost
    model_params: Dict[str, Any] = field(default_factory=dict)

    # ── OCR ───────────────────────────────────────────────────────────────
    ocr_backend: str = "tesseract"     # tesseract | paddleocr

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = ""
    embedding_dim: int = 384

    # ── QLoRA smoke ───────────────────────────────────────────────────────
    qlora_model_name: str = "Qwen/Qwen2-0.5B"
    qlora_bits: int = 4
    qlora_max_steps: int = 5
    qlora_max_seq_len: int = 128

    # ── Misc ──────────────────────────────────────────────────────────────
    verbose: bool = True
    dry_run: bool = False
    resume: bool = True            # resume from cache

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from CWD looking for pyproject.toml."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def load_config(
    config_dir: Optional[str] = None,
    worker_yaml: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
) -> WorkerConfig:
    """
    Build a WorkerConfig by layering:
      1. defaults from WorkerConfig dataclass
      2. configs/default.yaml
      3. configs/tasks/{task}.yaml
      4. worker.yaml  (or custom path)
      5. CLI arguments
      6. Environment variable overrides (WORKER_ID, TOTAL_WORKERS, WORKER_TASK)
    """
    root = _find_project_root()
    cfg_dir = Path(config_dir) if config_dir else root / "configs"

    # 1. Defaults — from dataclass
    base = WorkerConfig().to_dict()

    # 2. default.yaml
    base = _deep_merge(base, _load_yaml(cfg_dir / "default.yaml"))

    # 3. Parse CLI early to find --task for task-yaml lookup
    parser = _build_parser()
    parsed, _ = parser.parse_known_args(cli_args)
    cli_dict = {k: v for k, v in vars(parsed).items() if v is not None}

    # Determine task
    task = cli_dict.get("task", base.get("task", "etl"))

    # 4. tasks/{task}.yaml
    base = _deep_merge(base, _load_yaml(cfg_dir / "tasks" / f"{task}.yaml"))

    # 5. worker.yaml
    wpath = Path(worker_yaml) if worker_yaml else cfg_dir / "worker.yaml"
    base = _deep_merge(base, _load_yaml(wpath))

    # 6. CLI overrides
    base = _deep_merge(base, cli_dict)

    # 7. Environment variable overrides
    env_overrides = _env_overrides()
    base = _deep_merge(base, env_overrides)

    # Build dataclass
    # Filter to only known fields
    known_fields = {f.name for f in WorkerConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in base.items() if k in known_fields}

    return WorkerConfig(**filtered)


def _env_overrides() -> dict:
    """Read environment variable overrides."""
    overrides: Dict[str, Any] = {}
    mapping = {
        "WORKER_ID": ("worker_id", int),
        "TOTAL_WORKERS": ("total_workers", int),
        "WORKER_TASK": ("task", str),
    }
    for env_key, (cfg_key, cast) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            overrides[cfg_key] = cast(val)
    return overrides


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Amazon ML Worker", add_help=False)

    p.add_argument("--worker-id", dest="worker_id", type=int, default=None)
    p.add_argument("--total-workers", dest="total_workers", type=int, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--subtask", type=str, default=None)

    p.add_argument("--input-path", dest="input_path", type=str, default=None)
    p.add_argument("--output-path", dest="output_path", type=str, default=None)
    p.add_argument("--cache-path", dest="cache_path", type=str, default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    p.add_argument("--max-runtime", dest="max_runtime", type=int, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)

    p.add_argument("--id-column", dest="id_column", type=str, default=None)
    p.add_argument("--target-column", dest="target_column", type=str, default=None)
    p.add_argument("--group-column", dest="group_column", type=str, default=None)
    p.add_argument("--text-column", dest="text_column", type=str, default=None)
    p.add_argument("--image-column", dest="image_column", type=str, default=None)
    p.add_argument("--metric", type=str, default=None)

    p.add_argument("--split-method", dest="split_method", type=str, default=None)
    p.add_argument("--val-fraction", dest="val_fraction", type=float, default=None)

    p.add_argument("--model-type", dest="model_type", type=str, default=None)

    p.add_argument("--ocr-backend", dest="ocr_backend", type=str, default=None)

    p.add_argument("--enable-ocr", dest="enable_ocr", action="store_true", default=None)
    p.add_argument("--enable-cv", dest="enable_cv", action="store_true", default=None)
    p.add_argument("--enable-nlp", dest="enable_nlp", action="store_true", default=None)
    p.add_argument("--enable-embeddings", dest="enable_embeddings", action="store_true", default=None)
    p.add_argument("--enable-baseline", dest="enable_baseline", action="store_true", default=None)

    p.add_argument("--verbose", action="store_true", default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=None)

    p.add_argument("--config-dir", dest="config_dir", type=str, default=None)
    p.add_argument("--worker-yaml", dest="worker_yaml", type=str, default=None)

    return p


def load_config_from_cli() -> WorkerConfig:
    """Load configuration from actual sys.argv CLI arguments."""
    import sys
    parser = _build_parser()
    parsed, _ = parser.parse_known_args()
    return load_config(
        config_dir=parsed.config_dir,
        worker_yaml=parsed.worker_yaml,
        cli_args=sys.argv[1:],
    )
