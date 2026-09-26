"""
Amazon ML Worker — Main Worker Entry Point.

Usage:
    python scripts/run_worker.py --task ocr --worker-id 0 --total-workers 8
    python scripts/run_worker.py --task etl
    python scripts/run_worker.py --task baseline --enable-baseline
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config_from_cli
from src.utils.logging import setup_logging, get_logger
from src.utils.seeds import set_seed
from src.utils.timing import Timer
from src.utils.paths import worker_output_dir, resolve_path


def main():
    config = load_config_from_cli()
    setup_logging(worker_id=config.worker_id)
    log = get_logger("worker")
    set_seed(config.seed)

    timer = Timer()
    timer.start("total")

    # ── Resource safety: print workload summary ──────────────────────────
    log.info("=" * 60)
    log.info("  Amazon ML Worker")
    log.info("=" * 60)
    log.info("  task:          %s", config.task)
    log.info("  worker_id:     %d / %d", config.worker_id, config.total_workers)
    log.info("  seed:          %d", config.seed)
    log.info("  max_rows:      %s", config.max_rows if config.max_rows > 0 else "unlimited")
    log.info("  batch_size:    %d", config.batch_size)
    log.info("  input_path:    %s", config.input_path)
    log.info("  output_path:   %s", config.output_path)
    log.info("  cache_path:    %s", config.cache_path)
    log.info("  resume:        %s", config.resume)
    log.info("  dry_run:       %s", config.dry_run)
    log.info("=" * 60)

    if config.dry_run:
        log.info("DRY RUN — exiting without processing.")
        return

    # ── Create output directory ──────────────────────────────────────────
    out_dir = worker_output_dir(config.output_path, config.worker_id)

    # ── Save worker config ───────────────────────────────────────────────
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    # ── Dispatch by task ─────────────────────────────────────────────────
    try:
        if config.task == "etl":
            _run_etl(config, out_dir, log)
        elif config.task == "ocr":
            _run_ocr(config, out_dir, log)
        elif config.task == "cv":
            _run_cv(config, out_dir, log)
        elif config.task == "nlp":
            _run_nlp(config, out_dir, log)
        elif config.task == "embeddings":
            _run_embeddings(config, out_dir, log)
        elif config.task == "baseline":
            _run_baseline(config, out_dir, log)
        elif config.task == "inference":
            _run_inference(config, out_dir, log)
        elif config.task == "entity_resolution":
            _run_entity_resolution(config, out_dir, log)
        else:
            log.error("Unknown task: %s", config.task)
            sys.exit(1)
    except Exception as e:
        log.error("Worker FAILED: %s", e, exc_info=True)
        # Save error info
        with open(out_dir / "errors.csv", "w", encoding="utf-8") as f:
            f.write("error\n")
            f.write(f"{str(e)}\n")
        raise

    # ── Save runtime info ────────────────────────────────────────────────
    timer.stop("total")
    runtime = timer.summary()
    with open(out_dir / "runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2, default=str)

    # ── Save README ──────────────────────────────────────────────────────
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(f"Worker {config.worker_id}/{config.total_workers}\n")
        f.write(f"Task: {config.task}\n")
        f.write(f"Runtime: {runtime.get('total_elapsed', 0):.2f}s\n")

    log.info("Worker %d complete. Output: %s", config.worker_id, out_dir)
    log.info("Total runtime: %.2f seconds", runtime.get("total_elapsed", 0))


def _run_etl(config, out_dir, log):
    from src.etl.pipeline import run_etl
    df = run_etl(config)
    df.to_parquet(out_dir / "output.parquet", index=False)
    _save_metrics(out_dir, {"rows_processed": len(df)})


def _run_ocr(config, out_dir, log):
    from src.data.shard import get_shard
    from src.etl.pipeline import extract
    from src.ocr.base import run_ocr_batch, get_ocr_backend

    df = extract(config)
    shard = get_shard(df, config.worker_id, config.total_workers, config.id_column)

    backend = get_ocr_backend(config.ocr_backend)
    result = run_ocr_batch(
        shard, backend,
        id_column=config.id_column,
        image_column=config.image_column,
        cache_dir=config.cache_path,
    )
    result.to_parquet(out_dir / "output.parquet", index=False)
    _save_metrics(out_dir, {"rows_processed": len(result)})


def _run_cv(config, out_dir, log):
    from src.data.shard import get_shard
    from src.etl.pipeline import extract
    from src.features.builder import build_features

    df = extract(config)
    shard = get_shard(df, config.worker_id, config.total_workers, config.id_column)
    features = build_features(shard, config)
    features.to_parquet(out_dir / "output.parquet", index=False)
    _save_metrics(out_dir, {"rows_processed": len(features)})


def _run_nlp(config, out_dir, log):
    from src.data.shard import get_shard
    from src.etl.pipeline import extract
    from src.nlp.text_utils import batch_clean_text, add_text_features

    df = extract(config)
    shard = get_shard(df, config.worker_id, config.total_workers, config.id_column)
    if config.text_column in shard.columns:
        shard = batch_clean_text(shard, config.text_column)
        shard = add_text_features(shard, config.text_column)
    shard.to_parquet(out_dir / "output.parquet", index=False)
    _save_metrics(out_dir, {"rows_processed": len(shard)})


def _run_embeddings(config, out_dir, log):
    from src.data.shard import get_shard
    from src.etl.pipeline import extract

    df = extract(config)
    shard = get_shard(df, config.worker_id, config.total_workers, config.id_column)
    # Text embeddings
    if config.enable_embeddings and config.text_column in shard.columns:
        from src.nlp.embeddings import extract_text_embeddings
        emb = extract_text_embeddings(
            shard, id_column=config.id_column, text_column=config.text_column,
            model_name=config.embedding_model, embedding_dim=config.embedding_dim,
            cache_dir=config.cache_path, resume=config.resume,
        )
        emb.to_parquet(out_dir / "output.parquet", index=False)
        _save_metrics(out_dir, {"rows_processed": len(emb)})
    else:
        shard[[config.id_column]].to_parquet(out_dir / "output.parquet", index=False)
        _save_metrics(out_dir, {"rows_processed": 0})


def _run_baseline(config, out_dir, log):
    from src.etl.pipeline import extract
    from src.data.split import create_split
    from src.features.builder import build_features
    from src.models.baseline import train_baseline
    from src.evaluation.metrics import compute_metrics, format_metrics_report

    df = extract(config)
    df = build_features(df, config)

    train_df, val_df = create_split(
        df, method=config.split_method,
        target_column=config.target_column,
        group_column=config.group_column,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    model, _ = train_baseline(train_df, config)

    # Evaluate on val
    feat_cols = [c for c in val_df.columns if c not in (config.id_column, config.target_column)
                 and val_df[c].dtype in ('float64', 'float32', 'int64', 'int32', float, int)]
    import numpy as np
    X_val = val_df[feat_cols].values.astype(np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0)
    val_preds = model.predict(X_val)

    metrics = compute_metrics(val_df[config.target_column].values, val_preds, config.metric)
    log.info("\n%s", format_metrics_report(metrics, config.metric))

    # Save
    model.save(str(out_dir / "model.joblib"))
    val_df.to_parquet(out_dir / "output.parquet", index=False)
    _save_metrics(out_dir, metrics)


def _run_inference(config, out_dir, log):
    log.info("Inference task — requires a trained model. Placeholder for Day-1.")
    import pandas as pd
    pd.DataFrame(columns=["id", "prediction"]).to_parquet(out_dir / "output.parquet", index=False)
    _save_metrics(out_dir, {"status": "placeholder"})


def _run_entity_resolution(config, out_dir, log):
    from src.er.pipeline import run_entity_resolution
    metrics = run_entity_resolution(
        dataset_dir=config.dataset_dir,
        output_dir=config.output_dir,
        model_type=config.model_type,
        model_params=config.model_params or {},
        worker_id=config.worker_id,
        total_workers=config.total_workers,
        seed=config.seed,
        max_rows=config.max_rows,
    )
    _save_metrics(out_dir, metrics)


def _save_metrics(out_dir, metrics):
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


if __name__ == "__main__":
    main()
