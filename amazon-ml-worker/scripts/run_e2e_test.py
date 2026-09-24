"""
End-to-end integration test.

Exercises the full pipeline: synthetic data -> 3-worker sharding -> merge -> baseline -> eval.
"""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging, get_logger
from src.utils.seeds import set_seed

def main():
    setup_logging()
    log = get_logger("e2e")
    set_seed(42)

    log.info("=" * 60)
    log.info("  END-TO-END INTEGRATION TEST")
    log.info("=" * 60)

    results = {}

    # 1. Generate synthetic data
    log.info("[1/12] Generating synthetic data...")
    from synthetic_data.generate import generate_synthetic_dataset
    train_df = generate_synthetic_dataset(n_rows=300, n_classes=3, create_images=True, image_count=20)
    assert len(train_df) == 240  # 80% of 300
    results["synthetic_data"] = "PASS"
    log.info("  PASS: %d train rows", len(train_df))

    # 2. Data inspection
    log.info("[2/12] Data inspection...")
    from src.data.inspect import inspect_dataframe
    report = inspect_dataframe(train_df)
    assert report["shape"]["rows"] == 240
    results["data_inspection"] = "PASS"
    log.info("  PASS: %d rows, %d columns", report["shape"]["rows"], report["shape"]["columns"])

    # 3. Data validation
    log.info("[3/12] Data validation...")
    from src.data.validate import validate_dataframe
    val_result = validate_dataframe(train_df)
    assert val_result["valid"]
    results["data_validation"] = "PASS"
    log.info("  PASS")

    # 4. Train/val split
    log.info("[4/12] Train/val split...")
    from src.data.split import create_split
    train_split, val_split = create_split(train_df, method="stratified", target_column="target", seed=42)
    assert len(train_split) + len(val_split) == len(train_df)
    results["split"] = "PASS"
    log.info("  PASS: train=%d, val=%d", len(train_split), len(val_split))

    # 5. Three-worker sharding simulation
    log.info("[5/12] Three-worker sharding...")
    from src.data.shard import get_shard, verify_sharding
    shard_result = verify_sharding(train_df, total_workers=3)
    assert shard_result["valid"]
    all_ids = set()
    for wid in range(3):
        shard = get_shard(train_df, wid, 3)
        shard_ids = set(shard["id"])
        overlap = all_ids & shard_ids
        assert len(overlap) == 0, f"Worker {wid} has overlapping IDs!"
        all_ids.update(shard_ids)
    assert all_ids == set(train_df["id"])
    results["sharding"] = "PASS"
    log.info("  PASS: 3 workers, no overlap, complete coverage")

    # 6. Worker processing + output saving
    log.info("[6/12] Worker processing...")
    import pandas as pd
    import numpy as np
    out_base = Path("outputs/e2e_test")
    if out_base.exists():
        shutil.rmtree(out_base)
    for wid in range(3):
        shard = get_shard(train_df, wid, 3)
        w_dir = out_base / f"worker_{wid}"
        w_dir.mkdir(parents=True)
        shard.to_parquet(w_dir / "output.parquet", index=False)
    results["worker_processing"] = "PASS"
    log.info("  PASS: 3 worker outputs saved")

    # 7. Merge
    log.info("[7/12] Merging worker outputs...")
    from src.data.merge import merge_worker_outputs, verify_merge
    merged = merge_worker_outputs(str(out_base))
    merge_report = verify_merge(merged, train_df)
    assert merge_report["valid"]
    results["merge"] = "PASS"
    log.info("  PASS: %d merged rows, no duplicates", len(merged))

    # 8. Cache test
    log.info("[8/12] Cache test...")
    from src.etl.cache import CacheManager
    cache = CacheManager(str(out_base / "cache"), "test_cache")
    df_partial = pd.DataFrame({"id": ["a", "b", "c"], "val": [1, 2, 3]})
    cache.save_to_cache(df_partial)
    full = pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "val": [1, 2, 3, 4, 5]})
    remaining = cache.filter_uncached(full)
    assert len(remaining) == 2  # d and e
    assert set(remaining["id"]) == {"d", "e"}
    results["cache"] = "PASS"
    log.info("  PASS: cached 3, skipped correctly, 2 remaining")

    # 9. Feature building
    log.info("[9/12] Feature building...")
    from src.config.loader import WorkerConfig
    from src.features.builder import build_features
    from src.features.merger import merge_features
    config = WorkerConfig(text_column="text")
    features = build_features(train_split, config)
    assert "id" in features.columns
    assert len(features) == len(train_split)
    results["features"] = "PASS"
    log.info("  PASS: %d features built", len(features.columns))

    # 10. NLP test
    log.info("[10/12] NLP text processing...")
    from src.nlp.text_utils import batch_clean_text, add_text_features, compute_text_stats
    cleaned = batch_clean_text(train_split, "text")
    assert "text_clean" in cleaned.columns
    with_feats = add_text_features(cleaned, "text")
    assert "text_word_count" in with_feats.columns
    stats = compute_text_stats(train_split["text"])
    assert stats["count"] > 0
    results["nlp"] = "PASS"
    log.info("  PASS")

    # 11. Baseline training + evaluation
    log.info("[11/12] Baseline training & evaluation...")
    from src.models.baseline import train_baseline
    from src.evaluation.metrics import compute_metrics, format_metrics_report
    config_bl = WorkerConfig(model_type="logistic", text_column="text")
    feat_train = build_features(train_split, config_bl)
    feat_val = build_features(val_split, config_bl)
    model, train_preds = train_baseline(feat_train, config_bl)
    feat_cols = [c for c in feat_val.columns if c not in ("id", "target") and feat_val[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]
    X_val = np.nan_to_num(feat_val[feat_cols].values.astype(np.float32), nan=0.0)
    val_preds = model.predict(X_val)
    metrics = compute_metrics(val_split["target"].values, val_preds, "f1_macro")
    log.info("\n%s", format_metrics_report(metrics, "f1_macro"))
    assert "f1_macro" in metrics
    assert metrics["accuracy"] > 0.0
    results["baseline"] = "PASS"
    log.info("  PASS: f1_macro=%.4f", metrics["f1_macro"])

    # 12. Prediction + experiment tracking
    log.info("[12/12] Prediction & experiment tracking...")
    from src.inference.predict import batch_predict, validate_predictions
    pred_result = batch_predict(feat_val, model, feature_columns=feat_cols)
    pred_report = validate_predictions(pred_result, feat_val)
    assert pred_report["valid"]
    from src.experiments.tracker import ExperimentTracker
    tracker = ExperimentTracker(str(out_base / "results.csv"))
    tracker.log_experiment(worker_id=0, task="e2e_test", model="logistic",
                            metric_name="f1_macro", metric_value=metrics["f1_macro"])
    exp_results = tracker.get_results()
    assert len(exp_results) >= 1
    results["prediction"] = "PASS"
    results["experiment_tracking"] = "PASS"
    log.info("  PASS")

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("  END-TO-END TEST RESULTS")
    log.info("=" * 60)
    all_pass = True
    for test_name, status in results.items():
        icon = "OK" if status == "PASS" else "FAIL"
        log.info("  [%s] %s", icon, test_name)
        if status != "PASS":
            all_pass = False
    log.info("=" * 60)

    if all_pass:
        log.info("  ALL TESTS PASSED!")
    else:
        log.error("  SOME TESTS FAILED!")
        sys.exit(1)

    # Cleanup
    shutil.rmtree(out_base, ignore_errors=True)
    log.info("  Cleaned up test artifacts.")


if __name__ == "__main__":
    main()
