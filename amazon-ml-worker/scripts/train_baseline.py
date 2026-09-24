"""Train a baseline model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from src.config.loader import load_config_from_cli
from src.etl.pipeline import extract
from src.data.split import create_split
from src.features.builder import build_features
from src.models.baseline import train_baseline
from src.evaluation.metrics import compute_metrics, format_metrics_report
from src.experiments.tracker import ExperimentTracker
from src.utils.logging import setup_logging, get_logger
from src.utils.seeds import set_seed
from src.utils.paths import ensure_dir

def main():
    config = load_config_from_cli()
    setup_logging(worker_id=config.worker_id)
    log = get_logger("train_baseline")
    set_seed(config.seed)

    df = extract(config)
    df = build_features(df, config)
    train_df, val_df = create_split(df, method=config.split_method, target_column=config.target_column,
                                     group_column=config.group_column, val_fraction=config.val_fraction, seed=config.seed)

    model, _ = train_baseline(train_df, config)

    feat_cols = [c for c in val_df.columns if c not in (config.id_column, config.target_column)
                 and val_df[c].dtype in ('float64','float32','int64','int32',float,int)]
    X_val = np.nan_to_num(val_df[feat_cols].values.astype(np.float32), nan=0.0)
    val_preds = model.predict(X_val)
    metrics = compute_metrics(val_df[config.target_column].values, val_preds, config.metric)
    log.info("\n%s", format_metrics_report(metrics, config.metric))

    out_dir = ensure_dir(config.output_path, f"baseline_{config.model_type}")
    model.save(str(out_dir / "model.joblib"))

    tracker = ExperimentTracker(f"{config.experiments_path}/results.csv")
    tracker.log_experiment(worker_id=config.worker_id, task="baseline", model=config.model_type,
                           metric_name=config.metric, metric_value=metrics.get(config.metric, 0),
                           runtime_seconds=model.train_time)
    print(f"\nBaseline complete. Primary metric ({config.metric}): {metrics.get(config.metric, 0):.6f}")

if __name__ == "__main__":
    main()
