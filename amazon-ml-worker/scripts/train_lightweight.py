"""Train lightweight screening experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.config.loader import load_config_from_cli
from src.etl.pipeline import extract
from src.data.split import create_split
from src.features.builder import build_features
from src.models.lightweight import LightweightExperiment
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed

def main():
    config = load_config_from_cli()
    setup_logging(worker_id=config.worker_id)
    set_seed(config.seed)

    df = extract(config)
    df = build_features(df, config)
    train_df, val_df = create_split(df, method=config.split_method, target_column=config.target_column,
                                     val_fraction=config.val_fraction, seed=config.seed)

    feat_cols = [c for c in train_df.columns if c not in (config.id_column, config.target_column)
                 and train_df[c].dtype in ('float64','float32','int64','int32',float,int)]
    X_train = np.nan_to_num(train_df[feat_cols].values.astype(np.float32), nan=0.0)
    y_train = train_df[config.target_column].values
    X_val = np.nan_to_num(val_df[feat_cols].values.astype(np.float32), nan=0.0)
    y_val = val_df[config.target_column].values

    exp = LightweightExperiment(model_type=config.model_type, max_rows=config.max_rows if config.max_rows > 0 else 5000,
                                 metric=config.metric, seed=config.seed)
    results = exp.run(X_train, y_train, X_val, y_val)
    print(f"\nScreening complete: val_{config.metric} = {results['val_metric'].get(config.metric, 0):.6f}")

if __name__ == "__main__":
    main()
