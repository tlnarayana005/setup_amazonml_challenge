"""
Amazon ML Worker — Batch Inference.

CPU/GPU batch inference with stable IDs, output validation, and runtime tracking.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.baseline import BaselineModel
from src.utils.logging import get_logger
from src.utils.timing import Timer

log = get_logger(__name__)


def batch_predict(
    df: pd.DataFrame,
    model: BaselineModel,
    id_column: str = "id",
    feature_columns: Optional[List[str]] = None,
    batch_size: int = 1024,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run batch inference, preserving ID order.

    Returns DataFrame with [id, prediction] columns.
    """
    timer = Timer()
    timer.start("inference")

    if feature_columns is None:
        feature_columns = [
            c for c in df.columns
            if c != id_column
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        ]

    ids = df[id_column].values
    X = df[feature_columns].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    # Batch prediction
    all_preds: List[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch = X[start:end]
        preds = model.predict(batch)
        all_preds.append(preds)

    predictions = np.concatenate(all_preds)

    # Build result
    result = pd.DataFrame({
        id_column: ids,
        "prediction": predictions,
    })

    # Validation
    assert len(result) == len(df), f"Row count mismatch: {len(result)} vs {len(df)}"
    assert result[id_column].duplicated().sum() == 0, "Duplicate IDs in predictions"
    assert result["prediction"].isnull().sum() == 0, "Null predictions found"

    timer.stop("inference")
    log.info(
        "Inference complete: %d predictions in %.2f seconds.",
        len(result), timer.elapsed("inference"),
    )

    # Save
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        log.info("Saved predictions: %s", out_path)

    return result


def validate_predictions(
    predictions: pd.DataFrame,
    original_df: pd.DataFrame,
    id_column: str = "id",
) -> Dict[str, Any]:
    """Validate prediction output against the original data."""
    report = {"valid": True, "details": []}

    # Row count
    if len(predictions) != len(original_df):
        report["valid"] = False
        report["details"].append(
            f"Row count: predictions={len(predictions)}, original={len(original_df)}"
        )

    # Missing IDs
    if id_column in predictions.columns and id_column in original_df.columns:
        pred_ids = set(predictions[id_column])
        orig_ids = set(original_df[id_column])
        missing = orig_ids - pred_ids
        if missing:
            report["valid"] = False
            report["details"].append(f"Missing {len(missing)} IDs in predictions")

    # Duplicates
    n_dup = predictions[id_column].duplicated().sum() if id_column in predictions.columns else 0
    if n_dup > 0:
        report["valid"] = False
        report["details"].append(f"{n_dup} duplicate IDs in predictions")

    # Null predictions
    n_null = predictions["prediction"].isnull().sum() if "prediction" in predictions.columns else -1
    if n_null > 0:
        report["valid"] = False
        report["details"].append(f"{n_null} null predictions")

    if report["valid"]:
        log.info("Prediction validation PASSED.")
    else:
        log.error("Prediction validation FAILED: %s", report["details"])

    return report
