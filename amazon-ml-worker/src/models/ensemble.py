"""
Amazon ML Worker - Ensemble Utilities.

Combines predictions from multiple models with:
  - ID alignment verification
  - Duplicate/missing-ID detection
  - Simple averaging
  - Weighted averaging
  - Optional classification threshold search
  - Individual-model vs ensemble comparison
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.threshold import search_threshold
from src.utils.logging import get_logger

log = get_logger(__name__)


def _validate_id_alignment(
    dfs: List[pd.DataFrame],
    id_column: str = "id",
) -> Dict[str, Any]:
    """
    Verify that every prediction DataFrame shares the same set of IDs.
    """
    report: Dict[str, Any] = {"valid": True, "details": []}

    if not dfs:
        report["valid"] = False
        report["details"].append("No prediction DataFrames provided.")
        return report

    reference_ids = set(dfs[0][id_column])

    for i, df in enumerate(dfs):
        if id_column not in df.columns:
            report["valid"] = False
            detail = "Prediction set #" + str(i) + " missing ID column."
            report["details"].append(detail)
            continue

        current_ids = set(df[id_column])

        n_dup = df[id_column].duplicated().sum()
        if n_dup > 0:
            report["valid"] = False
            detail = "Prediction set #" + str(i) + " has " + str(n_dup) + " duplicate IDs."
            report["details"].append(detail)

        missing = reference_ids - current_ids
        if missing:
            report["valid"] = False
            detail = "Prediction set #" + str(i) + " missing " + str(len(missing)) + " IDs vs set #0."
            report["details"].append(detail)

        extra = current_ids - reference_ids
        if extra:
            report["valid"] = False
            detail = "Prediction set #" + str(i) + " has " + str(len(extra)) + " extra IDs vs set #0."
            report["details"].append(detail)

    if report["valid"]:
        log.info("ID alignment check PASSED across %d prediction sets.", len(dfs))
    else:
        log.error("ID alignment check FAILED: %s", report["details"])

    return report


def load_predictions(
    paths: List[str],
    id_column: str = "id",
    pred_column: str = "prediction",
) -> List[pd.DataFrame]:
    """Load prediction files (CSV or Parquet)."""
    dfs: List[pd.DataFrame] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            log.warning("Prediction file not found: %s - skipping.", p)
            continue
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if id_column not in df.columns:
            log.warning("File %s missing ID column - skipping.", p)
            continue
        if pred_column not in df.columns:
            log.warning("File %s missing prediction column - skipping.", p)
            continue

        dfs.append(df[[id_column, pred_column]])

    log.info("Loaded %d / %d prediction files.", len(dfs), len(paths))
    return dfs


def average_predictions(
    dfs: List[pd.DataFrame],
    id_column: str = "id",
    pred_column: str = "prediction",
) -> pd.DataFrame:
    """Compute simple arithmetic mean of predictions across models."""
    alignment = _validate_id_alignment(dfs, id_column)
    if not alignment["valid"]:
        details = str(alignment["details"])
        raise ValueError("Cannot average: ID alignment failed - " + details)

    sorted_dfs = [df.sort_values(id_column).reset_index(drop=True) for df in dfs]
    ids = sorted_dfs[0][id_column]

    pred_matrix = np.column_stack([df[pred_column].values for df in sorted_dfs])
    avg_preds = pred_matrix.mean(axis=1)

    result = pd.DataFrame({id_column: ids, pred_column: avg_preds})
    log.info("Averaged %d prediction sets -> %d rows.", len(dfs), len(result))
    return result


def weighted_average_predictions(
    dfs: List[pd.DataFrame],
    weights: List[float],
    id_column: str = "id",
    pred_column: str = "prediction",
) -> pd.DataFrame:
    """Compute weighted average of predictions. Weights are normalised internally."""
    if len(dfs) != len(weights):
        msg = "Number of DataFrames (" + str(len(dfs)) + ") must match number of weights (" + str(len(weights)) + ")."
        raise ValueError(msg)

    alignment = _validate_id_alignment(dfs, id_column)
    if not alignment["valid"]:
        details = str(alignment["details"])
        raise ValueError("Cannot blend: ID alignment failed - " + details)

    sorted_dfs = [df.sort_values(id_column).reset_index(drop=True) for df in dfs]
    ids = sorted_dfs[0][id_column]

    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    pred_matrix = np.column_stack([df[pred_column].values for df in sorted_dfs])
    blended = (pred_matrix * w[np.newaxis, :]).sum(axis=1)

    result = pd.DataFrame({id_column: ids, pred_column: blended})
    log.info(
        "Weighted-averaged %d prediction sets (weights=%s) -> %d rows.",
        len(dfs), [round(x, 4) for x in w.tolist()], len(result),
    )
    return result


def ensemble_with_threshold(
    dfs: List[pd.DataFrame],
    y_true: np.ndarray,
    weights: Optional[List[float]] = None,
    id_column: str = "id",
    pred_column: str = "prediction",
    n_thresholds: int = 100,
) -> Dict[str, Any]:
    """Blend predictions, then search for the optimal classification threshold."""
    if weights is not None:
        blended = weighted_average_predictions(dfs, weights, id_column, pred_column)
    else:
        blended = average_predictions(dfs, id_column, pred_column)

    threshold_result = search_threshold(
        y_true,
        blended[pred_column].values,
        n_thresholds=n_thresholds,
    )

    return {
        "ensemble_df": blended,
        "best_threshold": threshold_result["best_threshold"],
        "best_score": threshold_result["best_score"],
    }


def compare_individual_vs_ensemble(
    dfs: List[pd.DataFrame],
    y_true: np.ndarray,
    metric: str = "f1_macro",
    weights: Optional[List[float]] = None,
    id_column: str = "id",
    pred_column: str = "prediction",
    classify: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate each individual prediction set and the ensemble, then compare."""
    alignment = _validate_id_alignment(dfs, id_column)
    if not alignment["valid"]:
        raise ValueError("Cannot compare: " + str(alignment["details"]))

    individual_results: List[Dict[str, Any]] = []

    for i, df in enumerate(dfs):
        sorted_df = df.sort_values(id_column).reset_index(drop=True)
        preds = sorted_df[pred_column].values

        if classify:
            if len(np.unique(y_true)) == 2:
                preds_cls = (preds >= threshold).astype(int)
            else:
                preds_cls = np.round(preds).astype(int)
        else:
            preds_cls = preds

        metrics_i = compute_metrics(y_true, preds_cls, metric=metric, include_all=False)
        individual_results.append({
            "index": i,
            "metric_name": metric,
            "metric_value": metrics_i.get(metric, 0.0),
        })
        log.info("  Model #%d: %s = %.6f", i, metric, metrics_i.get(metric, 0.0))

    # Ensemble
    if weights is not None:
        ens_df = weighted_average_predictions(dfs, weights, id_column, pred_column)
    else:
        ens_df = average_predictions(dfs, id_column, pred_column)

    ens_sorted = ens_df.sort_values(id_column).reset_index(drop=True)
    ens_preds = ens_sorted[pred_column].values

    if classify:
        if len(np.unique(y_true)) == 2:
            ens_preds_cls = (ens_preds >= threshold).astype(int)
        else:
            ens_preds_cls = np.round(ens_preds).astype(int)
    else:
        ens_preds_cls = ens_preds

    ens_metrics = compute_metrics(y_true, ens_preds_cls, metric=metric, include_all=False)
    ens_value = ens_metrics.get(metric, 0.0)
    log.info("  Ensemble: %s = %.6f", metric, ens_value)

    best_individual = max(r["metric_value"] for r in individual_results)
    improves = ens_value > best_individual

    if improves:
        log.info("  Ensemble IMPROVES over best individual (%.6f > %.6f).", ens_value, best_individual)
    else:
        log.info("  Ensemble does NOT improve over best individual (%.6f <= %.6f).", ens_value, best_individual)

    return {
        "individual": individual_results,
        "ensemble": {"metric_name": metric, "metric_value": ens_value},
        "ensemble_improves": improves,
        "best_individual_value": best_individual,
    }
