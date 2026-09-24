"""
Amazon ML Worker — Evaluation Metrics.

Configurable metrics: F1 variants, precision, recall, confusion matrix.
Metric selection is configuration-driven — never assumes the competition metric.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logging import get_logger

log = get_logger(__name__)


# Registry of supported metrics
METRIC_REGISTRY = {
    "accuracy": lambda y, p: accuracy_score(y, p),
    "f1_macro": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
    "f1_micro": lambda y, p: f1_score(y, p, average="micro", zero_division=0),
    "f1_weighted": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
    "f1_binary": lambda y, p: f1_score(y, p, average="binary", zero_division=0),
    "precision_macro": lambda y, p: precision_score(y, p, average="macro", zero_division=0),
    "precision_micro": lambda y, p: precision_score(y, p, average="micro", zero_division=0),
    "recall_macro": lambda y, p: recall_score(y, p, average="macro", zero_division=0),
    "recall_micro": lambda y, p: recall_score(y, p, average="micro", zero_division=0),
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1_macro",
    include_all: bool = True,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric: Primary metric name (must be in METRIC_REGISTRY)
        include_all: If True, compute all available metrics

    Returns:
        Dict of metric_name → value
    """
    results: Dict[str, float] = {}

    if include_all:
        for name, fn in METRIC_REGISTRY.items():
            try:
                results[name] = round(float(fn(y_true, y_pred)), 6)
            except Exception:
                pass  # Skip metrics that don't apply (e.g., binary F1 for multiclass)
    else:
        if metric in METRIC_REGISTRY:
            results[metric] = round(float(METRIC_REGISTRY[metric](y_true, y_pred)), 6)
        else:
            log.warning("Unknown metric '%s'. Computing accuracy.", metric)
            results["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)

    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute confusion matrix and return as dict."""
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(list(y_true) + list(y_pred)))
    return {
        "matrix": cm.tolist(),
        "labels": [str(l) for l in labels],
    }


def format_metrics_report(
    metrics: Dict[str, float],
    primary_metric: str = "f1_macro",
) -> str:
    """Format metrics as a human-readable string."""
    lines = ["=" * 40, "  Evaluation Results", "=" * 40]
    for name, value in sorted(metrics.items()):
        marker = " <-- primary" if name == primary_metric else ""
        lines.append(f"  {name}: {value:.6f}{marker}")
    lines.append("=" * 40)
    return "\n".join(lines)
