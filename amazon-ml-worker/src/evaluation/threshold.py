"""
Amazon ML Worker — Threshold Search.

Search for optimal decision thresholds for binary/multi-class classification.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

from src.utils.logging import get_logger

log = get_logger(__name__)


def search_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1_binary",
    n_thresholds: int = 100,
) -> Dict[str, float]:
    """
    Search for the best threshold on probability predictions.

    Args:
        y_true: Ground truth binary labels (0/1)
        y_proba: Predicted probabilities for the positive class
        metric: Metric to optimize
        n_thresholds: Number of thresholds to try

    Returns:
        {"best_threshold": float, "best_score": float}
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -1.0
    best_threshold = 0.5

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        try:
            score = f1_score(y_true, preds, average="binary", zero_division=0)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_threshold = thresh

    log.info("Best threshold: %.4f (score=%.6f)", best_threshold, best_score)
    return {
        "best_threshold": round(float(best_threshold), 4),
        "best_score": round(float(best_score), 6),
    }
