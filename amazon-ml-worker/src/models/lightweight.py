"""
Amazon ML Worker — Lightweight Experiment Screening.

Fast experiment framework for determining if a direction is promising
before committing expensive GPU resources.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.baseline import BaselineModel
from src.evaluation.metrics import compute_metrics
from src.utils.logging import get_logger
from src.utils.timing import Timer

log = get_logger(__name__)


class LightweightExperiment:
    """
    Runs a quick screening experiment with limited data and steps.

    Records training loss proxies, validation metrics, runtime.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        model_params: Optional[Dict] = None,
        max_rows: int = 5000,
        metric: str = "f1_macro",
        seed: int = 42,
    ):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.max_rows = max_rows
        self.metric = metric
        self.seed = seed
        self.results: Dict[str, Any] = {}

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Run the screening experiment."""
        timer = Timer()
        timer.start("experiment")

        # Limit data
        if self.max_rows > 0 and len(X_train) > self.max_rows:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(X_train), self.max_rows, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            log.info("Limited training to %d rows.", self.max_rows)

        # Train
        model = BaselineModel(self.model_type, self.model_params)
        model.fit(X_train, y_train)

        # Evaluate
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        train_metrics = compute_metrics(y_train, train_preds, metric=self.metric)
        val_metrics = compute_metrics(y_val, val_preds, metric=self.metric)

        timer.stop("experiment")

        self.results = {
            "model_type": self.model_type,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "train_metric": train_metrics,
            "val_metric": val_metrics,
            "train_time": model.train_time,
            "total_time": timer.elapsed("experiment"),
            "metric_name": self.metric,
        }

        log.info(
            "Experiment: %s | train_%s=%.4f | val_%s=%.4f | time=%.2fs",
            self.model_type,
            self.metric, train_metrics.get(self.metric, 0),
            self.metric, val_metrics.get(self.metric, 0),
            self.results["total_time"],
        )

        return self.results
