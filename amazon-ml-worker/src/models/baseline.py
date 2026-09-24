"""
Amazon ML Worker — Baseline Models.

Configurable lightweight baselines: Logistic Regression, Random Forest,
LightGBM, XGBoost, CatBoost. Only imports requested model libraries.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config.loader import WorkerConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


class BaselineModel:
    """Wrapper around configurable baseline classifiers/regressors."""

    def __init__(self, model_type: str = "lightgbm", model_params: Optional[Dict] = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._train_time: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "BaselineModel":
        """Train the model."""
        start = time.time()

        # Encode string labels if needed
        if y.dtype == object or y.dtype.kind in ("U", "S"):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        self.model = self._create_model()
        self.model.fit(X, y)
        self._train_time = time.time() - start

        log.info(
            "Trained %s in %.2f seconds.",
            self.model_type, self._train_time,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        preds = self.model.predict(X)
        if self.label_encoder is not None:
            preds = self.label_encoder.inverse_transform(preds.astype(int))
        return preds

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Generate probability predictions (if supported)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "label_encoder": self.label_encoder,
        }
        joblib.dump(data, path)
        log.info("Saved model: %s", path)

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(
            model_type=data["model_type"],
            model_params=data.get("model_params", {}),
        )
        instance.model = data["model"]
        instance.label_encoder = data.get("label_encoder")
        return instance

    @property
    def train_time(self) -> float:
        return self._train_time

    def _create_model(self):
        """Factory: instantiate the requested model."""
        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                **self.model_params,
            )

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                **self.model_params,
            )

        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    verbosity=-1,
                    **self.model_params,
                )
            except ImportError:
                log.warning("LightGBM not installed; falling back to Random Forest.")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                )

        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    **self.model_params,
                )
            except ImportError:
                log.warning("XGBoost not installed; falling back to Random Forest.")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        elif self.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=100,
                    random_state=42,
                    verbose=0,
                    **self.model_params,
                )
            except ImportError:
                log.warning("CatBoost not installed; falling back to Random Forest.")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


def train_baseline(
    train_df: pd.DataFrame,
    config: WorkerConfig,
    feature_columns: Optional[list] = None,
) -> Tuple[BaselineModel, np.ndarray]:
    """
    Train a baseline model using configuration.

    Returns (model, predictions_on_train).
    """
    if config.target_column not in train_df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found.")

    y = train_df[config.target_column].values
    if feature_columns is None:
        feature_columns = [
            c for c in train_df.columns
            if c not in (config.id_column, config.target_column)
            and train_df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found for training.")

    X = train_df[feature_columns].values.astype(np.float32)

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    model = BaselineModel(
        model_type=config.model_type,
        model_params=config.model_params,
    )
    model.fit(X, y)
    preds = model.predict(X)

    return model, preds
