"""Test baseline models."""
import pytest
import numpy as np
import pandas as pd
from src.config.loader import WorkerConfig
from src.models.baseline import BaselineModel, train_baseline


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "id": range(n),
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.uniform(0, 1, n),
        "target": np.random.randint(0, 3, n),
    })


def test_logistic(training_data):
    model = BaselineModel("logistic")
    X = training_data[["f1", "f2", "f3"]].values
    y = training_data["target"].values
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert model.train_time > 0


def test_random_forest(training_data):
    model = BaselineModel("random_forest")
    X = training_data[["f1", "f2", "f3"]].values
    y = training_data["target"].values
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_lightgbm_or_fallback(training_data):
    """LightGBM if installed, else falls back to RF."""
    model = BaselineModel("lightgbm")
    X = training_data[["f1", "f2", "f3"]].values
    y = training_data["target"].values
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_predict_proba(training_data):
    model = BaselineModel("logistic")
    X = training_data[["f1", "f2", "f3"]].values
    y = training_data["target"].values
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba is not None
    assert proba.shape[0] == len(y)


def test_save_load(training_data, tmp_path):
    model = BaselineModel("logistic")
    X = training_data[["f1", "f2", "f3"]].values
    y = training_data["target"].values
    model.fit(X, y)

    path = str(tmp_path / "model.joblib")
    model.save(path)

    loaded = BaselineModel.load(path)
    preds = loaded.predict(X)
    assert len(preds) == len(y)


def test_train_baseline_fn(training_data):
    config = WorkerConfig(model_type="logistic")
    model, preds = train_baseline(training_data, config)
    assert len(preds) == len(training_data)


def test_string_labels():
    X = np.random.randn(100, 3)
    y = np.array(["cat", "dog", "bird"] * 33 + ["cat"])
    model = BaselineModel("logistic")
    model.fit(X, y)
    preds = model.predict(X)
    assert all(p in ("cat", "dog", "bird") for p in preds)
