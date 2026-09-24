"""Test batch prediction."""
import pytest
import numpy as np
import pandas as pd
from src.models.baseline import BaselineModel
from src.inference.predict import batch_predict, validate_predictions


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 3, 100)
    model = BaselineModel("logistic")
    model.fit(X, y)
    return model


@pytest.fixture
def test_df():
    np.random.seed(42)
    return pd.DataFrame({
        "id": [f"test_{i}" for i in range(50)],
        "f1": np.random.randn(50),
        "f2": np.random.randn(50),
        "f3": np.random.randn(50),
    })


def test_batch_predict(trained_model, test_df):
    result = batch_predict(test_df, trained_model, feature_columns=["f1", "f2", "f3"])
    assert len(result) == 50
    assert "id" in result.columns
    assert "prediction" in result.columns
    assert result["id"].duplicated().sum() == 0
    assert result["prediction"].isnull().sum() == 0


def test_batch_predict_save(trained_model, test_df, tmp_path):
    out = str(tmp_path / "preds.parquet")
    result = batch_predict(test_df, trained_model, feature_columns=["f1", "f2", "f3"], output_file=out)
    loaded = pd.read_parquet(out)
    assert len(loaded) == 50


def test_validate_predictions():
    original = pd.DataFrame({"id": [1, 2, 3]})
    preds = pd.DataFrame({"id": [1, 2, 3], "prediction": [0, 1, 2]})
    report = validate_predictions(preds, original)
    assert report["valid"] is True


def test_validate_missing_ids():
    original = pd.DataFrame({"id": [1, 2, 3, 4]})
    preds = pd.DataFrame({"id": [1, 2], "prediction": [0, 1]})
    report = validate_predictions(preds, original)
    assert report["valid"] is False
