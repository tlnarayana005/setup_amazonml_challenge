"""Test data inspection and validation."""
import pytest
import numpy as np
import pandas as pd
from src.data.inspect import inspect_dataframe
from src.data.validate import validate_dataframe, check_leakage


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(100),
        "text": [f"sample text {i}" for i in range(100)],
        "category": np.random.choice(["a", "b", "c"], 100),
        "price": np.random.uniform(10, 500, 100),
        "target": np.random.randint(0, 3, 100),
    })


def test_inspect_dataframe(sample_df):
    report = inspect_dataframe(sample_df)
    assert report["shape"]["rows"] == 100
    assert report["shape"]["columns"] == 5
    assert "columns" in report
    assert "dtypes" in report
    assert "unique_counts" in report
    assert report["duplicate_rows"] == 0


def test_inspect_with_missing():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "text": ["a", None, "c", None, "e"],
        "value": [1.0, 2.0, None, 4.0, 5.0],
        "target": [0, 1, 0, 1, 0],
    })
    report = inspect_dataframe(df)
    assert "text" in report["missing_values"]
    assert "value" in report["missing_values"]


def test_validate_valid(sample_df):
    result = validate_dataframe(sample_df)
    assert result["valid"] is True


def test_validate_empty():
    result = validate_dataframe(pd.DataFrame())
    assert result["valid"] is False


def test_check_no_leakage():
    train = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
    val = pd.DataFrame({"id": [4, 5, 6], "target": [1, 0, 1]})
    result = check_leakage(train, val)
    assert result["leakage_found"] is False


def test_check_leakage_detected():
    train = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
    val = pd.DataFrame({"id": [2, 3, 4], "target": [1, 0, 1]})
    result = check_leakage(train, val)
    assert result["leakage_found"] is True
