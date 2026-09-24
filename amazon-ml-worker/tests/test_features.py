"""Test feature builder and merger."""
import pytest
import numpy as np
import pandas as pd
from src.config.loader import WorkerConfig
from src.features.builder import build_features
from src.features.merger import merge_features, save_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(50),
        "text": [f"sample text {i}" for i in range(50)],
        "category": np.random.choice(["a", "b", "c"], 50),
        "price": np.random.uniform(10, 500, 50),
        "rating": np.random.uniform(1, 5, 50),
        "target": np.random.randint(0, 3, 50),
    })


def test_build_features(sample_df):
    config = WorkerConfig(text_column="text")
    feats = build_features(sample_df, config)
    assert "id" in feats.columns
    assert "target" in feats.columns
    assert "price" in feats.columns
    assert len(feats) == 50


def test_merge_features():
    base = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    feat1 = pd.DataFrame({"id": [1, 2, 3], "f1": [0.1, 0.2, 0.3]})
    feat2 = pd.DataFrame({"id": [1, 2, 3], "f2": [0.4, 0.5, 0.6]})
    result = merge_features(base, [feat1, feat2])
    assert len(result) == 3
    assert "f1" in result.columns
    assert "f2" in result.columns


def test_merge_features_partial():
    base = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    feat = pd.DataFrame({"id": [1, 2], "f1": [0.1, 0.2]})
    result = merge_features(base, [feat], how="left")
    assert len(result) == 3  # left join keeps all base rows


def test_merge_no_id_multiplication():
    base = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    feat = pd.DataFrame({"id": [1, 2], "f1": [0.1, 0.2]})
    result = merge_features(base, [feat])
    assert len(result) == 2  # no multiplication


def test_save_features(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "f": [0.1, 0.2]})
    path = str(tmp_path / "feats.parquet")
    save_features(df, path)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 2
