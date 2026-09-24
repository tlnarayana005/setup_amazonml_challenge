"""Test train/validation splits."""
import pytest
import numpy as np
import pandas as pd
from src.data.split import create_split


@pytest.fixture
def df():
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(200),
        "feature": np.random.randn(200),
        "target": np.random.choice([0, 1, 2], 200),
        "group": np.random.choice(["A", "B", "C", "D"], 200),
    })


def test_random_split(df):
    train, val = create_split(df, method="random", val_fraction=0.2, seed=42)
    assert len(train) + len(val) == len(df)
    assert len(val) == pytest.approx(40, abs=5)


def test_stratified_split(df):
    train, val = create_split(df, method="stratified", target_column="target", val_fraction=0.2, seed=42)
    assert len(train) + len(val) == len(df)


def test_group_split(df):
    train, val = create_split(df, method="group", group_column="group", val_fraction=0.25, seed=42)
    assert len(train) + len(val) == len(df)
    # No group overlap
    train_groups = set(train["group"])
    val_groups = set(val["group"])
    assert len(train_groups & val_groups) == 0


def test_deterministic_split(df):
    t1, v1 = create_split(df, method="stratified", target_column="target", seed=42)
    t2, v2 = create_split(df, method="stratified", target_column="target", seed=42)
    assert t1["id"].tolist() == t2["id"].tolist()
    assert v1["id"].tolist() == v2["id"].tolist()


def test_split_empty():
    with pytest.raises(ValueError):
        create_split(pd.DataFrame(), method="random")
