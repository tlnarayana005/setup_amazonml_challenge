"""Test cache manager."""
import pytest
import pandas as pd
from src.etl.cache import CacheManager, compute_config_hash


def test_cache_roundtrip(tmp_path):
    cache = CacheManager(str(tmp_path), cache_key="test")
    df = pd.DataFrame({"id": ["a", "b", "c"], "value": [1, 2, 3]})
    cache.save_to_cache(df, config_hash="abc123")

    loaded = cache.get_cached_data()
    assert loaded is not None
    assert len(loaded) == 3
    assert set(loaded["id"]) == {"a", "b", "c"}


def test_cached_ids(tmp_path):
    cache = CacheManager(str(tmp_path), cache_key="test")
    df = pd.DataFrame({"id": ["x", "y", "z"], "val": [10, 20, 30]})
    cache.save_to_cache(df)

    ids = cache.get_cached_ids()
    assert ids == {"x", "y", "z"}


def test_filter_uncached(tmp_path):
    cache = CacheManager(str(tmp_path), cache_key="test")
    cached = pd.DataFrame({"id": ["a", "b"], "val": [1, 2]})
    cache.save_to_cache(cached)

    full = pd.DataFrame({"id": ["a", "b", "c", "d"], "val": [1, 2, 3, 4]})
    remaining = cache.filter_uncached(full)
    assert len(remaining) == 2
    assert set(remaining["id"]) == {"c", "d"}


def test_append_cache(tmp_path):
    cache = CacheManager(str(tmp_path), cache_key="test")
    df1 = pd.DataFrame({"id": ["a", "b"], "val": [1, 2]})
    cache.save_to_cache(df1)
    df2 = pd.DataFrame({"id": ["c", "d"], "val": [3, 4]})
    cache.save_to_cache(df2, append=True)

    loaded = cache.get_cached_data()
    assert len(loaded) == 4


def test_config_hash():
    h1 = compute_config_hash({"a": 1, "b": 2})
    h2 = compute_config_hash({"b": 2, "a": 1})
    h3 = compute_config_hash({"a": 1, "b": 3})
    assert h1 == h2  # order independent
    assert h1 != h3


def test_cache_clear(tmp_path):
    cache = CacheManager(str(tmp_path), cache_key="test")
    df = pd.DataFrame({"id": ["a"], "val": [1]})
    cache.save_to_cache(df)
    assert cache.get_cached_data() is not None
    cache.clear()
    assert cache.get_cached_data() is None
