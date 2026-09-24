"""Test deterministic sharding — critical for multi-worker correctness."""
import pytest
import pandas as pd
from src.data.shard import get_shard, verify_sharding, compute_shard_assignment


@pytest.fixture
def df():
    return pd.DataFrame({
        "id": [f"item_{i}" for i in range(100)],
        "value": range(100),
    })


def test_no_overlap(df):
    """Shards must not overlap."""
    total_workers = 3
    all_ids = set()
    for wid in range(total_workers):
        shard = get_shard(df, wid, total_workers)
        shard_ids = set(shard["id"])
        overlap = all_ids & shard_ids
        assert len(overlap) == 0, f"Worker {wid} overlaps: {overlap}"
        all_ids.update(shard_ids)


def test_complete_coverage(df):
    """All rows must be assigned to exactly one shard."""
    total_workers = 3
    all_ids = set()
    for wid in range(total_workers):
        shard = get_shard(df, wid, total_workers)
        all_ids.update(shard["id"])
    assert all_ids == set(df["id"])


def test_deterministic(df):
    """Same input → same shards."""
    s1 = get_shard(df, 0, 3)
    s2 = get_shard(df, 0, 3)
    assert s1["id"].tolist() == s2["id"].tolist()


def test_verify_sharding(df):
    result = verify_sharding(df, total_workers=5)
    assert result["valid"] is True
    assert result["overlap"] == 0
    assert result["coverage_complete"] is True
    assert sum(result["per_shard"]) == len(df)


def test_single_worker(df):
    shard = get_shard(df, 0, 1)
    assert len(shard) == len(df)


def test_invalid_worker_id(df):
    with pytest.raises(ValueError):
        get_shard(df, 5, 3)


def test_three_worker_simulation(df):
    """Full 3-worker simulation as required by spec."""
    total = 3
    shards = []
    for wid in range(total):
        shard = get_shard(df, wid, total)
        shards.append(shard)

    # No overlap
    all_ids = []
    for s in shards:
        all_ids.extend(s["id"].tolist())
    assert len(all_ids) == len(set(all_ids)), "Duplicate IDs across shards!"

    # Complete coverage
    assert set(all_ids) == set(df["id"])

    # Correct total
    total_rows = sum(len(s) for s in shards)
    assert total_rows == len(df)
