"""Test worker output merging."""
import pytest
import tempfile
from pathlib import Path
import pandas as pd
from src.data.merge import merge_worker_outputs, verify_merge


@pytest.fixture
def worker_dir(tmp_path):
    """Create fake worker output directories."""
    for wid in range(3):
        d = tmp_path / f"worker_{wid}"
        d.mkdir()
        df = pd.DataFrame({
            "id": [f"item_{wid * 10 + i}" for i in range(10)],
            "value": range(10),
        })
        df.to_parquet(d / "output.parquet", index=False)
    return tmp_path


def test_merge_basic(worker_dir):
    merged = merge_worker_outputs(str(worker_dir))
    assert len(merged) == 30
    assert merged["id"].duplicated().sum() == 0


def test_merge_save(worker_dir):
    out = worker_dir / "merged.parquet"
    merged = merge_worker_outputs(str(worker_dir), output_file=str(out))
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert len(loaded) == 30


def test_merge_duplicate_ids(tmp_path):
    """Duplicate IDs across workers should raise."""
    for wid in range(2):
        d = tmp_path / f"worker_{wid}"
        d.mkdir()
        df = pd.DataFrame({"id": ["same_id_1", "same_id_2"], "v": [wid, wid]})
        df.to_parquet(d / "output.parquet", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        merge_worker_outputs(str(tmp_path))


def test_verify_merge():
    original = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    merged = pd.DataFrame({"id": [1, 2, 3], "result": [0.1, 0.2, 0.3]})
    report = verify_merge(merged, original)
    assert report["valid"] is True


def test_verify_merge_missing():
    original = pd.DataFrame({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    merged = pd.DataFrame({"id": [1, 2], "result": [0.1, 0.2]})
    report = verify_merge(merged, original)
    assert report["valid"] is False
