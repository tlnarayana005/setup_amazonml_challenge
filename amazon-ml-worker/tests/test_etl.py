"""Test ETL pipeline."""
import pytest
import tempfile
from pathlib import Path
import pandas as pd
from src.config.loader import WorkerConfig
from src.etl.pipeline import run_etl, extract
from src.etl.preprocess import preprocess_dataframe, clean_text


@pytest.fixture
def synth_csv(tmp_path):
    df = pd.DataFrame({
        "id": range(50),
        "text": [f"  Sample TEXT {i}  " for i in range(50)],
        "price": [10.0 + i for i in range(50)],
        "category": ["cat_A" if i % 2 == 0 else "cat_B" for i in range(50)],
        "target": [i % 3 for i in range(50)],
    })
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return tmp_path


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("café") == "café"


def test_preprocess(synth_csv):
    df = pd.read_csv(synth_csv / "train.csv")
    config = WorkerConfig()
    result = preprocess_dataframe(df, config)
    assert len(result) == 50
    # Text should be cleaned (stripped/lowered)
    assert result["text"].iloc[0] == result["text"].iloc[0].strip()


def test_extract(synth_csv):
    config = WorkerConfig(input_path=str(synth_csv))
    df = extract(config)
    assert len(df) == 50


def test_etl_full(synth_csv):
    config = WorkerConfig(
        input_path=str(synth_csv),
        data_path=str(synth_csv),
    )
    df = run_etl(config)
    assert len(df) > 0
    assert (synth_csv / "processed" / "processed_data.parquet").exists()
