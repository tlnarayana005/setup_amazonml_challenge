"""
Amazon ML Worker — ETL Pipeline.

Orchestrates Extract → Validate → Transform → Load.
Each stage is independently callable.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config.loader import WorkerConfig
from src.data.validate import validate_dataframe
from src.etl.preprocess import preprocess_dataframe
from src.etl.cache import CacheManager
from src.utils.logging import get_logger
from src.utils.timing import Timer

log = get_logger(__name__)


def run_etl(
    config: WorkerConfig,
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full ETL pipeline.

    1. Extract: load raw data
    2. Validate: check integrity
    3. Transform: clean & preprocess
    4. Load: save to processed directory

    Returns the processed DataFrame.
    """
    timer = Timer()
    timer.start("total")

    # ── Extract ───────────────────────────────────────────────────────────
    with timer.section("extract"):
        df = extract(config, input_file)
        log.info("Extracted %d rows, %d columns.", len(df), len(df.columns))

    # ── Validate ──────────────────────────────────────────────────────────
    with timer.section("validate"):
        validation = validate_dataframe(
            df,
            id_column=config.id_column,
            target_column=config.target_column,
        )
        if not validation["valid"]:
            log.warning("Validation errors: %s", validation["errors"])
        for w in validation.get("warnings", []):
            log.warning("  Warning: %s", w)

    # ── Transform ─────────────────────────────────────────────────────────
    with timer.section("transform"):
        df = preprocess_dataframe(df, config)
        log.info("Transformed: %d rows, %d columns.", len(df), len(df.columns))

    # ── Load ──────────────────────────────────────────────────────────────
    with timer.section("load"):
        output_dir = Path(config.data_path) / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "processed_data.parquet"
        df.to_parquet(output_path, index=False)
        log.info("Saved processed data: %s", output_path)

    timer.stop("total")
    log.info("ETL complete. %s", timer)

    return df


def extract(
    config: WorkerConfig,
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    """Load raw data from CSV or Parquet."""
    if input_file:
        path = Path(input_file)
    else:
        input_dir = Path(config.input_path)
        # Auto-detect file
        candidates = list(input_dir.glob("*.csv")) + list(input_dir.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No CSV/Parquet files found in {input_dir}")
        path = candidates[0]
        if len(candidates) > 1:
            log.warning("Multiple files found; using %s", path)

    log.info("Loading: %s", path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Apply max_rows limit
    if config.max_rows > 0 and len(df) > config.max_rows:
        log.info("Limiting to %d rows (from %d).", config.max_rows, len(df))
        df = df.head(config.max_rows)

    return df
