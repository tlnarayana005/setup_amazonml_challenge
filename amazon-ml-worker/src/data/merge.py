"""
Amazon ML Worker — Worker Output Merging.

Merges output.parquet files from multiple workers with integrity checks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def discover_worker_outputs(
    base_dir: str,
    filename: str = "output.parquet",
) -> List[Path]:
    """Find all worker_*/output.parquet files under *base_dir*."""
    base = Path(base_dir)
    outputs = sorted(base.glob(f"worker_*/{filename}"))
    log.info("Discovered %d worker output files in %s", len(outputs), base_dir)
    return outputs


def merge_worker_outputs(
    base_dir: str,
    id_column: str = "id",
    filename: str = "output.parquet",
    output_file: Optional[str] = None,
    expected_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Merge all worker output files with integrity verification.

    Args:
        base_dir: Directory containing worker_*/ subdirectories
        id_column: Column used for deduplication checks
        filename: Name of the output file in each worker directory
        output_file: If set, save merged result here
        expected_workers: If set, verify this many workers contributed

    Returns:
        Merged DataFrame

    Raises:
        ValueError: On integrity errors (duplicates, schema mismatch)
    """
    paths = discover_worker_outputs(base_dir, filename)

    if not paths:
        raise FileNotFoundError(f"No worker outputs found in {base_dir}")

    if expected_workers is not None and len(paths) != expected_workers:
        log.warning(
            "Expected %d workers but found %d output files.",
            expected_workers,
            len(paths),
        )

    # Load all DataFrames
    dfs: List[pd.DataFrame] = []
    reference_columns = None

    for path in paths:
        worker_name = path.parent.name
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise IOError(f"Failed to read {path}: {e}")

        # Schema check
        if reference_columns is None:
            reference_columns = set(df.columns)
        else:
            current_columns = set(df.columns)
            if current_columns != reference_columns:
                extra = current_columns - reference_columns
                missing = reference_columns - current_columns
                raise ValueError(
                    f"Schema mismatch in {worker_name}: "
                    f"extra={extra}, missing={missing}"
                )

        log.info("  %s: %d rows", worker_name, len(df))
        dfs.append(df)

    # Concatenate
    merged = pd.concat(dfs, ignore_index=True)
    log.info("Merged total: %d rows", len(merged))

    # ── Integrity checks ──────────────────────────────────────────────────
    if id_column in merged.columns:
        n_dup = merged[id_column].duplicated().sum()
        if n_dup > 0:
            raise ValueError(
                f"INTEGRITY ERROR: {n_dup} duplicate IDs found after merge!"
            )
        log.info("ID uniqueness check passed.")

    # ── Save ──────────────────────────────────────────────────────────────
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)
        log.info("Saved merged output: %s", out_path)

    return merged


def verify_merge(
    merged_df: pd.DataFrame,
    original_df: pd.DataFrame,
    id_column: str = "id",
) -> Dict[str, Any]:
    """
    Verify merged output against original data.

    Returns:
        {"valid": bool, "details": {...}}
    """
    report = {
        "valid": True,
        "merged_rows": len(merged_df),
        "original_rows": len(original_df),
        "details": [],
    }

    # Row count
    if len(merged_df) != len(original_df):
        report["details"].append(
            f"Row count mismatch: merged={len(merged_df)}, original={len(original_df)}"
        )
        report["valid"] = False

    # ID coverage
    if id_column in merged_df.columns and id_column in original_df.columns:
        merged_ids = set(merged_df[id_column])
        original_ids = set(original_df[id_column])

        missing = original_ids - merged_ids
        unexpected = merged_ids - original_ids

        if missing:
            report["details"].append(f"Missing IDs: {len(missing)}")
            report["valid"] = False
        if unexpected:
            report["details"].append(f"Unexpected IDs: {len(unexpected)}")
            report["valid"] = False

    if report["valid"]:
        log.info("Merge verification PASSED.")
    else:
        log.error("Merge verification FAILED: %s", report["details"])

    return report
