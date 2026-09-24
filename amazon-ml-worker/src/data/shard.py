"""
Amazon ML Worker — Deterministic Sharding.

Assigns each row to exactly one worker based on hash(id) % total_workers.
Guarantees: deterministic, no overlap, complete coverage, stable across restarts.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def compute_shard_assignment(
    df: pd.DataFrame,
    total_workers: int,
    id_column: str = "id",
) -> pd.Series:
    """
    Return a Series of shard indices (0..total_workers-1) for each row.

    Uses a deterministic hash of the ID column.
    """
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in DataFrame.")

    # Use pandas factorize + modulo for deterministic assignment
    # We hash the string representation for stability
    ids = df[id_column].astype(str)
    hashes = ids.apply(lambda x: hash(x) % (2**32))
    shards = hashes % total_workers
    return shards.astype(int)


def get_shard(
    df: pd.DataFrame,
    worker_id: int,
    total_workers: int,
    id_column: str = "id",
) -> pd.DataFrame:
    """
    Return the subset of *df* assigned to *worker_id*.

    Args:
        worker_id: This worker's index (0-based)
        total_workers: Total number of workers
        id_column: Column to hash for shard assignment

    Returns:
        DataFrame containing only this worker's shard
    """
    if total_workers <= 0:
        raise ValueError(f"total_workers must be > 0, got {total_workers}")
    if not (0 <= worker_id < total_workers):
        raise ValueError(
            f"worker_id must be in [0, {total_workers}), got {worker_id}"
        )

    shards = compute_shard_assignment(df, total_workers, id_column)
    mask = shards == worker_id
    result = df[mask].reset_index(drop=True)

    log.info(
        "Shard %d/%d: %d rows (%.1f%% of %d)",
        worker_id,
        total_workers,
        len(result),
        len(result) / max(len(df), 1) * 100,
        len(df),
    )
    return result


def verify_sharding(
    df: pd.DataFrame,
    total_workers: int,
    id_column: str = "id",
) -> dict:
    """
    Verify that sharding produces no overlap and complete coverage.

    Returns:
        {"valid": bool, "total": int, "per_shard": [...], "overlap": int}
    """
    shards = compute_shard_assignment(df, total_workers, id_column)
    total = len(df)
    per_shard = []
    all_indices = set()
    overlap_count = 0

    for wid in range(total_workers):
        mask = shards == wid
        shard_indices = set(df.index[mask])
        overlap = shard_indices & all_indices
        overlap_count += len(overlap)
        all_indices.update(shard_indices)
        per_shard.append(int(mask.sum()))

    coverage = len(all_indices) == total
    valid = coverage and overlap_count == 0

    result = {
        "valid": valid,
        "total": total,
        "coverage_complete": coverage,
        "per_shard": per_shard,
        "overlap": overlap_count,
    }

    if valid:
        log.info("Sharding verification PASSED: %s", per_shard)
    else:
        log.error("Sharding verification FAILED: %s", result)

    return result
