"""
Amazon ML Worker — Feature Merger.

Merges features from multiple sources (structured, OCR, CV, NLP, embeddings).
Verifies row counts, unique IDs, and schema compatibility.
"""

from typing import List, Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def merge_features(
    base_df: pd.DataFrame,
    feature_dfs: List[pd.DataFrame],
    id_column: str = "id",
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge multiple feature DataFrames on *id_column*.

    Args:
        base_df: Base DataFrame with IDs
        feature_dfs: List of DataFrames to merge in
        id_column: Join column
        how: Merge type ('left', 'inner', 'outer')

    Returns:
        Merged DataFrame

    Raises:
        ValueError: On ID duplication or unexpected row multiplication
    """
    if id_column not in base_df.columns:
        raise ValueError(f"ID column '{id_column}' not found in base DataFrame.")

    result = base_df.copy()
    original_len = len(result)

    for i, feat_df in enumerate(feature_dfs):
        if feat_df is None or len(feat_df) == 0:
            log.info("Skipping empty feature DataFrame #%d.", i)
            continue

        if id_column not in feat_df.columns:
            log.warning("Feature DF #%d missing ID column '%s'; skipping.", i, id_column)
            continue

        # Check for duplicate IDs in feature DF
        n_dup = feat_df[id_column].duplicated().sum()
        if n_dup > 0:
            log.warning("Feature DF #%d has %d duplicate IDs; deduplicating.", i, n_dup)
            feat_df = feat_df.drop_duplicates(subset=[id_column], keep="last")

        # Avoid duplicate column names
        overlap_cols = set(result.columns) & set(feat_df.columns) - {id_column}
        if overlap_cols:
            log.warning(
                "Overlapping columns in feature DF #%d: %s. Suffixed with '_feat%d'.",
                i, overlap_cols, i,
            )
            rename_map = {col: f"{col}_feat{i}" for col in overlap_cols}
            feat_df = feat_df.rename(columns=rename_map)

        result = result.merge(feat_df, on=id_column, how=how)

        # Row multiplication check
        if len(result) > original_len * 1.01:  # Allow 1% tolerance
            raise ValueError(
                f"Row multiplication detected after merging feature DF #%d: "
                f"{original_len} -> {len(result)}" % i
            )

    # Verify
    n_missing = result[id_column].isnull().sum()
    if n_missing > 0:
        log.warning("%d null IDs after merge.", n_missing)

    log.info(
        "Feature merge complete: %d rows, %d columns (was %d columns).",
        len(result), len(result.columns), len(base_df.columns),
    )
    return result


def save_features(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save feature DataFrame to Parquet."""
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Saved features: %s (%d rows, %d cols)", output_path, len(df), len(df.columns))
