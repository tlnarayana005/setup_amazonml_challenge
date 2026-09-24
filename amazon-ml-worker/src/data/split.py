"""
Amazon ML Worker — Train/Validation Split.

Supports random, stratified, and group-aware splits.
All splits are deterministic given the same seed.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedShuffleSplit,
    ShuffleSplit,
)

from src.utils.logging import get_logger

log = get_logger(__name__)


def create_split(
    df: pd.DataFrame,
    method: str = "stratified",
    target_column: str = "target",
    group_column: str = "",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into (train, val) DataFrames.

    Args:
        method: 'random', 'stratified', or 'group'
        target_column: Column to stratify on (for 'stratified')
        group_column: Column to split by groups (for 'group')
        val_fraction: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        (train_df, val_df)
    """
    n = len(df)
    if n == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    if method == "stratified":
        if target_column not in df.columns:
            log.warning(
                "Target column '%s' not found; falling back to random split.",
                target_column,
            )
            return _random_split(df, val_fraction, seed)
        return _stratified_split(df, target_column, val_fraction, seed)

    elif method == "group":
        if not group_column or group_column not in df.columns:
            log.warning(
                "Group column '%s' not found; falling back to random split.",
                group_column,
            )
            return _random_split(df, val_fraction, seed)
        return _group_split(df, group_column, val_fraction, seed)

    else:  # random
        return _random_split(df, val_fraction, seed)


def _random_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = ShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(df))
    log.info("Random split: train=%d, val=%d", len(train_idx), len(val_idx))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _stratified_split(
    df: pd.DataFrame,
    target_column: str,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target_column]
    # Handle edge case: too few samples per class
    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        log.warning(
            "Some classes have < 2 samples; falling back to random split."
        )
        return _random_split(df, val_fraction, seed)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(df, y))
    log.info("Stratified split: train=%d, val=%d", len(train_idx), len(val_idx))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _group_split(
    df: pd.DataFrame,
    group_column: str,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_column]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    log.info("Group split: train=%d, val=%d", len(train_idx), len(val_idx))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
