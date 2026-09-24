"""
Amazon ML Worker — Preprocessing Transforms.

Text cleaning, numeric cleaning, categorical normalization.
All transforms are configurable and independently callable.
"""

import re
import unicodedata
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config.loader import WorkerConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


def preprocess_dataframe(
    df: pd.DataFrame,
    config: WorkerConfig,
) -> pd.DataFrame:
    """Apply all relevant preprocessing transforms."""
    df = df.copy()

    # Handle duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        log.info("Removing %d duplicate rows.", n_dup)
        df = df.drop_duplicates().reset_index(drop=True)

    # Process text columns
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in text_cols:
        if col == config.id_column:
            continue  # Don't clean IDs
        df[col] = df[col].apply(lambda x: clean_text(x) if pd.notna(x) else x)

    # Process numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == config.id_column:
            continue
        # Fill numeric NaN with median (safe default)
        n_null = df[col].isnull().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info("Filled %d nulls in '%s' with median=%.4f", n_null, col, median_val)

    # Normalize categorical columns (lowercase, strip)
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        if col == config.id_column:
            continue
        df[col] = df[col].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

    return df


def clean_text(text: str) -> str:
    """Clean a single text string."""
    if not isinstance(text, str):
        return text

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_text_features(
    df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    """Add text-length features from a text column."""
    if text_column not in df.columns:
        return df

    df = df.copy()
    texts = df[text_column].fillna("").astype(str)
    df[f"{text_column}_len"] = texts.str.len()
    df[f"{text_column}_word_count"] = texts.str.split().str.len()
    df[f"{text_column}_char_count"] = texts.str.replace(r"\s", "", regex=True).str.len()

    return df
