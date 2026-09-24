"""
Amazon ML Worker — Text Utilities.

Text cleaning, normalization, Unicode handling, whitespace normalization,
text statistics, and tokenization support.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize a single text string."""
    if not isinstance(text, str):
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters (keep newlines/tabs initially)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize text: clean + optional lowercase."""
    text = clean_text(text)
    if lowercase:
        text = text.lower()
    return text


def compute_text_stats(texts: pd.Series) -> Dict[str, Any]:
    """Compute statistics for a Series of text strings."""
    non_null = texts.dropna().astype(str)
    if len(non_null) == 0:
        return {"count": 0}

    lengths = non_null.str.len()
    word_counts = non_null.str.split().str.len()

    return {
        "count": int(len(non_null)),
        "null_count": int(texts.isnull().sum()),
        "mean_char_length": round(float(lengths.mean()), 2),
        "median_char_length": int(lengths.median()),
        "min_char_length": int(lengths.min()),
        "max_char_length": int(lengths.max()),
        "mean_word_count": round(float(word_counts.mean()), 2),
        "median_word_count": int(word_counts.median()),
        "min_word_count": int(word_counts.min()),
        "max_word_count": int(word_counts.max()),
        "empty_strings": int((non_null.str.len() == 0).sum()),
    }


def batch_clean_text(
    df: pd.DataFrame,
    text_column: str,
    output_column: Optional[str] = None,
    lowercase: bool = True,
) -> pd.DataFrame:
    """Clean text column in a DataFrame."""
    df = df.copy()
    out_col = output_column or f"{text_column}_clean"
    df[out_col] = df[text_column].apply(
        lambda x: normalize_text(x, lowercase=lowercase) if pd.notna(x) else ""
    )
    return df


def add_text_features(
    df: pd.DataFrame,
    text_column: str,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Add text-derived features to a DataFrame."""
    df = df.copy()
    pfx = prefix or text_column
    texts = df[text_column].fillna("").astype(str)

    df[f"{pfx}_char_len"] = texts.str.len()
    df[f"{pfx}_word_count"] = texts.str.split().str.len().fillna(0).astype(int)
    df[f"{pfx}_has_digits"] = texts.str.contains(r"\d", regex=True).astype(int)
    df[f"{pfx}_has_upper"] = texts.str.contains(r"[A-Z]", regex=True).astype(int)
    df[f"{pfx}_unique_words"] = texts.apply(
        lambda x: len(set(x.lower().split())) if x else 0
    )

    return df
