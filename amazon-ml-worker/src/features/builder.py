"""
Amazon ML Worker — Feature Builder.

Generates features from structured data, text, and image columns.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.config.loader import WorkerConfig
from src.etl.preprocess import extract_text_features
from src.nlp.text_utils import add_text_features
from src.utils.logging import get_logger

log = get_logger(__name__)


def build_features(
    df: pd.DataFrame,
    config: WorkerConfig,
) -> pd.DataFrame:
    """
    Build features from a DataFrame based on configuration.

    Generates numeric features from text columns and passes through existing numeric/categorical columns.
    """
    df = df.copy()
    features = [df[[config.id_column]].copy()] if config.id_column in df.columns else []

    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != config.id_column and c != config.target_column]
    if numeric_cols:
        features.append(df[numeric_cols])
        log.info("Added %d numeric features.", len(numeric_cols))

    # Categorical features (label encoded)
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_cols = [
        c for c in cat_cols
        if c not in (config.id_column, config.target_column, config.text_column, config.image_column)
    ]
    for col in cat_cols:
        codes = df[col].astype("category").cat.codes
        features.append(pd.DataFrame({f"{col}_encoded": codes}))
    if cat_cols:
        log.info("Added %d categorical (label-encoded) features.", len(cat_cols))

    # Text features
    if config.text_column in df.columns:
        text_feats = add_text_features(df, config.text_column)
        text_feat_cols = [c for c in text_feats.columns if c not in df.columns]
        if text_feat_cols:
            features.append(text_feats[text_feat_cols])
            log.info("Added %d text features.", len(text_feat_cols))

    # Target (keep for training)
    if config.target_column in df.columns:
        features.append(df[[config.target_column]])

    # Merge
    result = pd.concat(features, axis=1)
    log.info("Built feature matrix: %d rows, %d columns.", len(result), len(result.columns))
    return result
