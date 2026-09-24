"""
Amazon ML Worker — Data Validation.

Validates data integrity: types, ranges, missing values, duplicates.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    id_column: str = "id",
    target_column: str = "target",
    expected_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run validation checks and return a report dict.

    Returns:
        {"valid": bool, "errors": [...], "warnings": [...]}
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ── Empty check ───────────────────────────────────────────────────────
    if len(df) == 0:
        errors.append("DataFrame is empty.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # ── Expected columns ──────────────────────────────────────────────────
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing expected columns: {missing_cols}")

    # ── ID column ─────────────────────────────────────────────────────────
    if id_column in df.columns:
        n_dup = df[id_column].duplicated().sum()
        if n_dup > 0:
            warnings.append(f"Duplicate IDs: {n_dup} duplicates in '{id_column}'.")
        n_null = df[id_column].isnull().sum()
        if n_null > 0:
            errors.append(f"Null IDs: {n_null} null values in '{id_column}'.")
    else:
        warnings.append(f"ID column '{id_column}' not found in DataFrame.")

    # ── Target column ─────────────────────────────────────────────────────
    if target_column and target_column in df.columns:
        n_null = df[target_column].isnull().sum()
        if n_null > 0:
            warnings.append(
                f"Missing targets: {n_null} null values in '{target_column}'."
            )
    elif target_column:
        warnings.append(f"Target column '{target_column}' not found.")

    # ── Overall missing ───────────────────────────────────────────────────
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / max(total_cells, 1)) * 100
    if missing_pct > 50:
        warnings.append(f"High missing rate: {missing_pct:.1f}% of all cells are null.")

    # ── Constant columns ──────────────────────────────────────────────────
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            warnings.append(f"Constant column: '{col}' has <= 1 unique value.")

    valid = len(errors) == 0
    if valid:
        log.info("Validation passed. %d warnings.", len(warnings))
    else:
        log.warning("Validation FAILED. %d errors, %d warnings.", len(errors), len(warnings))

    return {"valid": valid, "errors": errors, "warnings": warnings}


def check_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_column: str = "id",
    group_column: str = "",
) -> Dict[str, Any]:
    """Check for ID or group leakage between train and validation sets."""
    report: Dict[str, Any] = {"leakage_found": False, "details": []}

    if id_column in train_df.columns and id_column in val_df.columns:
        overlap = set(train_df[id_column]) & set(val_df[id_column])
        if overlap:
            report["leakage_found"] = True
            report["details"].append(
                f"ID leakage: {len(overlap)} IDs appear in both train and val."
            )
            log.warning("ID LEAKAGE DETECTED: %d overlapping IDs!", len(overlap))

    if group_column and group_column in train_df.columns and group_column in val_df.columns:
        overlap = set(train_df[group_column]) & set(val_df[group_column])
        if overlap:
            report["leakage_found"] = True
            report["details"].append(
                f"Group leakage: {len(overlap)} groups appear in both train and val."
            )
            log.warning("GROUP LEAKAGE DETECTED: %d overlapping groups!", len(overlap))

    if not report["leakage_found"]:
        log.info("No leakage detected.")

    return report
