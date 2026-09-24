"""
Amazon ML Worker — Data Inspection.

Schema-agnostic data inspection, producing structured reports.
Never assumes final column names, types, or modalities.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def inspect_dataframe(
    df: pd.DataFrame,
    id_column: str = "id",
    target_column: str = "target",
    image_dir: Optional[str] = None,
    image_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce a comprehensive inspection report for a DataFrame.

    Returns a dict suitable for JSON serialization.
    """
    report: Dict[str, Any] = {}

    # ── Basic shape ───────────────────────────────────────────────────────
    report["shape"] = {"rows": len(df), "columns": len(df.columns)}
    report["columns"] = list(df.columns)
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # ── Missing values ────────────────────────────────────────────────────
    missing = df.isnull().sum()
    report["missing_values"] = {col: int(v) for col, v in missing.items() if v > 0}
    report["missing_pct"] = {
        col: round(v / len(df) * 100, 2)
        for col, v in missing.items() if v > 0
    }

    # ── Duplicates ────────────────────────────────────────────────────────
    report["duplicate_rows"] = int(df.duplicated().sum())
    if id_column in df.columns:
        report["duplicate_ids"] = int(df[id_column].duplicated().sum())
    else:
        report["duplicate_ids"] = "id_column not found"

    # ── Unique counts ─────────────────────────────────────────────────────
    report["unique_counts"] = {col: int(df[col].nunique()) for col in df.columns}

    # ── Numeric summaries ─────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        report["numeric_summary"] = desc.to_dict()

    # ── Text statistics ───────────────────────────────────────────────────
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    text_stats = {}
    for col in text_cols:
        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue
        lengths = non_null.str.len()
        text_stats[col] = {
            "count": int(len(non_null)),
            "mean_length": round(float(lengths.mean()), 2),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "median_length": int(lengths.median()),
        }
    if text_stats:
        report["text_statistics"] = text_stats

    # ── Candidate columns ─────────────────────────────────────────────────
    report["candidate_id_columns"] = _guess_id_columns(df)
    report["candidate_target_columns"] = _guess_target_columns(df)

    # ── Image check ───────────────────────────────────────────────────────
    if image_dir and image_column and image_column in df.columns:
        image_report = _check_images(df, image_column, image_dir)
        report["images"] = image_report

    log.info("Inspection complete: %d rows, %d columns", len(df), len(df.columns))
    return report


def _guess_id_columns(df: pd.DataFrame) -> List[str]:
    """Heuristic: columns whose name contains 'id' and has high uniqueness."""
    candidates = []
    for col in df.columns:
        if "id" in col.lower():
            ratio = df[col].nunique() / max(len(df), 1)
            if ratio > 0.9:
                candidates.append(col)
    return candidates


def _guess_target_columns(df: pd.DataFrame) -> List[str]:
    """Heuristic: columns whose name contains 'target', 'label', 'class', 'y'."""
    keywords = ["target", "label", "class", "category", "output"]
    candidates = []
    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in keywords):
            candidates.append(col)
    return candidates


def _check_images(
    df: pd.DataFrame,
    image_column: str,
    image_dir: str,
) -> Dict[str, Any]:
    """Check image existence and basic corruption."""
    image_dir_path = Path(image_dir)
    total = 0
    missing = 0
    corrupted = 0
    for img_ref in df[image_column].dropna():
        total += 1
        img_path = image_dir_path / str(img_ref)
        if not img_path.exists():
            missing += 1
            continue
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            corrupted += 1
    return {
        "total": total,
        "missing": missing,
        "corrupted": corrupted,
        "valid": total - missing - corrupted,
    }


def save_inspection_report(
    report: Dict[str, Any],
    output_dir: str,
    prefix: str = "inspection",
) -> None:
    """Save report as JSON and human-readable TXT."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / f"{prefix}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Saved JSON report: %s", json_path)

    # TXT
    txt_path = out / f"{prefix}_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  Data Inspection Report\n")
        f.write("=" * 60 + "\n\n")
        _write_report_section(f, report)
    log.info("Saved TXT report: %s", txt_path)


def _write_report_section(f, data, indent=0):
    """Recursively write report sections."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                f.write(f"{prefix}{key}:\n")
                _write_report_section(f, value, indent + 1)
            else:
                f.write(f"{prefix}{key}: {value}\n")
    elif isinstance(data, list):
        for item in data:
            f.write(f"{prefix}- {item}\n")
    else:
        f.write(f"{prefix}{data}\n")
