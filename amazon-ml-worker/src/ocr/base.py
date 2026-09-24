"""
Amazon ML Worker — OCR Base Interface.

Abstract OCR interface with batch processing, caching, and error tracking.
"""

import abc
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.etl.cache import CacheManager
from src.utils.logging import get_logger
from src.utils.timing import Timer

log = get_logger(__name__)


class OCRBackend(abc.ABC):
    """Abstract OCR backend interface."""

    @abc.abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from a single image. Return empty string on failure."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        ...


def run_ocr_batch(
    df: pd.DataFrame,
    backend: OCRBackend,
    id_column: str = "id",
    image_column: str = "image",
    image_dir: str = "data/raw/images",
    cache_dir: str = "cache",
    output_dir: str = "features/ocr",
    batch_size: int = 100,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run OCR on a batch of images with caching and error tracking.

    Returns DataFrame with columns [id, ocr_text].
    """
    if not backend.is_available():
        log.warning("OCR backend '%s' is not available.", backend.name)
        return pd.DataFrame(columns=[id_column, "ocr_text"])

    timer = Timer()
    timer.start("ocr")

    # Cache setup
    cache = CacheManager(cache_dir, cache_key=f"ocr_{backend.name}")

    # Filter already-cached IDs
    if resume:
        df = cache.filter_uncached(df, id_column=id_column)

    if len(df) == 0:
        log.info("All IDs already cached. Nothing to process.")
        cached = cache.get_cached_data()
        return cached if cached is not None else pd.DataFrame(columns=[id_column, "ocr_text"])

    log.info(
        "Processing %d images with '%s' backend.",
        len(df), backend.name,
    )

    results: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    image_dir_path = Path(image_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"OCR ({backend.name})"):
        row_id = str(row[id_column])
        img_ref = str(row.get(image_column, ""))
        img_path = image_dir_path / img_ref

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            text = backend.extract_text(str(img_path))
            results.append({id_column: row_id, "ocr_text": text})
        except Exception as e:
            errors.append({id_column: row_id, "error": str(e)})
            results.append({id_column: row_id, "ocr_text": ""})

    # Build results DataFrame
    result_df = pd.DataFrame(results)

    # Save errors
    if errors:
        err_dir = Path(output_dir)
        err_dir.mkdir(parents=True, exist_ok=True)
        err_path = err_dir / "ocr_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        log.warning("OCR errors: %d (saved to %s)", len(errors), err_path)

    # Update cache
    cache.save_to_cache(result_df, append=True)

    timer.stop("ocr")
    log.info("OCR complete: %d results, %d errors. %s", len(results), len(errors), timer)

    # Return all cached results
    all_cached = cache.get_cached_data()
    return all_cached if all_cached is not None else result_df


def get_ocr_backend(backend_name: str) -> OCRBackend:
    """Factory: return the requested OCR backend."""
    if backend_name == "tesseract":
        from src.ocr.tesseract_backend import TesseractBackend
        return TesseractBackend()
    elif backend_name == "paddleocr":
        from src.ocr.paddleocr_backend import PaddleOCRBackend
        return PaddleOCRBackend()
    else:
        raise ValueError(f"Unknown OCR backend: {backend_name}")
