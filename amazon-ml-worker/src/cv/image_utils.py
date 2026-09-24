"""
Amazon ML Worker — Image Utilities.

Loading, validation, resizing, cropping, normalization, conversion.
Supports batches and preserves IDs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.utils.logging import get_logger

log = get_logger(__name__)

# Try importing OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_image(
    path: str,
    target_size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB",
) -> Optional[np.ndarray]:
    """
    Load an image file as a NumPy array.

    Args:
        path: Path to the image file
        target_size: (width, height) to resize to, or None
        mode: PIL mode ('RGB', 'L', etc.)

    Returns:
        NumPy array or None if loading fails
    """
    try:
        img = Image.open(path).convert(mode)
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        log.warning("Failed to load image %s: %s", path, e)
        return None


def validate_image(path: str) -> Dict[str, Any]:
    """
    Validate an image file.

    Returns:
        {"valid": bool, "width": int, "height": int, "mode": str, "error": str}
    """
    result: Dict[str, Any] = {"valid": False, "path": path}
    try:
        with Image.open(path) as img:
            img.verify()
        # Re-open to get properties (verify closes the file)
        with Image.open(path) as img:
            result["valid"] = True
            result["width"] = img.width
            result["height"] = img.height
            result["mode"] = img.mode
            result["format"] = img.format
    except Exception as e:
        result["error"] = str(e)
    return result


def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect: bool = True,
) -> np.ndarray:
    """Resize an image array. target_size is (width, height)."""
    pil_img = Image.fromarray(img)

    if keep_aspect:
        pil_img.thumbnail(target_size, Image.LANCZOS)
        # Pad to exact size
        padded = Image.new(pil_img.mode, target_size, (0, 0, 0))
        offset = (
            (target_size[0] - pil_img.width) // 2,
            (target_size[1] - pil_img.height) // 2,
        )
        padded.paste(pil_img, offset)
        return np.array(padded)
    else:
        return np.array(pil_img.resize(target_size, Image.LANCZOS))


def normalize_image(
    img: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize image to [0,1] and apply ImageNet-style normalization."""
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return img


def batch_validate_images(
    image_paths: List[str],
    ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate a batch of images.

    Returns:
        {"total": int, "valid": int, "invalid": int, "errors": [...]}
    """
    total = len(image_paths)
    valid_count = 0
    errors = []

    for i, path in enumerate(image_paths):
        result = validate_image(path)
        if result["valid"]:
            valid_count += 1
        else:
            err_id = ids[i] if ids else str(i)
            errors.append({"id": err_id, "path": path, "error": result.get("error", "unknown")})

    return {
        "total": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "errors": errors,
    }
