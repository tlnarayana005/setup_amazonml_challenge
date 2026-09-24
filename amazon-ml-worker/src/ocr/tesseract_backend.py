"""
Amazon ML Worker — Tesseract OCR Backend.

Wraps pytesseract for text extraction. Conditionally imported.
"""

from src.ocr.base import OCRBackend
from src.utils.logging import get_logger

log = get_logger(__name__)


class TesseractBackend(OCRBackend):
    """OCR backend using Tesseract via pytesseract."""

    @property
    def name(self) -> str:
        return "tesseract"

    def is_available(self) -> bool:
        try:
            import pytesseract
            # Quick check: tesseract binary reachable
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text(self, image_path: str) -> str:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
