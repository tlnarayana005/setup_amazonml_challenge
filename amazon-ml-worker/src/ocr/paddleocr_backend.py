"""
Amazon ML Worker — PaddleOCR Backend.

Wraps PaddleOCR for text extraction. Conditionally imported.
"""

from src.ocr.base import OCRBackend
from src.utils.logging import get_logger

log = get_logger(__name__)


class PaddleOCRBackend(OCRBackend):
    """OCR backend using PaddleOCR."""

    def __init__(self):
        self._ocr = None

    @property
    def name(self) -> str:
        return "paddleocr"

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False

    def _init_engine(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def extract_text(self, image_path: str) -> str:
        self._init_engine()
        result = self._ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                lines.append(line[1][0])
        return " ".join(lines).strip()
