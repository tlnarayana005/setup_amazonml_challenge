"""Test OCR interface (backends may not be installed)."""
import pytest
from src.ocr.base import OCRBackend, get_ocr_backend
from src.ocr.tesseract_backend import TesseractBackend
from src.ocr.paddleocr_backend import PaddleOCRBackend


def test_tesseract_interface():
    backend = TesseractBackend()
    assert backend.name == "tesseract"
    assert isinstance(backend.is_available(), bool)


def test_paddleocr_interface():
    backend = PaddleOCRBackend()
    assert backend.name == "paddleocr"
    assert isinstance(backend.is_available(), bool)


def test_get_backend_tesseract():
    backend = get_ocr_backend("tesseract")
    assert isinstance(backend, TesseractBackend)


def test_get_backend_paddleocr():
    backend = get_ocr_backend("paddleocr")
    assert isinstance(backend, PaddleOCRBackend)


def test_get_backend_invalid():
    with pytest.raises(ValueError):
        get_ocr_backend("nonexistent")
