"""Test CV image utilities."""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from src.cv.image_utils import load_image, validate_image, resize_image, normalize_image, batch_validate_images


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(path)
    return str(path)


def test_load_image(sample_image):
    img = load_image(sample_image)
    assert img is not None
    assert img.shape == (100, 100, 3)


def test_load_image_resize(sample_image):
    img = load_image(sample_image, target_size=(50, 50))
    assert img.shape == (50, 50, 3)


def test_load_image_missing():
    img = load_image("/nonexistent/path.png")
    assert img is None


def test_validate_image(sample_image):
    result = validate_image(sample_image)
    assert result["valid"] is True
    assert result["width"] == 100
    assert result["height"] == 100


def test_validate_invalid(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    result = validate_image(str(bad))
    assert result["valid"] is False


def test_resize_image():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    resized = resize_image(img, (50, 50), keep_aspect=False)
    assert resized.shape == (50, 50, 3)


def test_normalize_image():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 128
    normed = normalize_image(img)
    assert np.issubdtype(normed.dtype, np.floating)
    assert normed.shape == (10, 10, 3)


def test_batch_validate(sample_image, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("corrupt")
    result = batch_validate_images([sample_image, str(bad)], ids=["good", "bad"])
    assert result["valid"] == 1
    assert result["invalid"] == 1
