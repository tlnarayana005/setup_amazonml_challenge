"""Test hardware detection."""
import pytest
from src.hardware.detect import detect_hardware, detect_environment_type


def test_detect_hardware():
    report = detect_hardware()
    assert "python_version" in report
    assert "platform" in report
    assert "cpu_count" in report
    assert report["cpu_count"] > 0
    assert "disk_total_gb" in report
    assert "cuda_available" in report
    assert "pandas" in report
    assert "numpy" in report


def test_detect_environment_type():
    env = detect_environment_type()
    assert env in ("kaggle", "colab", "local")
