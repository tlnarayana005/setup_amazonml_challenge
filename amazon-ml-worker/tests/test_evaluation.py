"""Test evaluation metrics."""
import pytest
import numpy as np
from src.evaluation.metrics import compute_metrics, compute_confusion_matrix, format_metrics_report
from src.evaluation.threshold import search_threshold


def test_compute_metrics_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_macro"] <= 1


def test_compute_metrics_binary():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    metrics = compute_metrics(y_true, y_pred, metric="f1_binary")
    assert "f1_binary" in metrics


def test_compute_metrics_single():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_metrics(y_true, y_pred, metric="accuracy", include_all=False)
    assert "accuracy" in metrics
    assert len(metrics) == 1


def test_confusion_matrix():
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 2])
    result = compute_confusion_matrix(y_true, y_pred)
    assert "matrix" in result
    assert "labels" in result
    assert len(result["matrix"]) == 3


def test_format_report():
    metrics = {"accuracy": 0.85, "f1_macro": 0.82}
    report = format_metrics_report(metrics, "f1_macro")
    assert "f1_macro" in report
    assert "<-- primary" in report


def test_threshold_search():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.3, 0.7, 0.8, 0.6, 0.2, 0.9, 0.4])
    result = search_threshold(y_true, y_proba)
    assert "best_threshold" in result
    assert "best_score" in result
    assert 0 < result["best_threshold"] < 1
