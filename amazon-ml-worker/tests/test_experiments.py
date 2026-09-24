"""Test experiment tracker."""
import pytest
import pandas as pd
from src.experiments.tracker import ExperimentTracker


def test_tracker_create(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "results.csv"))
    assert (tmp_path / "results.csv").exists()


def test_log_experiment(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "results.csv"))
    exp_id = tracker.log_experiment(
        worker_id=0, task="baseline", model="logistic",
        metric_name="f1_macro", metric_value=0.85,
        runtime_seconds=12.5, notes="test run",
    )
    assert isinstance(exp_id, str)
    assert len(exp_id) > 0

    results = tracker.get_results()
    assert len(results) == 1
    assert results["metric_value"].iloc[0] == 0.85


def test_multiple_experiments(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "results.csv"))
    for i in range(5):
        tracker.log_experiment(
            worker_id=i, task="test", model=f"model_{i}",
            metric_name="accuracy", metric_value=0.5 + i * 0.1,
        )
    results = tracker.get_results()
    assert len(results) == 5


def test_results_mergeable(tmp_path):
    """Results from different workers should merge cleanly."""
    from src.experiments.tracker import merge_experiment_results

    for wid in range(3):
        tracker = ExperimentTracker(str(tmp_path / f"w{wid}_results.csv"))
        tracker.log_experiment(worker_id=wid, task="test", model="lr", metric_name="acc", metric_value=0.8)

    files = [str(tmp_path / f"w{i}_results.csv") for i in range(3)]
    merged = merge_experiment_results(files, str(tmp_path / "merged.csv"))
    assert len(merged) == 3
