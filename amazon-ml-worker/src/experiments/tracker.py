"""
Amazon ML Worker — Experiment Tracker.

Lightweight CSV-based experiment tracking. No external service required.
Tracks experiment_id, worker_id, metrics, params, runtime, etc.
Results are mergeable across all 20 machines.
"""

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


TRACKER_COLUMNS = [
    "experiment_id",
    "worker_id",
    "timestamp",
    "task",
    "data_version",
    "code_version",
    "model",
    "features",
    "gpu",
    "runtime_seconds",
    "parameters",
    "training_steps",
    "metric_name",
    "metric_value",
    "output_path",
    "notes",
]


class ExperimentTracker:
    """CSV-based experiment tracker."""

    def __init__(self, results_file: str = "experiments/results.csv"):
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file with header if it doesn't exist
        if not self.results_file.exists():
            with open(self.results_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRACKER_COLUMNS)
                writer.writeheader()

    def log_experiment(
        self,
        worker_id: int = 0,
        task: str = "",
        model: str = "",
        metric_name: str = "",
        metric_value: float = 0.0,
        features: str = "",
        parameters: Optional[Dict] = None,
        training_steps: int = 0,
        runtime_seconds: float = 0.0,
        gpu: str = "",
        data_version: str = "",
        output_path: str = "",
        notes: str = "",
    ) -> str:
        """
        Log an experiment to the CSV file.

        Returns the generated experiment_id.
        """
        experiment_id = self._generate_id(worker_id, task, model)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Try to get git hash
        code_version = self._get_git_hash()

        row = {
            "experiment_id": experiment_id,
            "worker_id": worker_id,
            "timestamp": timestamp,
            "task": task,
            "data_version": data_version,
            "code_version": code_version,
            "model": model,
            "features": features,
            "gpu": gpu,
            "runtime_seconds": round(runtime_seconds, 2),
            "parameters": json.dumps(parameters or {}, default=str),
            "training_steps": training_steps,
            "metric_name": metric_name,
            "metric_value": round(metric_value, 6),
            "output_path": output_path,
            "notes": notes,
        }

        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACKER_COLUMNS)
            writer.writerow(row)

        log.info(
            "Logged experiment %s: %s=%s (model=%s)",
            experiment_id, metric_name, metric_value, model,
        )
        return experiment_id

    def get_results(self) -> pd.DataFrame:
        """Load all experiment results."""
        if not self.results_file.exists():
            return pd.DataFrame(columns=TRACKER_COLUMNS)
        return pd.read_csv(self.results_file)

    def _generate_id(self, worker_id: int, task: str, model: str) -> str:
        """Generate a unique experiment ID."""
        raw = f"{worker_id}-{task}-{model}-{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _get_git_hash(self) -> str:
        """Try to get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"


def merge_experiment_results(
    result_files: list,
    output_file: str,
) -> pd.DataFrame:
    """Merge experiment result CSVs from multiple workers."""
    dfs = []
    for f in result_files:
        if Path(f).exists():
            dfs.append(pd.read_csv(f))
    if not dfs:
        return pd.DataFrame(columns=TRACKER_COLUMNS)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["experiment_id"])
    merged = merged.sort_values("timestamp")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    log.info("Merged %d experiment results -> %s", len(merged), output_file)
    return merged
