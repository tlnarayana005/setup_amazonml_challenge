"""Test configuration system."""
import os
import pytest
import tempfile
from pathlib import Path

from src.config.loader import WorkerConfig, load_config, _deep_merge


class TestWorkerConfig:
    def test_defaults(self):
        cfg = WorkerConfig()
        assert cfg.worker_id == 0
        assert cfg.total_workers == 1
        assert cfg.task == "etl"
        assert cfg.seed == 42
        assert cfg.enable_ocr is False
        assert cfg.max_rows == -1
        assert cfg.batch_size == 32

    def test_to_dict(self):
        cfg = WorkerConfig(worker_id=5, task="ocr")
        d = cfg.to_dict()
        assert d["worker_id"] == 5
        assert d["task"] == "ocr"
        assert isinstance(d, dict)

    def test_custom_values(self):
        cfg = WorkerConfig(
            worker_id=3, total_workers=10, task="cv",
            enable_cv=True, max_rows=5000, metric="f1_weighted",
        )
        assert cfg.worker_id == 3
        assert cfg.total_workers == 10
        assert cfg.enable_cv is True
        assert cfg.max_rows == 5000
        assert cfg.metric == "f1_weighted"


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result["a"] == {"x": 1, "y": 99, "z": 100}
        assert result["b"] == 3


class TestLoadConfig:
    def test_load_with_cli_args(self):
        cfg = load_config(cli_args=["--worker-id", "5", "--task", "ocr", "--seed", "123"])
        assert cfg.worker_id == 5
        assert cfg.task == "ocr"
        assert cfg.seed == 123

    def test_load_default(self):
        cfg = load_config(cli_args=[])
        assert isinstance(cfg, WorkerConfig)
        assert cfg.worker_id == 0

    def test_env_override(self):
        os.environ["WORKER_ID"] = "7"
        os.environ["WORKER_TASK"] = "nlp"
        try:
            cfg = load_config(cli_args=[])
            assert cfg.worker_id == 7
            assert cfg.task == "nlp"
        finally:
            del os.environ["WORKER_ID"]
            del os.environ["WORKER_TASK"]
