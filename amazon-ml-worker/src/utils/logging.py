"""
Amazon ML Worker — Structured Logging.

Provides a configured logger with worker_id prefix for all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] [W%(worker_id)s] [%(levelname)s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


class _WorkerFilter(logging.Filter):
    """Inject worker_id into every log record."""

    def __init__(self, worker_id: int = 0):
        super().__init__()
        self.worker_id = worker_id

    def filter(self, record):
        record.worker_id = self.worker_id
        return True


def setup_logging(
    worker_id: int = 0,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger. Idempotent — safe to call multiple times."""
    global _initialized
    root = logging.getLogger()

    if _initialized:
        # Update filter worker_id if changed
        for f in root.filters:
            if isinstance(f, _WorkerFilter):
                f.worker_id = worker_id
        return root

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    wf = _WorkerFilter(worker_id)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.addFilter(wf)
    root.addHandler(ch)
    root.addFilter(wf)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.addFilter(wf)
        root.addHandler(fh)

    _initialized = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger."""
    return logging.getLogger(name)
