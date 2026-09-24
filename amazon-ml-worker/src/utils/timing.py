"""
Amazon ML Worker — Runtime Tracking.

Context-manager timer and ETA estimation.
"""

import time
from contextlib import contextmanager
from typing import Dict, Optional


class Timer:
    """Accumulating timer supporting multiple named sections."""

    def __init__(self):
        self._sections: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}
        self._global_start: Optional[float] = None

    def start(self, name: str = "total") -> "Timer":
        self._starts[name] = time.time()
        if self._global_start is None:
            self._global_start = self._starts[name]
        return self

    def stop(self, name: str = "total") -> float:
        elapsed = time.time() - self._starts.pop(name, time.time())
        self._sections[name] = self._sections.get(name, 0.0) + elapsed
        return elapsed

    @contextmanager
    def section(self, name: str):
        """Context manager: ``with timer.section('etl'): ...``"""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def elapsed(self, name: str = "total") -> float:
        """Return accumulated time for *name*."""
        if name in self._starts:
            return self._sections.get(name, 0.0) + (time.time() - self._starts[name])
        return self._sections.get(name, 0.0)

    def total_elapsed(self) -> float:
        if self._global_start is None:
            return 0.0
        return time.time() - self._global_start

    def summary(self) -> Dict[str, float]:
        return {**self._sections, "total_elapsed": self.total_elapsed()}

    def eta(self, done: int, total: int, name: str = "total") -> float:
        """Estimate seconds remaining based on progress."""
        if done <= 0:
            return float("inf")
        rate = self.elapsed(name) / done
        return rate * (total - done)

    def __repr__(self) -> str:
        parts = [f"{k}={v:.2f}s" for k, v in self.summary().items()]
        return f"Timer({', '.join(parts)})"
