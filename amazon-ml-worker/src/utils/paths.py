"""
Amazon ML Worker — Path Management.

Standardised paths and directory creation for worker outputs.
"""

from pathlib import Path
from typing import Optional


def project_root() -> Path:
    """Walk up from CWD looking for pyproject.toml."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def ensure_dir(*parts: str) -> Path:
    """Join path parts, create directory, and return it."""
    p = Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def worker_output_dir(
    base_output: str,
    worker_id: int,
) -> Path:
    """Create and return ``<base>/worker_<id>/``."""
    d = ensure_dir(base_output, f"worker_{worker_id}")
    return d


def resolve_path(path_str: str, root: Optional[Path] = None) -> Path:
    """Resolve a potentially relative path against *root* (default: project root)."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    root = root or project_root()
    return root / p
