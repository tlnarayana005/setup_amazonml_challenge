"""
Amazon ML Worker — Cache Manager.

Versioned caching with data-version and config-version tracking.
Workers skip already-completed IDs on restart.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


class CacheManager:
    """
    Manages cached processing results.

    Each cache entry is identified by (cache_key, data_version, config_hash).
    Workers can check which IDs are already cached and skip them.
    """

    def __init__(self, cache_dir: str, cache_key: str = "default"):
        self.cache_dir = Path(cache_dir)
        self.cache_key = cache_key
        self._dir = self.cache_dir / cache_key
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "cache_meta.json"

    def get_cached_ids(self) -> Set[str]:
        """Return the set of IDs that have been successfully cached."""
        data_path = self._dir / "cached_data.parquet"
        if not data_path.exists():
            return set()
        try:
            df = pd.read_parquet(data_path, columns=["id"])
            return set(df["id"].astype(str))
        except Exception:
            return set()

    def get_cached_data(self) -> Optional[pd.DataFrame]:
        """Load the cached DataFrame, if it exists."""
        data_path = self._dir / "cached_data.parquet"
        if not data_path.exists():
            return None
        try:
            return pd.read_parquet(data_path)
        except Exception as e:
            log.warning("Failed to load cache: %s", e)
            return None

    def save_to_cache(
        self,
        df: pd.DataFrame,
        config_hash: str = "",
        data_version: str = "",
        append: bool = True,
    ) -> None:
        """
        Save results to cache.

        If append=True, merge with existing cached data (deduplicating by 'id').
        """
        data_path = self._dir / "cached_data.parquet"

        if append and data_path.exists():
            existing = pd.read_parquet(data_path)
            df = pd.concat([existing, df], ignore_index=True)
            if "id" in df.columns:
                df = df.drop_duplicates(subset=["id"], keep="last")

        df.to_parquet(data_path, index=False)

        # Save metadata
        meta = {
            "cache_key": self.cache_key,
            "config_hash": config_hash,
            "data_version": data_version,
            "row_count": len(df),
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log.info(
            "Cached %d rows for '%s' (version=%s)",
            len(df), self.cache_key, data_version or "unset",
        )

    def filter_uncached(
        self,
        df: pd.DataFrame,
        id_column: str = "id",
    ) -> pd.DataFrame:
        """Return only rows whose IDs are NOT already cached."""
        cached_ids = self.get_cached_ids()
        if not cached_ids:
            return df

        if id_column not in df.columns:
            log.warning("ID column '%s' not found; cannot filter cached.", id_column)
            return df

        mask = ~df[id_column].astype(str).isin(cached_ids)
        n_skip = (~mask).sum()
        if n_skip > 0:
            log.info("Skipping %d already-cached IDs.", n_skip)
        return df[mask].reset_index(drop=True)

    def is_valid(self, config_hash: str = "", data_version: str = "") -> bool:
        """Check if cache metadata matches current configuration."""
        if not self._meta_path.exists():
            return False
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if config_hash and meta.get("config_hash") != config_hash:
                return False
            if data_version and meta.get("data_version") != data_version:
                return False
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Remove all cached data."""
        import shutil
        if self._dir.exists():
            shutil.rmtree(self._dir)
            self._dir.mkdir(parents=True, exist_ok=True)
            log.info("Cache cleared: %s", self.cache_key)


def compute_config_hash(config_dict: dict) -> str:
    """Compute a stable hash of the configuration for cache invalidation."""
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
