"""
Launcher for Optimized Entity Resolution Pipeline (v2).

Uses pipeline_v2 with rapidfuzz + vectorized ops.

Usage:
    python scripts/run_er_v2.py --task entity_resolution --dataset-dir smoke_dataset --output-dir output_smoke
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config_from_cli
from src.utils.logging import setup_logging, get_logger
from src.utils.seeds import set_seed


def main():
    config = load_config_from_cli()
    setup_logging(worker_id=config.worker_id)
    log = get_logger("entity_resolution_v2")
    set_seed(config.seed)

    dataset_dir = config.dataset_dir
    output_dir = config.output_dir
    max_rows = config.max_rows

    log.info("=" * 60)
    log.info("  Entity Resolution Pipeline V2 (OPTIMIZED)")
    log.info("=" * 60)
    log.info("  dataset_dir:   %s", dataset_dir)
    log.info("  output_dir:    %s", output_dir)
    log.info("  max_rows:      %s", "unlimited" if max_rows < 0 else str(max_rows))
    log.info("  model_type:    %s", config.model_type)
    log.info("=" * 60)

    from src.er.pipeline_v3 import run_entity_resolution

    metrics = run_entity_resolution(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model_type=config.model_type,
        model_params=config.model_params or {},
        worker_id=config.worker_id,
        total_workers=config.total_workers,
        seed=config.seed,
        max_rows=max_rows,
    )

    out_path = Path(output_dir) / "pipeline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("Done. Metrics: %s", out_path)


if __name__ == "__main__":
    main()
