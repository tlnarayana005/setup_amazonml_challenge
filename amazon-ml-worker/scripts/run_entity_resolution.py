"""
Amazon ML Worker — Run Entity Resolution Pipeline.

Usage:
    python scripts/run_entity_resolution.py
    python scripts/run_entity_resolution.py --worker-id 0 --total-workers 4
    python scripts/run_entity_resolution.py --dataset-dir dataset --output-dir output
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config_from_cli
from src.utils.logging import setup_logging, get_logger
from src.utils.seeds import set_seed
from src.utils.timing import Timer


def main():
    config = load_config_from_cli()
    setup_logging(worker_id=config.worker_id)
    log = get_logger("entity_resolution")
    set_seed(config.seed)

    log.info("=" * 60)
    log.info("  Entity Resolution Pipeline")
    log.info("=" * 60)
    log.info("  worker_id:     %d / %d", config.worker_id, config.total_workers)
    log.info("  model_type:    %s", config.model_type)
    log.info("  seed:          %d", config.seed)
    log.info("=" * 60)

    from src.er.pipeline import run_entity_resolution

    # Use dataset_dir and output_dir from the task YAML or defaults
    dataset_dir = "dataset"
    output_dir = "output"

    metrics = run_entity_resolution(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        model_type=config.model_type,
        model_params=config.model_params or {},
        worker_id=config.worker_id,
        total_workers=config.total_workers,
        seed=config.seed,
    )

    # Save metrics
    out_path = Path(output_dir) / "pipeline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("Pipeline complete. Metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
