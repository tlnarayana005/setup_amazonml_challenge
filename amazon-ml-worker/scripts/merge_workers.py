"""Merge worker outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.merge import merge_worker_outputs
from src.utils.logging import setup_logging, get_logger

def main():
    setup_logging()
    log = get_logger("merge")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="outputs/worker_results")
    p.add_argument("--output-file", default="outputs/merged_output.parquet")
    p.add_argument("--id-column", default="id")
    p.add_argument("--expected-workers", type=int, default=None)
    args = p.parse_args()

    merged = merge_worker_outputs(
        args.input_dir, id_column=args.id_column,
        output_file=args.output_file, expected_workers=args.expected_workers,
    )
    log.info("Merged %d rows → %s", len(merged), args.output_file)

if __name__ == "__main__":
    main()
