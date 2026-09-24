"""Run ETL pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config_from_cli
from src.etl.pipeline import run_etl
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed

def main():
    config = load_config_from_cli()
    setup_logging(worker_id=config.worker_id)
    set_seed(config.seed)
    df = run_etl(config)
    print(f"\nETL complete: {len(df)} rows processed.")

if __name__ == "__main__":
    main()
