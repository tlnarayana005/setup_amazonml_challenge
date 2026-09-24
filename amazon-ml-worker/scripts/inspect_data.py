"""Inspect data and generate reports."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.inspect import inspect_dataframe, save_inspection_report
from src.utils.logging import setup_logging

def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/train.csv")
    p.add_argument("--output-dir", default="outputs/logs")
    p.add_argument("--id-column", default="id")
    p.add_argument("--target-column", default="target")
    args = p.parse_args()

    path = Path(args.input)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    report = inspect_dataframe(df, id_column=args.id_column, target_column=args.target_column)
    save_inspection_report(report, args.output_dir)
    print(f"\nInspection report saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
