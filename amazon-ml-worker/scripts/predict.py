"""Run batch prediction."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.models.baseline import BaselineModel
from src.inference.predict import batch_predict
from src.utils.logging import setup_logging

def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved model")
    p.add_argument("--input", required=True, help="Input CSV/Parquet")
    p.add_argument("--output", default="outputs/predictions/predictions.parquet")
    p.add_argument("--id-column", default="id")
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()

    model = BaselineModel.load(args.model)
    df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_parquet(args.input)
    result = batch_predict(df, model, id_column=args.id_column, batch_size=args.batch_size, output_file=args.output)
    print(f"\nPredictions: {len(result)} rows → {args.output}")

if __name__ == "__main__":
    main()
