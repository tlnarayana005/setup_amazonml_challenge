"""Evaluate predictions against ground truth."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd, numpy as np
from src.evaluation.metrics import compute_metrics, format_metrics_report
from src.utils.logging import setup_logging

def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--id-column", default="id")
    p.add_argument("--target-column", default="target")
    p.add_argument("--pred-column", default="prediction")
    p.add_argument("--metric", default="f1_macro")
    args = p.parse_args()

    preds = pd.read_csv(args.predictions) if args.predictions.endswith(".csv") else pd.read_parquet(args.predictions)
    truth = pd.read_csv(args.ground_truth) if args.ground_truth.endswith(".csv") else pd.read_parquet(args.ground_truth)

    merged = truth.merge(preds, on=args.id_column)
    metrics = compute_metrics(merged[args.target_column].values, merged[args.pred_column].values, args.metric)
    print(format_metrics_report(metrics, args.metric))

if __name__ == "__main__":
    main()
