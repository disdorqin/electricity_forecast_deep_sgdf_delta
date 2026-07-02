#!/usr/bin/env python
"""Evaluate DeepSGDFDelta predictions and generate go/no-go report.

Usage:
    python scripts/evaluate_deep_sgdf_delta.py --predictions reports/local/deep_sgdf_delta/run_xxx/predictions.csv
    python scripts/evaluate_deep_sgdf_delta.py --help
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from models.deep_sgdf_delta.evaluate import evaluate_predictions, BASELINE_SGDFNET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_deep_sgdf_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSGDFDelta predictions and generate go/no-go report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to same dir as predictions)")
    parser.add_argument("--run-id", type=str, default="eval_run",
                        help="Run ID for report header")
    parser.add_argument("--sgdfnet-baseline", type=float, default=BASELINE_SGDFNET,
                        help=f"SGDFNet baseline sMAPE for comparison (default: {BASELINE_SGDFNET})")
    parser.add_argument("--spike-threshold", type=float, default=500.0,
                        help="Price threshold for spike classification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(pred_df)} predictions from {pred_path}")

    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent

    metrics = evaluate_predictions(
        pred_df,
        run_id=args.run_id,
        output_dir=output_dir,
        sgdfnet_baseline=args.sgdfnet_baseline,
        spike_threshold=args.spike_threshold,
    )

    logger.info(f"Verdict: {metrics['verdict']} — {metrics['verdict_detail']}")
    logger.info(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    main()
