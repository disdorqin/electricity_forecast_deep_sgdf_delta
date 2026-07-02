#!/usr/bin/env python
"""P0: Reproduce SGDFNet baseline metrics for DeepSGDFDelta comparison.

Runs SGDFNet's Protocol B cutoff walk-forward and generates:
  reports/local/deep_sgdf_delta/baseline_sgdfnet/metrics_summary.json
  reports/local/deep_sgdf_delta/baseline_sgdfnet/monthly_metrics.csv

Usage:
    python scripts/p0_reproduce_sgdfnet_baseline.py
    python scripts/p0_reproduce_sgdfnet_baseline.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ORIG_PROJECT = Path(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp")
_ORIG_SGDFNET = _ORIG_PROJECT / "SGDFNet" / "src"
if _ORIG_SGDFNET.exists() and str(_ORIG_SGDFNET) not in sys.path:
    sys.path.insert(0, str(_ORIG_SGDFNET))

from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment  # noqa: E402
from sgdfnet.metrics import build_metrics_frame, build_segment_metrics  # noqa: E402

from models.deep_sgdf_delta.metrics import compute_full_metrics, compute_monthly_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("p0_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce SGDFNet corrected cutoff-safe baseline for comparison",
    )
    parser.add_argument("--config", type=str,
                        default=None,
                        help="SGDFNet config YAML path")
    parser.add_argument("--output-dir", type=str,
                        default="reports/local/deep_sgdf_delta/baseline_sgdfnet",
                        help="Output directory for baseline metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default SGDFNet config path
    if args.config:
        sgdfnet_config = Path(args.config)
    else:
        sgdfnet_config = _ORIG_PROJECT / "SGDFNet" / "configs" / "cutoff_recovery_2026_diag_a_prune_actualside.yaml"

    if not sgdfnet_config.exists():
        raise FileNotFoundError(f"SGDFNet config not found: {sgdfnet_config}")

    logger.info(f"Running SGDFNet baseline with config: {sgdfnet_config}")

    # Run SGDFNet Protocol B cutoff experiment
    run_dir = run_protocol_b_cutoff_experiment(str(sgdfnet_config))
    logger.info(f"SGDFNet experiment output: {run_dir}")

    # Load predictions
    pred_df = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    logger.info(f"Loaded {len(pred_df)} SGDFNet predictions")

    # Rename columns for compatibility with our metrics
    pred_df = pred_df.rename(columns={
        "rt_hat": "rt_pred",
        "rt_actual": "y_true",
        "delta_hat": "delta_pred",
        "delta_target": "delta_target",
        "hour": "hour",
        "da_anchor": "da_anchor",
        "segment": "period",
        "timestamp": "ds",
    })

    # Compute metrics
    metrics_df = pred_df.copy()
    if "rt_actual" not in metrics_df.columns and "y_true" in metrics_df.columns:
        metrics_df["rt_actual"] = metrics_df["y_true"]
    if "rt_pred" not in metrics_df.columns and "y_true" in metrics_df.columns:
        metrics_df["rt_pred"] = metrics_df["rt_pred"]

    full_metrics = compute_full_metrics(metrics_df)
    monthly_df = compute_monthly_metrics(metrics_df)

    if not monthly_df.empty:
        full_metrics["monthly_avg_sMAPE_floor50"] = float(monthly_df["sMAPE_floor50"].mean())

    # Output
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, ensure_ascii=False, indent=2)

    if not monthly_df.empty:
        monthly_df.to_csv(output_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")

    logger.info("SGDFNet Baseline Metrics:")
    for k, v in full_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    logger.info(f"Baseline metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
