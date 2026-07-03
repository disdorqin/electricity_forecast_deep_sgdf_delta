#!/usr/bin/env python
"""Export Solar916 correction pack for 9_16 hours.

Outputs a CSV that can be consumed by the ledger/fusion system.

Modes:
  --mode eval   : includes y_true and residual_true (for offline evaluation)
  --mode online  : no y_true (for production use)

Output fields:
  business_day, hour_business, ds, base_model_name, base_pred,
  solar916_residual_pred, solar916_corrected_pred, solar916_confidence,
  solar916_trigger_flag, feature_missing_flag, correction_reason

Usage:
    python scripts/export_solar916_correction_pack.py \\
        --predictions reports/local/phase7/solar916/predictions.csv \\
        --mode eval --out reports/local/phase7/solar916/correction_pack.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export Solar916 Correction Pack")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to Solar916 predictions.csv")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to metrics_summary.json (for confidence)")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "online"])
    parser.add_argument("--base-model-name", type=str, default="sgdfnet")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--apply-guardrail", action="store_true", default=False,
                        help="Apply Solar916 guardrail before exporting")
    parser.add_argument("--max-abs-correction", type=float, default=100.0)
    parser.add_argument("--disabled-hours", type=str, default="",
                        help="Comma-separated hours to disable, e.g. '9,11'")
    parser.add_argument("--negative-risk-weight", type=float, default=0.3)
    args = parser.parse_args()

    # Load predictions
    pred_df = pd.read_csv(args.predictions, encoding="utf-8-sig")
    logger.info("Loaded %d predictions", len(pred_df))

    # Load metrics if available
    metrics = {}
    if args.metrics and Path(args.metrics).exists():
        with open(args.metrics, encoding="utf-8") as f:
            metrics = json.load(f)

    # Apply guardrail if requested
    guarded_metrics = None
    if args.apply_guardrail:
        from models.deep_sgdf_delta.solar916_guardrail import (
            Solar916GuardrailConfig,
            apply_guardrail,
            compute_guarded_metrics,
        )
        disabled_hours = []
        if args.disabled_hours:
            disabled_hours = [int(h.strip()) for h in args.disabled_hours.split(",") if h.strip()]
        gr_config = Solar916GuardrailConfig(
            max_abs_correction=args.max_abs_correction,
            disabled_hours=disabled_hours,
            negative_risk_weight=args.negative_risk_weight,
        )
        pred_df = apply_guardrail(pred_df, gr_config)
        guarded_metrics = compute_guarded_metrics(pred_df)
        # Override correction columns with guarded versions
        pred_df["solar916_residual_pred"] = pred_df["solar916_residual_pred_after_guardrail"]
        pred_df["solar916_corrected_pred"] = pred_df["solar916_corrected_pred"]
        logger.info("Guardrail applied. Guarded metrics: %s", guarded_metrics["overall"])

    # Build correction pack
    pack = pd.DataFrame()
    pack["business_day"] = pred_df["business_day"]
    pack["hour_business"] = pred_df["hour_business"]
    pack["ds"] = pred_df.get("ds", pred_df["business_day"])
    pack["base_model_name"] = args.base_model_name
    pack["base_pred"] = pred_df["sgdfnet_pred"]
    pack["solar916_residual_pred"] = pred_df["solar916_residual_pred"]
    pack["solar916_corrected_pred"] = pred_df["solar916_corrected_pred"]

    # Confidence: based on validation sMAPE (lower is better → higher confidence)
    val_smape = metrics.get("val_smape", 50.0)
    confidence = max(0.0, min(1.0, 1.0 - val_smape / 100.0))
    pack["solar916_confidence"] = confidence

    # Trigger flag: always True for 9_16 hours (this model only operates in 9_16)
    pack["solar916_trigger_flag"] = True

    # Feature missing flag
    missing = metrics.get("missing_features", [])
    pack["feature_missing_flag"] = len(missing) > 0

    # Correction reason
    pack["correction_reason"] = "solar_volatility_916_residual_correction"

    # Eval mode: add ground truth
    if args.mode == "eval":
        if "rt_actual" in pred_df.columns:
            pack["y_true"] = pred_df["rt_actual"]
            pack["residual_true"] = pred_df["rt_actual"] - pred_df["sgdfnet_pred"]

    # Only 9_16
    assert (pred_df["period"] == "9_16").all() if "period" in pred_df.columns else True

    # Output
    out_path = args.out or str(Path(args.predictions).parent / "correction_pack.csv")
    pack.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Correction pack written to %s (%d rows, mode=%s)",
                out_path, len(pack), args.mode)

    # Write guarded metrics if guardrail was applied
    if guarded_metrics is not None:
        out_dir = Path(out_path).parent
        # guarded_metrics_summary.json
        with open(out_dir / "guarded_metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(guarded_metrics["overall"], f, ensure_ascii=False, indent=2)
        # guarded_hourly_metrics.csv
        pd.DataFrame(guarded_metrics["hourly"]).to_csv(
            out_dir / "guarded_hourly_metrics.csv", index=False, encoding="utf-8-sig"
        )
        # guarded_bucket_metrics.csv
        pd.DataFrame(guarded_metrics["buckets"]).to_csv(
            out_dir / "guarded_bucket_metrics.csv", index=False, encoding="utf-8-sig"
        )
        logger.info("Guarded metrics written to %s", out_dir)


if __name__ == "__main__":
    main()
