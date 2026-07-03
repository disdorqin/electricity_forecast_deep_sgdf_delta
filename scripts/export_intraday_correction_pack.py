#!/usr/bin/env python
"""Export Intraday correction pack — Phase 9.

Outputs a CSV for the intraday correction, mode=INTRADAY only.
Does NOT output full-day packs.

Usage:
    python scripts/export_intraday_correction_pack.py \\
        --predictions reports/local/phase9/intraday_tracker/predictions.csv \\
        --cutoff-hour 12 --mode eval \\
        --out reports/local/phase9/intraday_tracker/correction_pack.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export Intraday Correction Pack")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--cutoff-hour", type=int, default=None,
                        help="Filter to specific cutoff hour (optional)")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "online"])
    parser.add_argument("--base-model-name", type=str, default="sgdfnet")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions, encoding="utf-8-sig")
    logger.info("Loaded %d predictions", len(pred_df))

    # Filter by cutoff hour if specified
    if args.cutoff_hour is not None:
        pred_df = pred_df[pred_df["cutoff_hour"] == args.cutoff_hour].copy()
        logger.info("Filtered to cutoff_hour=%d: %d rows", args.cutoff_hour, len(pred_df))

    if len(pred_df) == 0:
        logger.warning("No predictions to export. Writing empty pack.")
        out_path = args.out or "reports/local/phase9/intraday_tracker/correction_pack.csv"
        pd.DataFrame(columns=[
            "business_day", "cutoff_hour", "target_hour", "ds",
            "base_model_name", "base_pred", "intraday_residual_state",
            "intraday_correction", "intraday_corrected_pred",
            "intraday_confidence", "intraday_trigger_flag",
            "guardrail_reason", "mode",
        ]).to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    # Build correction pack
    pack = pd.DataFrame()
    pack["business_day"] = pred_df["business_day"]
    pack["cutoff_hour"] = pred_df["cutoff_hour"]
    pack["target_hour"] = pred_df["target_hour"]
    pack["ds"] = pred_df.get("ds", pred_df["business_day"])
    pack["base_model_name"] = args.base_model_name
    pack["base_pred"] = pred_df["sgdfnet_pred"]

    # Residual state summary
    pack["intraday_residual_state"] = pred_df.apply(
        lambda r: json.dumps({
            "n_observed": int(r.get("n_observed", 0)),
            "bias_direction": r.get("bias_direction", "unknown"),
        }), axis=1
    )
    pack["intraday_correction"] = pred_df["intraday_correction"]
    pack["intraday_corrected_pred"] = pred_df["intraday_corrected_pred"]
    pack["intraday_confidence"] = pred_df.get("confidence", 0.5)
    pack["intraday_trigger_flag"] = True  # always triggered for INTRADAY
    pack["guardrail_reason"] = pred_df.get("guardrail_reason", "")
    pack["mode"] = "INTRADAY"  # always INTRADAY

    # Eval mode: add ground truth
    if args.mode == "eval" and "rt_actual" in pred_df.columns:
        pack["y_true"] = pred_df["rt_actual"]

    # Output
    out_path = args.out or str(Path(args.predictions).parent / "correction_pack.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pack.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Correction pack written to %s (%d rows, mode=%s)",
                out_path, len(pack), args.mode)
    logger.info("Mode: INTRADAY (full-day pack NOT exported)")


if __name__ == "__main__":
    main()
