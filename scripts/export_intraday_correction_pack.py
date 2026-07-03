#!/usr/bin/env python
"""Export Intraday correction pack — Phase 10.

Outputs a CSV for the intraday correction, mode=INTRADAY only.
Includes full Phase 10 correction pipeline fields and policy gating.

Usage:
    python scripts/export_intraday_correction_pack.py \\
        --predictions reports/local/phase10/intraday_tracker_stability/predictions.csv \\
        --cutoff-hour 12 --mode eval --policy-enabled \\
        --out reports/local/phase10/intraday_tracker_stability/correction_pack.csv
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

from models.deep_sgdf_delta.intraday_residual_tracker import (
    IntradayResidualState,
    IntradayTrackerConfig,
    apply_intraday_correction,
    compute_intraday_residual_state,
)
from models.deep_sgdf_delta.intraday_tracker_policy import (
    PolicyConfig,
    PolicyDecision,
    evaluate_policy,
)
from models.deep_sgdf_delta.prediction_modes import PredictionMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "business_day", "cutoff_hour", "target_hour",
    "sgdfnet_pred", "hour_business",
]

EVAL_COLUMNS = [
    "y_true", "baseline_error", "corrected_error",
]

ONLINE_COLUMNS = [
    "business_day", "cutoff_hour", "target_hour", "ds", "mode",
    "base_model_name", "base_pred",
    "intraday_base_correction", "intraday_model_weight",
    "intraday_pre_guardrail_correction", "intraday_guardrail_weight",
    "intraday_final_correction", "intraday_corrected_pred",
    "intraday_confidence", "policy_decision", "fusion_weight",
    "shadow_only_flag", "guardrail_reason",
    "observed_hours", "n_observed", "residual_std_today", "bias_direction",
]


def main():
    parser = argparse.ArgumentParser(description="Export Intraday Correction Pack (Phase 10)")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions CSV from evaluate script")
    parser.add_argument("--cutoff-hour", type=int, default=None,
                        help="Filter to specific cutoff hour (optional)")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "online"])
    parser.add_argument("--base-model-name", type=str, default="sgdfnet")
    parser.add_argument("--policy-enabled", action="store_true", default=False,
                        help="Apply Phase 10 policy gating")
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
        out_path = args.out or "reports/local/phase10/intraday_tracker_stability/correction_pack.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=ONLINE_COLUMNS).to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    # Build correction pack with Phase 10 fields
    pack = pd.DataFrame()
    pack["business_day"] = pred_df["business_day"]
    pack["cutoff_hour"] = pred_df["cutoff_hour"]
    pack["target_hour"] = pred_df["target_hour"]
    pack["ds"] = pred_df.get("ds", pred_df["business_day"])
    pack["mode"] = "INTRADAY"
    pack["base_model_name"] = args.base_model_name
    pack["base_pred"] = pred_df["sgdfnet_pred"]

    # Phase 10 correction pipeline fields
    pack["intraday_base_correction"] = pred_df.get("intraday_base_correction", 0.0)
    pack["intraday_model_weight"] = pred_df.get("intraday_model_weight", 0.0)
    pack["intraday_pre_guardrail_correction"] = pred_df.get("intraday_pre_guardrail_correction", 0.0)
    pack["intraday_guardrail_weight"] = pred_df.get("intraday_guardrail_weight", 1.0)
    pack["intraday_final_correction"] = pred_df.get("intraday_final_correction", pred_df.get("intraday_correction", 0.0))
    pack["intraday_corrected_pred"] = pred_df["intraday_corrected_pred"]
    pack["intraday_confidence"] = pred_df.get("confidence", 0.5)
    pack["guardrail_reason"] = pred_df.get("guardrail_reason", "")

    # Policy fields
    if args.policy_enabled and "n_observed" in pred_df.columns:
        # Compute policy per row
        policy_decisions = []
        fusion_weights = []
        shadow_flags = []
        policy_reasons = []
        policy_config = PolicyConfig()

        for _, row in pred_df.iterrows():
            state = IntradayResidualState(
                business_day=pd.Timestamp(row["business_day"]),
                cutoff_hour=int(row["cutoff_hour"]),
                n_observed=int(row.get("n_observed", 0)),
                mean_residual_today=float(row.get("mean_residual_today", 0)),
                median_residual_today=float(row.get("median_residual_today", 0)),
                ewm_residual_today=float(row.get("ewm_residual_today", 0)),
                last_residual=float(row.get("last_residual", 0)),
                residual_std_today=float(row.get("residual_std_today", 0)),
                bias_direction=str(row.get("bias_direction", "unknown")),
                confidence=float(row.get("confidence", 0.5)),
            )
            has_neg_risk = bool(row.get("da_anchor", 0) < 0) if "da_anchor" in pred_df.columns else False
            result = evaluate_policy(state, PredictionMode.INTRADAY, policy_config, has_neg_risk)
            policy_decisions.append(result.policy_decision.value)
            fusion_weights.append(result.fusion_weight)
            shadow_flags.append(result.shadow_only_flag)
            policy_reasons.append(result.reason)

        pack["policy_decision"] = policy_decisions
        pack["fusion_weight"] = fusion_weights
        pack["shadow_only_flag"] = shadow_flags
    else:
        pack["policy_decision"] = "LOW_WEIGHT"
        pack["fusion_weight"] = 0.12
        pack["shadow_only_flag"] = False

    # Observed state fields
    pack["observed_hours"] = pred_df.get("observed_hours", "[]")
    pack["n_observed"] = pred_df.get("n_observed", 0)
    pack["residual_std_today"] = pred_df.get("residual_std_today", 0.0)
    pack["bias_direction"] = pred_df.get("bias_direction", "unknown")

    # Eval mode: add ground truth
    if args.mode == "eval" and "rt_actual" in pred_df.columns:
        pack["y_true"] = pred_df["rt_actual"]
        pack["baseline_error"] = pred_df.get("baseline_error", np.nan)
        pack["corrected_error"] = pred_df.get("corrected_error", np.nan)

    # Output
    out_path = args.out or str(Path(args.predictions).parent / "correction_pack.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pack.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Correction pack written to %s (%d rows, mode=%s, policy=%s)",
                out_path, len(pack), args.mode, args.policy_enabled)
    logger.info("Mode: INTRADAY (full-day pack NOT exported)")


if __name__ == "__main__":
    main()
