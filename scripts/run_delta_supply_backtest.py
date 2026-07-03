#!/usr/bin/env python
"""Multi-month walk-forward backtest for DeltaSupply module.

Usage:
    python scripts/run_delta_supply_backtest.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-months 2026-01,2026-02,2026-03,2026-04,2026-05 \
        --out-dir reports/local/delta_supply/backtest_2026_01_05
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.delta_supply_targets import (
    DeltaSupplyThresholds,
    compute_delta_supply_targets,
)
from models.deep_sgdf_delta.delta_supply_features import build_delta_supply_features
from models.deep_sgdf_delta.delta_supply_model import DeltaSupplyConfig, DeltaSupplyModel
from scripts.train_delta_supply_module import load_data, split_train_test
from scripts.evaluate_delta_supply_module import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_smape_floor50,
    run_correction_simulation,
    determine_go_nogo,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DeltaSupply multi-month backtest")
    p.add_argument("--data-path", required=True)
    p.add_argument("--target-months", required=True, help="Comma-separated YYYY-MM")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold-upward", type=float, default=100.0)
    p.add_argument("--threshold-downward", type=float, default=-100.0)
    p.add_argument("--threshold-abs-large", type=float, default=150.0)
    p.add_argument("--clip", type=float, default=500.0)
    p.add_argument("--mode", default="FULL_DAY")
    return p.parse_args()


def run_single_month(
    raw_df: pd.DataFrame,
    target_month: str,
    thresholds: DeltaSupplyThresholds,
    mode: str,
    correction_weights: list,
) -> dict:
    """Run train+eval for a single target month."""
    logger.info("=== Backtest month: %s ===", target_month)

    # Compute targets
    target_result = compute_delta_supply_targets(raw_df, thresholds=thresholds)

    # Build features
    feature_result = build_delta_supply_features(raw_df, mode=mode)
    feature_cols = [c for c in feature_result.feature_columns if c in feature_result.df.columns]

    # Merge
    work = feature_result.df.copy()
    for col in ["price_delta", "upward_deviation_label", "downward_deviation_label",
                 "large_abs_deviation_label", "deviation_magnitude_target"]:
        if col in target_result.df.columns and col not in work.columns:
            work[col] = target_result.df[col].values

    # Split
    train_mask, val_mask, test_mask = split_train_test(work, target_month)

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        logger.warning("Insufficient data for %s (train=%d, test=%d)",
                       target_month, train_mask.sum(), test_mask.sum())
        return {"month": target_month, "status": "insufficient_data"}

    # Train
    X_train = work.loc[train_mask, feature_cols].fillna(0)
    y_up = work.loc[train_mask, "upward_deviation_label"].values
    y_down = work.loc[train_mask, "downward_deviation_label"].values
    y_large = work.loc[train_mask, "large_abs_deviation_label"].values
    y_mag = work.loc[train_mask, "deviation_magnitude_target"].values

    model = DeltaSupplyModel(DeltaSupplyConfig())
    model.fit(X_train, y_up, y_down, y_large, y_mag)

    # Predict test
    X_test = work.loc[test_mask, feature_cols].fillna(0)
    pred = model.predict(X_test)
    pred_df = pred.df.copy()
    pred_df["ds"] = work.loc[test_mask, "ds"].values
    pred_df["da_anchor"] = work.loc[test_mask, "da_anchor"].values
    pred_df["rt_actual"] = work.loc[test_mask, "rt_actual"].values
    pred_df["price_delta"] = work.loc[test_mask, "price_delta"].values
    pred_df["upward_deviation_label"] = work.loc[test_mask, "upward_deviation_label"].values
    pred_df["downward_deviation_label"] = work.loc[test_mask, "downward_deviation_label"].values
    pred_df["large_abs_deviation_label"] = work.loc[test_mask, "large_abs_deviation_label"].values
    pred_df["deviation_magnitude_target"] = work.loc[test_mask, "deviation_magnitude_target"].values
    pred_df["period"] = work.loc[test_mask, "period"].values
    pred_df["hour_business"] = work.loc[test_mask, "hour_business"].values

    # Classification metrics
    class_metrics = {}
    for label, prob_col in [
        ("upward_deviation_label", "upward_deviation_prob"),
        ("downward_deviation_label", "downward_deviation_prob"),
        ("large_abs_deviation_label", "large_abs_deviation_prob"),
    ]:
        m = compute_classification_metrics(pred_df[label].values, pred_df[prob_col].values, label)
        class_metrics[label] = m

    # Regression metrics
    reg_metrics = compute_regression_metrics(
        pred_df["deviation_magnitude_target"].values,
        pred_df["deviation_magnitude_pred"].values,
    )

    # Correction simulation
    correction_df = run_correction_simulation(pred_df, correction_weights)
    verdict, reason = determine_go_nogo(correction_df)

    best_idx = correction_df["improvement_pp"].idxmax() if not correction_df.empty else 0
    best_weight = correction_df.loc[best_idx, "correction_weight"] if not correction_df.empty else 0
    best_improvement = correction_df.loc[best_idx, "improvement_pp"] if not correction_df.empty else 0
    da_smape = correction_df.loc[best_idx, "da_anchor_smape"] if not correction_df.empty else np.nan

    # Feature importance
    fi = model.get_feature_importance()

    return {
        "month": target_month,
        "status": "ok",
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_features": len(feature_cols),
        "feature_audit_verdict": feature_result.audit.verdict,
        "classification": class_metrics,
        "regression": reg_metrics,
        "best_correction_weight": best_weight,
        "da_anchor_smape": float(da_smape) if not np.isnan(da_smape) else None,
        "improvement_pp": best_improvement,
        "verdict": verdict,
        "reason": reason,
        "pred_df": pred_df,
        "feature_importance": fi,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_months = [m.strip() for m in args.target_months.split(",")]
    correction_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    thresholds = DeltaSupplyThresholds(
        upward=args.threshold_upward,
        downward=args.threshold_downward,
        abs_large=args.threshold_abs_large,
        clip=args.clip,
    )

    raw_df = load_data(args.data_path)

    monthly_results = []
    all_pred_dfs = []
    all_fis = []

    for month in target_months:
        result = run_single_month(raw_df, month, thresholds, args.mode, correction_weights)
        monthly_results.append(result)
        if result["status"] == "ok":
            all_pred_dfs.append(result.pop("pred_df"))
            all_fis.append(result.pop("feature_importance"))

    # ── Leaderboard ───────────────────────────────────────────────────
    leaderboard = []
    for r in monthly_results:
        if r["status"] == "ok":
            leaderboard.append({
                "month": r["month"],
                "da_anchor_smape": r["da_anchor_smape"],
                "best_weight": r["best_correction_weight"],
                "improvement_pp": r["improvement_pp"],
                "verdict": r["verdict"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            })
    pd.DataFrame(leaderboard).to_csv(out_dir / "leaderboard.csv", index=False)

    # ── Monthly metrics ───────────────────────────────────────────────
    monthly_rows = []
    for r in monthly_results:
        if r["status"] == "ok":
            for label, m in r["classification"].items():
                if isinstance(m, dict) and "f1" in m:
                    monthly_rows.append({
                        "month": r["month"], "target": label,
                        "f1": m["f1"], "roc_auc": m.get("roc_auc", np.nan),
                        "precision": m.get("precision", np.nan),
                        "recall": m.get("recall", np.nan),
                    })
    pd.DataFrame(monthly_rows).to_csv(out_dir / "monthly_metrics.csv", index=False)

    # ── Period / bucket metrics ───────────────────────────────────────
    if all_pred_dfs:
        all_preds = pd.concat(all_pred_dfs, ignore_index=True)

        period_rows = []
        for period, group in all_preds.groupby("period"):
            smape = compute_smape_floor50(group["rt_actual"].values, group["da_anchor"].values)
            period_rows.append({"period": period, "da_anchor_smape": smape, "n": len(group)})
        pd.DataFrame(period_rows).to_csv(out_dir / "period_metrics.csv", index=False)

        bucket_rows = []
        for bucket_name, mask_fn in [
            ("normal", lambda x: (x >= 0) & (x < 500)),
            ("negative", lambda x: x < 0),
            ("spike", lambda x: x >= 500),
        ]:
            mask = mask_fn(all_preds["rt_actual"].values)
            if mask.any():
                smape = compute_smape_floor50(
                    all_preds.loc[mask, "rt_actual"].values,
                    all_preds.loc[mask, "da_anchor"].values,
                )
                bucket_rows.append({"bucket": bucket_name, "da_anchor_smape": smape, "n": int(mask.sum())})
        pd.DataFrame(bucket_rows).to_csv(out_dir / "bucket_metrics.csv", index=False)

    # ── Feature importance summary ────────────────────────────────────
    if all_fis:
        combined_fi = all_fis[0].copy()
        for fi in all_fis[1:]:
            for col in fi.columns:
                if col != "feature" and col in combined_fi.columns:
                    combined_fi[col] = (combined_fi[col] + fi[col]) / 2
        combined_fi.to_csv(out_dir / "feature_importance_summary.csv", index=False)

    # ── Champion summary ──────────────────────────────────────────────
    improvements = [r["improvement_pp"] for r in monthly_results if r["status"] == "ok"]
    mean_improvement = np.mean(improvements) if improvements else 0

    if mean_improvement >= 0.01:
        overall_verdict = "STRONG"
    elif mean_improvement >= 0.003:
        overall_verdict = "ACCEPTABLE"
    elif mean_improvement > 0:
        overall_verdict = "LOW_VALUE"
    else:
        overall_verdict = "NO-GO"

    champion = {
        "overall_verdict": overall_verdict,
        "mean_monthly_improvement_pp": float(mean_improvement),
        "n_months": len(target_months),
        "n_successful": len(improvements),
        "monthly_verdicts": {r["month"]: r["verdict"] for r in monthly_results if r["status"] == "ok"},
    }
    with open(out_dir / "champion_summary.json", "w", encoding="utf-8") as f:
        json.dump(champion, f, ensure_ascii=False, indent=2)

    # ── Champion report ───────────────────────────────────────────────
    report = f"""# DeltaSupply Backtest Report

## Overall Verdict: {overall_verdict}

**Mean monthly improvement**: {mean_improvement:.4f} ({mean_improvement*100:.2f}pp)

## Monthly Leaderboard

{pd.DataFrame(leaderboard).to_csv(index=False) if leaderboard else "No successful months."}

## Summary

- Months tested: {len(target_months)}
- Successful: {len(improvements)}
- Overall verdict: {overall_verdict}
"""
    with open(out_dir / "champion_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Backtest complete. Overall verdict: %s", overall_verdict)
    logger.info("Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
