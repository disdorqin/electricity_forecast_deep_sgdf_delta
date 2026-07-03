#!/usr/bin/env python
"""Multi-month walk-forward backtest for Spike Risk module.

Evaluates the spike probability columns as a *risk signal*.
For each target month the script:

1. Splits data into train / val / test (val = 30 days before the target month).
2. Builds features via ``build_spike_risk_features``.
3. Computes targets via ``compute_spike_risk_targets``.
4. Trains a ``SpikeRiskModel``.
5. Selects the best probability threshold on the *validation* set using
   ``risk_calibration.select_threshold_by_objective``.
6. Evaluates on the *test* set: classification metrics, top-k capture,
   and decile calibration.

Targets:
    spike_label         : rt_actual >= 500
    extreme_spike_label : rt_actual >= 800
    relative_spike_label: rt_actual - da_anchor >= 200

Outputs (all written to ``--out-dir``):
    monthly_metrics.csv, target_metrics.csv, topk_metrics.csv,
    calibration_metrics.csv, thresholds.csv,
    feature_importance_summary.csv, spike_risk_backtest_report.md

Verdict:
    SPIKE_CHAMPION    : mean top10 lift >= 2.5, mean recall@top20 >= 0.5,
                        >= 3 sufficient months
    SPIKE_ACCEPTABLE  : mean top10 lift >= 1.8
    SPIKE_LOW_VALUE   : top-k useful but recall unstable
    SPIKE_NO_GO       : otherwise

Usage:
    python scripts/run_spike_risk_backtest.py \\
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \\
        --target-months 2026-01,2026-02,2026-03,2026-04,2026-05 \\
        --out-dir reports/local/risk_modules/spike_risk_backtest_2026_01_05
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
sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.spike_risk_targets import (
    SpikeRiskThresholds,
    compute_spike_risk_targets,
)
from models.deep_sgdf_delta.spike_risk_features import build_spike_risk_features
from models.deep_sgdf_delta.spike_risk_model import SpikeRiskConfig, SpikeRiskModel
from models.deep_sgdf_delta.risk_calibration import (
    select_threshold_by_objective,
    compute_classification_metrics,
    compute_topk_metrics,
    compute_calibration_metrics,
)
from models.deep_sgdf_delta.realtime_column_mapping import rename_chinese_columns
from models.deep_sgdf_delta.metrics import smape_floor50
from scripts.train_delta_supply_module import load_data, split_train_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Risk target specification ────────────────────────────────────────────────
RISK_TARGETS = [
    ("spike", "spike_prob", "spike_label"),
    ("extreme_spike", "extreme_spike_prob", "extreme_spike_label"),
    ("relative_spike", "relative_spike_prob", "relative_spike_label"),
]

MIN_SPIKE_EVENTS = 10


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Spike risk multi-month backtest")
    p.add_argument("--data-path", required=True)
    p.add_argument("--target-months", required=True, help="Comma-separated YYYY-MM")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold-spike", type=float, default=500.0)
    p.add_argument("--threshold-extreme-spike", type=float, default=800.0)
    p.add_argument("--threshold-relative-spike", type=float, default=200.0)
    p.add_argument("--mode", default="FULL_DAY")
    p.add_argument("--val-days", type=int, default=30)
    p.add_argument("--objective", default="f1",
                   choices=["f1", "precision_recall", "recall_precision"])
    return p.parse_args()


# ── Single-month runner ──────────────────────────────────────────────────────

def run_single_month(
    raw_df: pd.DataFrame,
    target_month: str,
    thresholds: SpikeRiskThresholds,
    mode: str,
    val_days: int,
    objective: str,
    out_dir: Path,
) -> dict:
    """Run train + val-threshold-selection + test-eval for one month."""
    logger.info("=== Spike risk backtest month: %s ===", target_month)

    # Targets
    target_result = compute_spike_risk_targets(raw_df, thresholds=thresholds)

    # Features
    feature_result = build_spike_risk_features(raw_df, mode=mode)
    feature_cols = [c for c in feature_result.feature_columns if c in feature_result.df.columns]

    # Merge target columns into feature df
    work = feature_result.df.copy()
    for col in ["spike_label", "extreme_spike_label", "relative_spike_label"]:
        if col in target_result.df.columns and col not in work.columns:
            work[col] = target_result.df[col].values

    # Split
    train_mask, val_mask, test_mask = split_train_test(work, target_month, val_days=val_days)

    n_train, n_val, n_test = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
    logger.info("  train=%d  val=%d  test=%d", n_train, n_val, n_test)

    if n_train < 10 or n_test < 10:
        return {"month": target_month, "status": "INSUFFICIENT_DATA",
                "reason": f"train={n_train}, test={n_test}"}

    # ── Train ─────────────────────────────────────────────────────────────
    X_train = work.loc[train_mask, feature_cols].fillna(0)
    y_spike = work.loc[train_mask, "spike_label"].values
    y_extreme = work.loc[train_mask, "extreme_spike_label"].values
    y_relative = work.loc[train_mask, "relative_spike_label"].values

    model = SpikeRiskModel(SpikeRiskConfig())
    model.fit(X_train, y_spike, y_extreme, y_relative)

    # ── Predict on val + test ─────────────────────────────────────────────
    val_pred_df = model.predict(work.loc[val_mask, feature_cols].fillna(0)).df
    test_pred_df = model.predict(work.loc[test_mask, feature_cols].fillna(0)).df

    # Attach labels to test predictions
    for label_col in ["spike_label", "extreme_spike_label", "relative_spike_label"]:
        test_pred_df[label_col] = work.loc[test_mask, label_col].values

    # ── Threshold selection on validation set ─────────────────────────────
    val_thresholds = {}
    for direction, prob_col, label_col in RISK_TARGETS:
        if prob_col in val_pred_df.columns and label_col in work.columns:
            val_labels = work.loc[val_mask, label_col].values
            val_probs = val_pred_df[prob_col].values
            result = select_threshold_by_objective(val_labels, val_probs, objective=objective)
            val_thresholds[direction] = result
        else:
            val_thresholds[direction] = {"best_threshold": 0.5, "f1": 0.0,
                                         "precision": 0.0, "recall": 0.0}

    # ── Test-set evaluation ───────────────────────────────────────────────
    monthly_rows = []
    topk_rows = []
    calibration_rows = []
    threshold_rows = []
    all_sufficient = True

    for direction, prob_col, label_col in RISK_TARGETS:
        y_true = test_pred_df[label_col].values.astype(float)
        y_prob = test_pred_df[prob_col].values.astype(float)

        # Check spike event count
        valid = y_true[np.isfinite(y_true)]
        n_positive = int((valid >= 1).sum())

        if n_positive < MIN_SPIKE_EVENTS:
            logger.warning("  [%s] INSUFFICIENT_SPIKE_EVENTS: only %d positives (need %d)",
                           direction, n_positive, MIN_SPIKE_EVENTS)
            monthly_rows.append({
                "month": target_month, "direction": direction,
                "status": "INSUFFICIENT_SPIKE_EVENTS", "n_positive": n_positive,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                "roc_auc": np.nan, "threshold": np.nan,
            })
            all_sufficient = False
            continue

        thr = val_thresholds[direction]["best_threshold"]

        # Classification metrics
        cls = compute_classification_metrics(y_true, y_prob, threshold=thr)
        monthly_rows.append({
            "month": target_month, "direction": direction,
            "status": "ok", "n_positive": n_positive,
            "precision": cls["precision"], "recall": cls["recall"],
            "f1": cls["f1"], "roc_auc": cls["roc_auc"], "threshold": thr,
        })

        # Threshold selection record
        threshold_rows.append({
            "month": target_month, "direction": direction,
            "best_threshold": thr,
            "val_f1": val_thresholds[direction]["f1"],
            "val_precision": val_thresholds[direction]["precision"],
            "val_recall": val_thresholds[direction]["recall"],
        })

        # Top-k capture
        topk_df = compute_topk_metrics(y_true, y_prob)
        topk_df["month"] = target_month
        topk_df["direction"] = direction
        topk_rows.append(topk_df)

        # Calibration
        cal_df = compute_calibration_metrics(y_true, y_prob)
        cal_df["month"] = target_month
        cal_df["direction"] = direction
        calibration_rows.append(cal_df)

    # ── Feature importance ────────────────────────────────────────────────
    fi = model.get_feature_importance()

    # ── Save per-month predictions ────────────────────────────────────────
    pred_out = test_pred_df.copy()
    for col in ["business_day", "hour_business", "ds"]:
        if col in work.columns:
            pred_out[col] = work.loc[test_mask, col].values
    pred_csv = out_dir / f"predictions_{target_month.replace('-', '_')}.csv"
    pred_out.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    return {
        "month": target_month,
        "status": "ok" if all_sufficient else "PARTIAL",
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_features": len(feature_cols),
        "feature_audit_verdict": feature_result.audit.verdict,
        "monthly_rows": monthly_rows,
        "topk_rows": topk_rows,
        "calibration_rows": calibration_rows,
        "threshold_rows": threshold_rows,
        "feature_importance": fi,
    }


# ── Verdict logic ────────────────────────────────────────────────────────────

def determine_overall_verdict(monthly_results: list) -> dict:
    """Aggregate top-k lift and recall across months.

    SPIKE_CHAMPION    : mean top10 lift >= 2.5, mean recall@top20 >= 0.5,
                        >= 3 sufficient months
    SPIKE_ACCEPTABLE  : mean top10 lift >= 1.8
    SPIKE_LOW_VALUE   : top-k useful but recall unstable
    SPIKE_NO_GO       : otherwise
    """
    lifts_top10 = []
    recalls_top20 = []
    n_sufficient = 0

    for r in monthly_results:
        if r.get("status") == "INSUFFICIENT_DATA":
            continue

        month_has_ok = False
        for row in r.get("monthly_rows", []):
            if row.get("status") == "ok":
                month_has_ok = True

        for topk_df in r.get("topk_rows", []):
            row_10 = topk_df[topk_df["topk_pct"] == 10]
            if not row_10.empty:
                lifts_top10.append(float(row_10["lift_vs_random"].iloc[0]))
            row_20 = topk_df[topk_df["topk_pct"] == 20]
            if not row_20.empty:
                recalls_top20.append(float(row_20["recall_at_k"].iloc[0]))

        if month_has_ok:
            n_sufficient += 1

    mean_lift = float(np.mean(lifts_top10)) if lifts_top10 else 0.0
    mean_recall = float(np.mean(recalls_top20)) if recalls_top20 else 0.0

    if mean_lift >= 2.5 and mean_recall >= 0.5 and n_sufficient >= 3:
        verdict = "SPIKE_CHAMPION"
    elif mean_lift >= 1.8:
        verdict = "SPIKE_ACCEPTABLE"
    elif mean_lift >= 1.2:
        verdict = "SPIKE_LOW_VALUE"
    else:
        verdict = "SPIKE_NO_GO"

    return {
        "verdict": verdict,
        "mean_top10_lift": mean_lift,
        "mean_recall_top20": mean_recall,
        "n_sufficient_months": n_sufficient,
        "n_months_total": len(monthly_results),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_months = [m.strip() for m in args.target_months.split(",")]
    thresholds = SpikeRiskThresholds(
        spike=args.threshold_spike,
        extreme_spike=args.threshold_extreme_spike,
        relative_spike=args.threshold_relative_spike,
    )

    raw_df = load_data(args.data_path)
    raw_df = rename_chinese_columns(raw_df)

    monthly_results = []
    for month in target_months:
        result = run_single_month(
            raw_df, month, thresholds, args.mode, args.val_days, args.objective, out_dir,
        )
        monthly_results.append(result)

    # ── Aggregate outputs ─────────────────────────────────────────────────
    all_monthly_rows = []
    all_topk_rows = []
    all_cal_rows = []
    all_threshold_rows = []
    all_fis = []

    for r in monthly_results:
        if r.get("status") in ("INSUFFICIENT_DATA",):
            continue
        all_monthly_rows.extend(r.get("monthly_rows", []))
        for df in r.get("topk_rows", []):
            all_topk_rows.append(df)
        for df in r.get("calibration_rows", []):
            all_cal_rows.append(df)
        all_threshold_rows.extend(r.get("threshold_rows", []))
        fi = r.get("feature_importance")
        if fi is not None and not fi.empty:
            all_fis.append(fi)

    # monthly_metrics.csv
    pd.DataFrame(all_monthly_rows).to_csv(out_dir / "monthly_metrics.csv", index=False)

    # target_metrics.csv
    if all_monthly_rows:
        target_df = pd.DataFrame(all_monthly_rows)
        ok_rows = target_df[target_df.get("status", pd.Series(dtype=str)) == "ok"]
        if not ok_rows.empty:
            target_summary = ok_rows.groupby("direction").agg(
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_f1=("f1", "mean"),
                mean_roc_auc=("roc_auc", "mean"),
                n_months=("month", "nunique"),
            ).reset_index()
        else:
            target_summary = pd.DataFrame()
        target_summary.to_csv(out_dir / "target_metrics.csv", index=False)

    # topk_metrics.csv
    if all_topk_rows:
        pd.concat(all_topk_rows, ignore_index=True).to_csv(
            out_dir / "topk_metrics.csv", index=False)

    # calibration_metrics.csv
    if all_cal_rows:
        pd.concat(all_cal_rows, ignore_index=True).to_csv(
            out_dir / "calibration_metrics.csv", index=False)

    # thresholds.csv
    pd.DataFrame(all_threshold_rows).to_csv(out_dir / "thresholds.csv", index=False)

    # feature_importance_summary.csv
    if all_fis:
        combined_fi = all_fis[0].copy()
        for fi in all_fis[1:]:
            for col in fi.columns:
                if col != "feature" and col in combined_fi.columns:
                    combined_fi[col] = (combined_fi[col] + fi[col]) / 2
        combined_fi.to_csv(out_dir / "feature_importance_summary.csv", index=False)

    # ── Overall verdict ───────────────────────────────────────────────────
    verdict_info = determine_overall_verdict(monthly_results)

    verdict_json = {
        "overall_verdict": verdict_info["verdict"],
        "mean_top10_lift": verdict_info["mean_top10_lift"],
        "mean_recall_top20": verdict_info["mean_recall_top20"],
        "n_sufficient_months": verdict_info["n_sufficient_months"],
        "n_months": len(target_months),
        "created_at": datetime.now().isoformat(),
    }
    with open(out_dir / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_json, f, ensure_ascii=False, indent=2)

    # ── Markdown report ───────────────────────────────────────────────────
    ok_months = [r for r in monthly_results if r.get("status") not in ("INSUFFICIENT_DATA",)]
    insufficient = [r for r in monthly_results if r.get("status") == "INSUFFICIENT_DATA"]

    report_lines = [
        "# Spike Risk Backtest Report",
        "",
        f"**Overall Verdict: {verdict_info['verdict']}**",
        "",
        f"- Mean top-10% lift: {verdict_info['mean_top10_lift']:.2f}",
        f"- Mean recall@top-20%: {verdict_info['mean_recall_top20']:.2f}",
        f"- Sufficient months: {verdict_info['n_sufficient_months']}",
        f"- Months tested: {len(target_months)}",
        f"- Successful months: {len(ok_months)}",
        f"- Insufficient data months: {len(insufficient)}",
        "",
        "## Monthly Metrics",
        "",
    ]
    if all_monthly_rows:
        report_lines.append(pd.DataFrame(all_monthly_rows).to_csv(index=False))
    else:
        report_lines.append("No successful months.")

    if insufficient:
        report_lines.append("")
        report_lines.append("## Insufficient Data Months")
        report_lines.append("")
        for r in insufficient:
            report_lines.append(f"- {r['month']}: {r.get('reason', 'unknown')}")

    report_lines.append("")
    report_lines.append("## Verdict Criteria")
    report_lines.append("")
    report_lines.append("| Verdict | Condition |")
    report_lines.append("|---------|-----------|")
    report_lines.append("| SPIKE_CHAMPION | mean top10 lift >= 2.5, mean recall@top20 >= 0.5, >= 3 sufficient months |")
    report_lines.append("| SPIKE_ACCEPTABLE | mean top10 lift >= 1.8 |")
    report_lines.append("| SPIKE_LOW_VALUE | top-k useful but recall unstable |")
    report_lines.append("| SPIKE_NO_GO | otherwise |")
    report_lines.append("")

    with open(out_dir / "spike_risk_backtest_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("Backtest complete. Verdict: %s", verdict_info["verdict"])
    logger.info("Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
