#!/usr/bin/env python
"""Evaluate Intraday Residual Tracker — Phase 10 stability validation.

Supports multi-month backtest with optional policy gating.

For each business_day D in the test period:
  For each cutoff_hour in [8, 9, 10, 11, 12, 13, 14, 15]:
    - Use hours <= cutoff_hour as observed
    - Predict hours cutoff_hour+1 to 16 (9_16 segment)
    - Compare with SGDFNet baseline
    - Optionally apply policy gating

Outputs to reports/local/phase10/intraday_tracker_stability/ (or --out-dir):
  monthly_metrics.csv, cutoff_metrics.csv, bucket_metrics.csv,
  policy_metrics.csv, predictions.csv, stability_report.md

Usage:
    python scripts/evaluate_intraday_residual_tracker.py \\
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx \\
        --sgdfnet-predictions ../electricity_forecast_model2.0_exp/outputs/.../predictions.csv \\
        --months 2026-01,2026-02,2026-03 \\
        --cutoff-hours 10,11,12,13,14,15 \\
        --policy-enabled \\
        --out-dir reports/local/phase10/intraday_tracker_stability
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


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 50.0) -> float:
    yt = np.clip(np.abs(y_true), floor, None)
    yp = np.clip(np.abs(y_pred), floor, None)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


def load_raw_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot read {p}")


def month_range(month_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert '2026-02' to (start, end) timestamps."""
    start = pd.Timestamp(month_str + "-01")
    end = start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    return start, end


def main():
    parser = argparse.ArgumentParser(description="Evaluate Intraday Residual Tracker — Phase 10")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--sgdfnet-predictions", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date (overrides --months if given)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (overrides --months if given)")
    parser.add_argument("--months", type=str, default=None,
                        help="Comma-separated months: 2026-01,2026-02,2026-03")
    parser.add_argument("--out-dir", type=str,
                        default="reports/local/phase10/intraday_tracker_stability")
    parser.add_argument("--cutoff-hours", type=str, default="10,11,12,13,14,15")
    parser.add_argument("--min-observed-hours", type=int, default=2)
    parser.add_argument("--max-abs-correction", type=float, default=80.0)
    parser.add_argument("--policy-enabled", action="store_true", default=False,
                        help="Apply Phase 10 policy gating")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff_hours = [int(h) for h in args.cutoff_hours.split(",")]

    # Determine date range
    if args.months:
        months_list = [m.strip() for m in args.months.split(",")]
        ranges = [month_range(m) for m in months_list]
        global_start = min(r[0] for r in ranges)
        global_end = max(r[1] for r in ranges)
        logger.info("Months mode: %s → %s to %s", months_list, global_start.date(), global_end.date())
    elif args.start_date and args.end_date:
        global_start = pd.Timestamp(args.start_date)
        global_end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
        months_list = None
        logger.info("Date mode: %s to %s", global_start.date(), global_end.date())
    else:
        global_start = pd.Timestamp("2026-02-01")
        global_end = pd.Timestamp("2026-02-28") + pd.Timedelta(days=1)
        months_list = None

    logger.info("=" * 60)
    logger.info("Intraday Residual Tracker Backtest — Phase 10")
    logger.info("=" * 60)
    logger.info("  Period: %s to %s", global_start.date(), global_end.date())
    logger.info("  Cutoff hours: %s", cutoff_hours)
    logger.info("  Policy enabled: %s", args.policy_enabled)

    # ── Load data ────────────────────────────────────────────────────
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    from models.deep_sgdf_delta.intraday_residual_tracker import (
        IntradayTrackerConfig,
        apply_intraday_correction,
        compute_intraday_residual_state,
    )

    if args.policy_enabled:
        from models.deep_sgdf_delta.intraday_tracker_policy import (
            PolicyConfig,
            evaluate_policy,
        )
        from models.deep_sgdf_delta.prediction_modes import PredictionMode

    raw_df = load_raw_data(args.data_path)
    logger.info("Raw data: %d rows", len(raw_df))

    # Detect timestamp column
    ts_col = None
    for c in ["时刻", "timestamp", "time", "ds"]:
        if c in raw_df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("No timestamp column found")

    df = add_business_time_columns(raw_df, timestamp_col=ts_col)

    # Detect price columns
    da_col = rt_col = None
    for c in ["日前电价", "da_price", "dayahead"]:
        if c in df.columns:
            da_col = c
            break
    for c in ["实时电价", "rt_price", "realtime"]:
        if c in df.columns:
            rt_col = c
            break
    if da_col is None or rt_col is None:
        raise ValueError("Cannot find price columns")

    df = df.rename(columns={da_col: "da_price", rt_col: "rt_price"})
    df["da_price"] = pd.to_numeric(df["da_price"], errors="coerce")
    df["rt_price"] = pd.to_numeric(df["rt_price"], errors="coerce")

    # Load SGDFNet predictions
    sgdf = pd.read_csv(args.sgdfnet_predictions, encoding="utf-8-sig")
    sgdf["business_day"] = pd.to_datetime(sgdf["business_day"])
    # Detect prediction column
    pred_col = None
    for c in ("rt_hat", "teacher_pred", "y_pred"):
        if c in sgdf.columns:
            pred_col = c
            break
    if pred_col is None:
        raise ValueError("Cannot find prediction column in SGDFNet predictions")
    sgdf = sgdf.rename(columns={pred_col: "sgdfnet_pred"})

    # Filter to 9_16 segment
    df_916 = df[df["period"] == "9_16"].copy()
    logger.info("9_16 rows: %d", len(df_916))

    # Merge SGDFNet predictions
    sgdf_merge = sgdf[["business_day", "hour", "sgdfnet_pred", "da_anchor"]].copy()
    sgdf_merge = sgdf_merge.rename(columns={"hour": "hour_business"})
    df_916 = df_916.merge(
        sgdf_merge, on=["business_day", "hour_business"], how="left", suffixes=("", "_sgdf")
    )
    if "da_anchor_sgdf" in df_916.columns:
        df_916["da_anchor"] = df_916["da_anchor_sgdf"].fillna(df_916["da_price"])
    else:
        df_916["da_anchor"] = df_916["da_price"]

    df_916 = df_916.dropna(subset=["sgdfnet_pred", "rt_price"]).copy()
    df_916 = df_916.rename(columns={"rt_price": "rt_actual"})
    logger.info("After merge: %d rows with SGDFNet predictions", len(df_916))

    # Date filter
    df_916 = df_916[(df_916["business_day"] >= global_start) & (df_916["business_day"] < global_end)]
    logger.info("Test period: %d rows", len(df_916))

    # ── Backtest ─────────────────────────────────────────────────────
    config = IntradayTrackerConfig(
        max_abs_correction=args.max_abs_correction,
        min_observed_hours=args.min_observed_hours,
    )
    policy_config = PolicyConfig() if args.policy_enabled else None

    all_predictions = []
    business_days = sorted(df_916["business_day"].unique())
    logger.info("Testing %d business days", len(business_days))

    for bd in business_days:
        day_data = df_916[df_916["business_day"] == bd].copy()
        if len(day_data) == 0:
            continue

        for cutoff in cutoff_hours:
            observed = day_data[day_data["hour_business"] <= cutoff]
            future = day_data[day_data["hour_business"] > cutoff]

            if len(observed) < config.min_observed_hours or len(future) == 0:
                continue

            state = compute_intraday_residual_state(observed, bd, cutoff)
            corrected = apply_intraday_correction(future, state, config)

            # Policy evaluation
            if args.policy_enabled:
                has_neg_risk = bool((future["da_anchor"].fillna(0) < 0).any())
                policy_result = evaluate_policy(
                    state, PredictionMode.INTRADAY, policy_config, has_neg_risk
                )
                policy_decision = policy_result.policy_decision.value
                fusion_weight = policy_result.fusion_weight
                shadow_flag = policy_result.shadow_only_flag
                policy_reason = policy_result.reason
            else:
                policy_decision = "N/A"
                fusion_weight = 1.0
                shadow_flag = False
                policy_reason = ""

            # Store results with Phase 10 fields
            for _, row in corrected.iterrows():
                all_predictions.append({
                    "business_day": bd,
                    "cutoff_hour": cutoff,
                    "target_hour": int(row["hour_business"]),
                    "ds": row.get("ds", bd),
                    "rt_actual": row["rt_actual"],
                    "sgdfnet_pred": row["sgdfnet_pred"],
                    # Phase 10 pipeline fields
                    "intraday_base_correction": row.get("intraday_base_correction", 0.0),
                    "intraday_model_weight": row.get("intraday_model_weight", 0.0),
                    "intraday_pre_guardrail_correction": row.get("intraday_pre_guardrail_correction", 0.0),
                    "intraday_guardrail_weight": row.get("intraday_guardrail_weight", 1.0),
                    "intraday_final_correction": row.get("intraday_final_correction", 0.0),
                    "intraday_correction": row["intraday_correction"],  # backward compat
                    "intraday_corrected_pred": row["intraday_corrected_pred"],
                    # State fields
                    "confidence": state.confidence,
                    "n_observed": state.n_observed,
                    "mean_residual_today": state.mean_residual_today,
                    "median_residual_today": state.median_residual_today,
                    "ewm_residual_today": state.ewm_residual_today,
                    "last_residual": state.last_residual,
                    "residual_std_today": state.residual_std_today,
                    "bias_direction": state.bias_direction,
                    "observed_hours": str(state.observed_hours),
                    # Guardrail
                    "guardrail_reason": row.get("guardrail_reason", ""),
                    "da_anchor": row.get("da_anchor", 0),
                    # Policy fields
                    "policy_decision": policy_decision,
                    "fusion_weight": fusion_weight,
                    "shadow_only_flag": shadow_flag,
                    "policy_reason": policy_reason,
                })

    if not all_predictions:
        logger.error("No predictions generated!")
        sys.exit(1)

    pred_df = pd.DataFrame(all_predictions)
    logger.info("Total predictions: %d", len(pred_df))

    # ── Compute metrics ──────────────────────────────────────────────
    rt = pred_df["rt_actual"].values
    base = pred_df["sgdfnet_pred"].values
    corr = pred_df["intraday_corrected_pred"].values

    overall_base = smape_floor50(rt, base)
    overall_corr = smape_floor50(rt, corr)
    overall_improvement = overall_base - overall_corr
    logger.info("Overall: baseline=%.4f, corrected=%.4f, improvement=%.4f",
                overall_base, overall_corr, overall_improvement)

    # By month
    pred_df["month"] = pred_df["business_day"].dt.to_period("M").astype(str)
    monthly_rows = []
    for month, grp in pred_df.groupby("month"):
        rt_m = grp["rt_actual"].values
        base_m = grp["sgdfnet_pred"].values
        corr_m = grp["intraday_corrected_pred"].values
        b = smape_floor50(rt_m, base_m)
        c = smape_floor50(rt_m, corr_m)
        monthly_rows.append({
            "month": month,
            "count": len(grp),
            "baseline_smape": b,
            "corrected_smape": c,
            "improvement": b - c,
        })
    monthly_df = pd.DataFrame(monthly_rows)

    # By cutoff hour
    cutoff_rows = []
    for cutoff in cutoff_hours:
        mask = pred_df["cutoff_hour"] == cutoff
        if mask.sum() == 0:
            continue
        b = smape_floor50(rt[mask], base[mask])
        c = smape_floor50(rt[mask], corr[mask])
        cutoff_rows.append({
            "cutoff_hour": cutoff,
            "count": int(mask.sum()),
            "baseline_smape": b,
            "corrected_smape": c,
            "improvement": b - c,
        })
    cutoff_df = pd.DataFrame(cutoff_rows)

    # By bucket
    bucket_rows = []
    for bucket_name, bucket_fn in [
        ("normal", lambda x: (np.abs(x) <= 500) & (x >= 0)),
        ("spike", lambda x: np.abs(x) > 500),
        ("negative", lambda x: x < 0),
    ]:
        mask = bucket_fn(rt)
        if mask.sum() == 0:
            continue
        b = smape_floor50(rt[mask], base[mask])
        c = smape_floor50(rt[mask], corr[mask])
        bucket_rows.append({
            "bucket": bucket_name,
            "count": int(mask.sum()),
            "baseline_smape": b,
            "corrected_smape": c,
            "improvement": b - c,
        })
    bucket_df = pd.DataFrame(bucket_rows)

    # By policy decision
    policy_rows = []
    if args.policy_enabled:
        for decision, grp in pred_df.groupby("policy_decision"):
            rt_p = grp["rt_actual"].values
            base_p = grp["sgdfnet_pred"].values
            corr_p = grp["intraday_corrected_pred"].values
            b = smape_floor50(rt_p, base_p)
            c = smape_floor50(rt_p, corr_p)
            policy_rows.append({
                "policy_decision": decision,
                "count": len(grp),
                "baseline_smape": b,
                "corrected_smape": c,
                "improvement": b - c,
                "avg_fusion_weight": grp["fusion_weight"].mean(),
            })
    policy_df = pd.DataFrame(policy_rows)

    # ── Verdict ──────────────────────────────────────────────────────
    high_cutoff = cutoff_df[cutoff_df["cutoff_hour"] >= 10]
    high_cutoff_improvement = high_cutoff["improvement"].mean() if len(high_cutoff) > 0 else 0

    neg_improvement = None
    for row in bucket_rows:
        if row["bucket"] == "negative":
            neg_improvement = row["improvement"]
            break

    # Monthly stability check
    monthly_stable = True
    monthly_degraded_months = []
    for row in monthly_rows:
        if row["improvement"] < -1.0:
            monthly_stable = False
            monthly_degraded_months.append(row["month"])

    if high_cutoff_improvement >= 1.0 and (neg_improvement is None or neg_improvement >= -1.0) and monthly_stable:
        verdict = "GO"
    elif high_cutoff_improvement >= 0.3 and monthly_stable:
        verdict = "LOW-WEIGHT"
    else:
        verdict = "NO-GO"

    logger.info("Verdict: %s (high_cutoff_improvement=%.4f, monthly_stable=%s)",
                verdict, high_cutoff_improvement, monthly_stable)

    # ── Write outputs ────────────────────────────────────────────────
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(out_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    cutoff_df.to_csv(out_dir / "cutoff_metrics.csv", index=False, encoding="utf-8-sig")
    bucket_df.to_csv(out_dir / "bucket_metrics.csv", index=False, encoding="utf-8-sig")
    if args.policy_enabled and len(policy_df) > 0:
        policy_df.to_csv(out_dir / "policy_metrics.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "start_date": str(global_start.date()),
        "end_date": str((global_end - pd.Timedelta(days=1)).date()),
        "cutoff_hours": cutoff_hours,
        "policy_enabled": args.policy_enabled,
        "n_predictions": len(pred_df),
        "n_business_days": len(business_days),
        "overall_baseline_smape": overall_base,
        "overall_corrected_smape": overall_corr,
        "overall_improvement": overall_improvement,
        "high_cutoff_improvement": float(high_cutoff_improvement),
        "negative_bucket_improvement": neg_improvement,
        "monthly_stable": monthly_stable,
        "monthly_degraded_months": monthly_degraded_months,
        "verdict": verdict,
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Stability report
    lines = [
        "# Intraday Tracker Stability Report — Phase 10",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Period:** {global_start.date()} to {(global_end - pd.Timedelta(days=1)).date()}",
        f"**Policy Enabled:** {args.policy_enabled}",
        f"**Verdict:** {verdict}",
        "",
        "## Overall Metrics",
        "",
        f"- Baseline sMAPE: {overall_base:.4f}",
        f"- Corrected sMAPE: {overall_corr:.4f}",
        f"- Improvement: {overall_improvement:.4f}",
        f"- High Cutoff (>=10) Avg Improvement: {high_cutoff_improvement:.4f}",
        f"- Negative Bucket Improvement: {neg_improvement if neg_improvement is not None else 'N/A'}",
        f"- Monthly Stable: {monthly_stable}",
        "",
        "## Monthly Breakdown",
        "",
        "| Month | Count | Baseline | Corrected | Improvement |",
        "|-------|-------|----------|-----------|-------------|",
    ]
    for _, row in monthly_df.iterrows():
        lines.append(f"| {row['month']} | {int(row['count'])} | "
                     f"{row['baseline_smape']:.4f} | {row['corrected_smape']:.4f} | "
                     f"{row['improvement']:.4f} |")

    lines.extend([
        "",
        "## By Cutoff Hour",
        "",
        "| Cutoff | Count | Baseline | Corrected | Improvement |",
        "|--------|-------|----------|-----------|-------------|",
    ])
    for _, row in cutoff_df.iterrows():
        lines.append(f"| {int(row['cutoff_hour'])} | {int(row['count'])} | "
                     f"{row['baseline_smape']:.4f} | {row['corrected_smape']:.4f} | "
                     f"{row['improvement']:.4f} |")

    lines.extend([
        "",
        "## By Bucket",
        "",
        "| Bucket | Count | Baseline | Corrected | Improvement |",
        "|--------|-------|----------|-----------|-------------|",
    ])
    for _, row in bucket_df.iterrows():
        lines.append(f"| {row['bucket']} | {int(row['count'])} | "
                     f"{row['baseline_smape']:.4f} | {row['corrected_smape']:.4f} | "
                     f"{row['improvement']:.4f} |")

    if args.policy_enabled and len(policy_df) > 0:
        lines.extend([
            "",
            "## By Policy Decision",
            "",
            "| Decision | Count | Baseline | Corrected | Improvement | Avg Fusion Weight |",
            "|----------|-------|----------|-----------|-------------|-------------------|",
        ])
        for _, row in policy_df.iterrows():
            lines.append(f"| {row['policy_decision']} | {int(row['count'])} | "
                         f"{row['baseline_smape']:.4f} | {row['corrected_smape']:.4f} | "
                         f"{row['improvement']:.4f} | {row['avg_fusion_weight']:.4f} |")

    lines.extend([
        "",
        "## Stability Assessment",
        "",
        f"- Monthly stable: {monthly_stable}",
    ])
    if monthly_degraded_months:
        lines.append(f"- Degraded months (improvement < -1.0): {monthly_degraded_months}")
    lines.extend([
        f"- Negative bucket controlled: {neg_improvement is not None and neg_improvement >= -1.0}",
        "",
        "## Verdict Criteria",
        "",
        "- GO: cutoff>=10 improvement >= 1.0 AND negative bucket >= -1.0 AND monthly stable",
        "- LOW-WEIGHT: improvement 0.3~1.0 AND monthly stable",
        "- NO-GO: improvement < 0.3 OR monthly unstable",
    ])

    (out_dir / "stability_report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("All outputs written to %s", out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
