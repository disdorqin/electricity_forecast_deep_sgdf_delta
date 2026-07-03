#!/usr/bin/env python
"""Evaluation entry point for TrendKnightRT.

Loads model predictions and ground-truth data, computes comprehensive
metrics (overall, per-period, per-hour, per-bucket), and generates
a go/no-go verdict report.

Usage:
    python scripts/evaluate_realtime_deep_model.py \
        --predictions reports/local/deep_final/predictions_2026_02.csv \
        --ground-truth ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --out reports/local/deep_final/eval_2026_02
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

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.metrics import (  # noqa: E402
    compute_full_metrics,
    compute_monthly_metrics,
    compute_period_mask,
    smape_floor50,
)
from models.deep_sgdf_delta.business_time import (  # noqa: E402
    add_business_time_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_realtime_deep_model")


# -- Verdict thresholds -------------------------------------------------------

PASS_THRESHOLD = 15.0
STRONG_THRESHOLD = 17.0
STRONG_916_THRESHOLD = 22.0
ACCEPTABLE_THRESHOLD = 20.0


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TrendKnightRT predictions and generate go/no-go report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions CSV (output of predict_realtime_deep_model.py)",
    )
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to ground truth CSV with ds, rt_price, forecast_price columns",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output directory for evaluation reports",
    )
    parser.add_argument(
        "--spike-threshold", type=float, default=500.0,
        help="Price threshold for spike classification (default: 500)",
    )
    return parser.parse_args()


# -- Data loading -------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV, trying utf-8-sig first then gbk."""
    p = Path(path)
    if not p.exists():
        alt = PROJECT_ROOT / path
        if alt.exists():
            p = alt
        else:
            alt2 = PROJECT_ROOT.parent / path
            if alt2.exists():
                p = alt2
            else:
                raise FileNotFoundError(f"File not found: {path}")

    logger.info("Loading %s", p)
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except UnicodeDecodeError:
        logger.info("utf-8-sig failed, retrying with gbk encoding")
        df = pd.read_csv(p, encoding="gbk")

    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


def prepare_ground_truth(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare ground truth DataFrame with standard column names.

    Renames rt_price -> rt_actual, forecast_price -> da_anchor,
    adds business_time columns.
    """
    df = gt_df.copy()

    # Rename columns (Chinese + English aliases)
    rename_map = {}
    _CN_MAP = {"时刻": "ds", "日前电价": "da_anchor", "实时电价": "rt_actual"}
    for cn, en in _CN_MAP.items():
        if cn in df.columns:
            rename_map[cn] = en
    if "rt_price" in df.columns and "rt_actual" not in rename_map.values():
        rename_map["rt_price"] = "rt_actual"
    if "forecast_price" in df.columns and "da_anchor" not in rename_map.values():
        rename_map["forecast_price"] = "da_anchor"
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("Ground truth renamed %d columns", len(rename_map))

    # Ensure ds is datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])

    # Add business time columns
    df = add_business_time_columns(df, timestamp_col="ds")

    return df


def prepare_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare predictions DataFrame for evaluation."""
    df = pred_df.copy()

    # Ensure business_day is datetime
    df["business_day"] = pd.to_datetime(df["business_day"])

    # Ensure hour_business is int
    df["hour_business"] = df["hour_business"].astype(int)

    return df


# -- Metrics computation ------------------------------------------------------

def compute_period_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-period (1_8, 9_16, 17_24) sMAPE_floor50."""
    hours = merged_df["hour_business"].to_numpy(dtype=int)
    yt = merged_df["rt_actual"].to_numpy(dtype=float)
    yp = merged_df["rt_pred"].to_numpy(dtype=float)

    rows = []
    for period in ("1_8", "9_16", "17_24"):
        mask = compute_period_mask(hours, period)
        n = int(mask.sum())
        if n > 0:
            smape = smape_floor50(yt[mask], yp[mask])
        else:
            smape = float("nan")
        rows.append({
            "period": period,
            "sMAPE_floor50": smape,
            "count": n,
        })

    return pd.DataFrame(rows)


def compute_hourly_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-hour sMAPE_floor50."""
    rows = []
    for hour in range(1, 25):
        mask = merged_df["hour_business"].to_numpy(dtype=int) == hour
        n = int(mask.sum())
        if n > 0:
            yt = merged_df.loc[mask, "rt_actual"].to_numpy(dtype=float)
            yp = merged_df.loc[mask, "rt_pred"].to_numpy(dtype=float)
            smape = smape_floor50(yt, yp)
        else:
            smape = float("nan")
        rows.append({
            "hour": hour,
            "sMAPE_floor50": smape,
            "count": n,
        })

    return pd.DataFrame(rows)


def compute_bucket_metrics(
    merged_df: pd.DataFrame,
    spike_threshold: float = 500.0,
) -> pd.DataFrame:
    """Compute per-bucket (normal, negative, spike) sMAPE_floor50."""
    yt = merged_df["rt_actual"].to_numpy(dtype=float)
    yp = merged_df["rt_pred"].to_numpy(dtype=float)

    spike_mask = np.abs(yt) > spike_threshold
    neg_mask = yt < 0.0
    normal_mask = ~spike_mask & ~neg_mask

    rows = []
    for bucket_name, mask in [
        ("normal", normal_mask),
        ("negative", neg_mask),
        ("spike", spike_mask),
    ]:
        n = int(mask.sum())
        if n > 0:
            smape = smape_floor50(yt[mask], yp[mask])
        else:
            smape = float("nan")
        rows.append({
            "bucket": bucket_name,
            "sMAPE_floor50": smape,
            "count": n,
        })

    return pd.DataFrame(rows)


# -- Verdict ------------------------------------------------------------------

def determine_verdict(metrics: dict) -> tuple[str, str]:
    """Determine go/no-go verdict based on metrics.

    Rules (evaluated in order):
      - STRONG:     overall < 17 AND 9_16 < 22
      - PASS:       overall < 15
      - ACCEPTABLE: overall < 20
      - NO-GO:      overall >= 20

    Returns:
        (verdict, detail) tuple
    """
    overall = metrics.get("overall_sMAPE_floor50", float("nan"))
    period_916 = metrics.get("9_16_sMAPE_floor50", float("nan"))

    if np.isnan(overall):
        return "NO-GO", "Overall sMAPE_floor50 is NaN (no valid data)"

    # STRONG: best tier
    if overall < STRONG_THRESHOLD and not np.isnan(period_916) and period_916 < STRONG_916_THRESHOLD:
        return (
            "STRONG",
            f"Overall sMAPE_floor50={overall:.2f} < {STRONG_THRESHOLD} "
            f"AND 9_16 sMAPE={period_916:.2f} < {STRONG_916_THRESHOLD}"
        )

    # PASS
    if overall < PASS_THRESHOLD:
        return (
            "PASS",
            f"Overall sMAPE_floor50={overall:.2f} < {PASS_THRESHOLD}"
        )

    # ACCEPTABLE
    if overall < ACCEPTABLE_THRESHOLD:
        return (
            "ACCEPTABLE",
            f"Overall sMAPE_floor50={overall:.2f} < {ACCEPTABLE_THRESHOLD}"
        )

    # NO-GO
    return (
        "NO-GO",
        f"Overall sMAPE_floor50={overall:.2f} >= {ACCEPTABLE_THRESHOLD}"
    )


# -- Report generation --------------------------------------------------------

def write_go_nogo_report(
    output_dir: Path,
    metrics: dict,
    verdict: str,
    verdict_detail: str,
    monthly_df: pd.DataFrame,
    period_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
) -> None:
    """Write the go_nogo.md verdict report."""
    lines = [
        "# TrendKnightRT Go/No-Go Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Verdict:** {verdict}",
        "",
        f"{verdict_detail}",
        "",
        "## Verdict Thresholds",
        "",
        f"| Level | Condition |",
        f"|-------|-----------|",
        f"| STRONG | overall < {STRONG_THRESHOLD} AND 9_16 < {STRONG_916_THRESHOLD} |",
        f"| PASS | overall < {PASS_THRESHOLD} |",
        f"| ACCEPTABLE | overall < {ACCEPTABLE_THRESHOLD} |",
        f"| NO-GO | overall >= {ACCEPTABLE_THRESHOLD} |",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall sMAPE_floor50 | {metrics.get('overall_sMAPE_floor50', float('nan')):.4f} |",
        f"| 1_8 sMAPE_floor50 | {metrics.get('1_8_sMAPE_floor50', float('nan')):.4f} |",
        f"| 9_16 sMAPE_floor50 | {metrics.get('9_16_sMAPE_floor50', float('nan')):.4f} |",
        f"| 17_24 sMAPE_floor50 | {metrics.get('17_24_sMAPE_floor50', float('nan')):.4f} |",
        f"| Normal sMAPE_floor50 | {metrics.get('normal_sMAPE_floor50', float('nan')):.4f} |",
        f"| Negative sMAPE_floor50 | {metrics.get('negative_sMAPE_floor50', float('nan')):.4f} |",
        f"| Spike sMAPE_floor50 | {metrics.get('spike_sMAPE_floor50', float('nan')):.4f} |",
        f"| Delta MAE | {metrics.get('delta_mae', float('nan')):.4f} |",
        f"| Gap to 15 | {metrics.get('gap_to_15', float('nan')):.4f} |",
        f"| Gap to 20 | {metrics.get('gap_to_20', float('nan')):.4f} |",
        f"| Rows total | {metrics.get('rows_total', 0)} |",
        f"| Rows missing | {metrics.get('rows_missing', 0)} |",
        "",
    ]

    # Monthly breakdown
    if not monthly_df.empty:
        lines.append("## Monthly Breakdown")
        lines.append("")
        lines.append("| Month | sMAPE_floor50 | Count |")
        lines.append("|-------|---------------|-------|")
        for _, row in monthly_df.iterrows():
            lines.append(f"| {row['month']} | {row['sMAPE_floor50']:.4f} | {row['count']} |")
        lines.append("")

    # Period breakdown
    if not period_df.empty:
        lines.append("## Period Breakdown")
        lines.append("")
        lines.append("| Period | sMAPE_floor50 | Count |")
        lines.append("|--------|---------------|-------|")
        for _, row in period_df.iterrows():
            lines.append(f"| {row['period']} | {row['sMAPE_floor50']:.4f} | {row['count']} |")
        lines.append("")

    # Bucket breakdown
    if not bucket_df.empty:
        lines.append("## Bucket Breakdown")
        lines.append("")
        lines.append("| Bucket | sMAPE_floor50 | Count |")
        lines.append("|--------|---------------|-------|")
        for _, row in bucket_df.iterrows():
            lines.append(f"| {row['bucket']} | {row['sMAPE_floor50']:.4f} | {row['count']} |")
        lines.append("")

    report_path = output_dir / "go_nogo.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Go/No-Go report written to %s", report_path)


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Load predictions
    pred_df = load_csv(args.predictions)
    pred_df = prepare_predictions(pred_df)

    # Load ground truth
    gt_df = load_csv(args.ground_truth)
    gt_df = prepare_ground_truth(gt_df)

    # Merge predictions with ground truth on (business_day, hour_business)
    # Ground truth has rt_actual; predictions have rt_pred
    gt_cols = ["business_day", "hour_business", "rt_actual", "da_anchor"]
    available_gt_cols = [c for c in gt_cols if c in gt_df.columns]

    # If delta_target is needed, compute it
    if "rt_actual" in gt_df.columns and "da_anchor" in gt_df.columns:
        gt_df["delta_target"] = gt_df["rt_actual"] - gt_df["da_anchor"]
        available_gt_cols.append("delta_target")

    pred_cols = [
        "business_day", "hour_business", "rt_pred", "delta_pred",
        "da_anchor", "sgdfnet_pred", "residual_to_sgdfnet",
        "confidence", "period", "model_name",
    ]
    available_pred_cols = [c for c in pred_cols if c in pred_df.columns]

    merged = pred_df[available_pred_cols].merge(
        gt_df[available_gt_cols],
        on=["business_day", "hour_business"],
        how="inner",
        suffixes=("", "_gt"),
    )

    # Use ground truth da_anchor if available (prefer gt)
    if "da_anchor_gt" in merged.columns:
        merged["da_anchor"] = merged["da_anchor_gt"]
        merged = merged.drop(columns=["da_anchor_gt"])

    logger.info(
        "Merged: %d rows (%d prediction rows, %d ground truth rows)",
        len(merged), len(pred_df), len(gt_df),
    )

    if merged.empty:
        logger.error("No matching rows after merge. Check business_day/hour_business alignment.")
        sys.exit(1)

    # Add hour column (alias for compute_full_metrics)
    merged["hour"] = merged["hour_business"].astype(int)

    # -- Compute metrics ------------------------------------------------------

    # Full metrics (overall, period, bucket)
    full_metrics = compute_full_metrics(merged, spike_threshold=args.spike_threshold)

    # Gap metrics
    overall = full_metrics.get("overall_sMAPE_floor50", float("nan"))
    full_metrics["gap_to_15"] = overall - PASS_THRESHOLD
    full_metrics["gap_to_20"] = overall - ACCEPTABLE_THRESHOLD

    # Monthly metrics
    monthly_df = compute_monthly_metrics(merged)

    # Period metrics
    period_df = compute_period_metrics(merged)

    # Hourly metrics
    hourly_df = compute_hourly_metrics(merged)

    # Bucket metrics
    bucket_df = compute_bucket_metrics(merged, spike_threshold=args.spike_threshold)

    # Verdict
    verdict, verdict_detail = determine_verdict(full_metrics)
    full_metrics["verdict"] = verdict
    full_metrics["verdict_detail"] = verdict_detail

    # -- Write outputs --------------------------------------------------------

    # 1. metrics_summary.json
    # Convert NaN to null for JSON
    metrics_json = {}
    for k, v in full_metrics.items():
        if isinstance(v, float) and np.isnan(v):
            metrics_json[k] = None
        else:
            metrics_json[k] = v

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    logger.info("Metrics summary written")

    # 2. monthly_metrics.csv
    if not monthly_df.empty:
        monthly_df.to_csv(output_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info("Monthly metrics written (%d months)", len(monthly_df))

    # 3. period_metrics.csv
    if not period_df.empty:
        period_df.to_csv(output_dir / "period_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info("Period metrics written")

    # 4. hourly_metrics.csv
    if not hourly_df.empty:
        hourly_df.to_csv(output_dir / "hourly_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info("Hourly metrics written (%d hours)", len(hourly_df))

    # 5. bucket_metrics.csv
    if not bucket_df.empty:
        bucket_df.to_csv(output_dir / "bucket_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info("Bucket metrics written")

    # 6. go_nogo.md
    write_go_nogo_report(output_dir, full_metrics, verdict, verdict_detail, monthly_df, period_df, bucket_df)

    # -- Summary to stdout ----------------------------------------------------
    print()
    print("=" * 60)
    print("  TrendKnightRT Evaluation Complete")
    print("=" * 60)
    print(f"  Verdict:          {verdict}")
    print(f"  {verdict_detail}")
    print(f"  Overall sMAPE:    {overall:.4f}")
    print(f"  Gap to 15:        {full_metrics['gap_to_15']:.4f}")
    print(f"  Gap to 20:        {full_metrics['gap_to_20']:.4f}")
    print(f"  1_8 sMAPE:        {full_metrics.get('1_8_sMAPE_floor50', float('nan')):.4f}")
    print(f"  9_16 sMAPE:       {full_metrics.get('9_16_sMAPE_floor50', float('nan')):.4f}")
    print(f"  17_24 sMAPE:      {full_metrics.get('17_24_sMAPE_floor50', float('nan')):.4f}")
    print(f"  Normal sMAPE:     {full_metrics.get('normal_sMAPE_floor50', float('nan')):.4f}")
    print(f"  Negative sMAPE:   {full_metrics.get('negative_sMAPE_floor50', float('nan')):.4f}")
    print(f"  Spike sMAPE:      {full_metrics.get('spike_sMAPE_floor50', float('nan')):.4f}")
    print(f"  Rows evaluated:   {full_metrics.get('rows_total', 0)}")
    print(f"  Rows missing:     {full_metrics.get('rows_missing', 0)}")
    print(f"  Output:           {output_dir}")
    print("=" * 60)
    print()
    print("Output files:")
    for fname in [
        "metrics_summary.json",
        "monthly_metrics.csv",
        "period_metrics.csv",
        "hourly_metrics.csv",
        "bucket_metrics.csv",
        "go_nogo.md",
    ]:
        fpath = output_dir / fname
        status = "OK" if fpath.exists() else "MISSING"
        print(f"  [{status}] {fpath}")


if __name__ == "__main__":
    main()
