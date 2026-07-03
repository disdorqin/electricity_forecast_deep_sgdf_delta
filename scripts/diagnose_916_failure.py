#!/usr/bin/env python
"""9_16 Segment Diagnosis — Phase 5 Task F.

Analyzes why 9_16 segment (hours 9-16, solar-volatile period) has high error.

Breaks down by:
  - Hour (9, 10, 11, 12, 13, 14, 15, 16)
  - Price bucket (normal, spike, negative)
  - Features (光伏预测, 新能源预测, 竞价空间, 净负荷, 日前价格)

Output:
  docs/DIAGNOSE_916_FAILURE.md

Usage:
    python scripts/diagnose_916_failure.py \\
        --start-date 2026-01-01 --end-date 2026-03-31 \\
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx \\
        --source-repo-root ../electricity_forecast_model2.0_exp
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw electricity data with encoding fallback."""
    p = Path(path)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot read {p} with any encoding")


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 50.0) -> float:
    """Compute sMAPE with floor-50 capping."""
    yt = np.clip(np.abs(y_true), floor, None)
    yp = np.clip(np.abs(y_pred), floor, None)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


def main():
    parser = argparse.ArgumentParser(description="9_16 Segment Diagnosis")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--source-repo-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="reports/local/phase5/diagnose_916")
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("9_16 Segment Diagnosis — Phase 5 Task F")
    logger.info("=" * 60)
    logger.info("  Period: %s to %s", start_date.date(), end_date.date())

    # Load data
    raw_df = load_raw_data(args.data_path)
    logger.info("Raw data: %d rows", len(raw_df))

    # Load SGDFNet predictions
    try:
        from models.deep_sgdf_delta.teacher_adapters import sgdfnet_teacher
        sgdf_df = sgdfnet_teacher.load_predictions(
            source_repo_root=args.source_repo_root,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except Exception as exc:
        logger.warning("Failed to load SGDFNet predictions: %s", exc)
        sgdf_df = None

    # Build analysis frame
    df = raw_df.copy()

    # Detect timestamp column
    ts_col = None
    for c in ["时刻", "timestamp", "time", "ds"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("No timestamp column found")

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df[(df[ts_col] >= start_date) & (df[ts_col] < end_date)]

    # Detect price columns
    da_col = None
    rt_col = None
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

    df["ds"] = df[ts_col]
    df["da_price"] = df[da_col].astype(float)
    df["rt_price"] = df[rt_col].astype(float)

    # Business day alignment
    df["business_day"] = df["ds"].dt.normalize() - pd.Timedelta(days=1)
    df["hour"] = df["ds"].dt.hour
    df.loc[df["hour"] == 0, "hour"] = 24
    df.loc[df["hour"] == 0, "business_day"] = df.loc[df["hour"] == 0, "ds"].dt.normalize()

    # Period
    h = df["hour"].astype(int)
    df["period"] = pd.cut(h, bins=[0, 8, 16, 24], labels=["1_8", "9_16", "17_24"], include_lowest=True).astype(str)

    # Bucket
    df["bucket"] = "normal"
    df.loc[df["rt_price"].abs() > 500, "bucket"] = "spike"
    df.loc[df["rt_price"] < 0, "bucket"] = "negative"

    # Filter to 9_16 only
    df_916 = df[df["period"] == "9_16"].copy()
    logger.info("9_16 segment: %d rows", len(df_916))

    # ── Analysis 1: By hour ──────────────────────────────────────────
    logger.info("\n── Analysis 1: By hour (9-16) ──")
    hourly_stats = []
    for hour in range(9, 17):
        subset = df_916[df_916["hour"] == hour]
        if len(subset) == 0:
            continue
        rt_mean = subset["rt_price"].mean()
        rt_std = subset["rt_price"].std()
        da_mean = subset["da_price"].mean()
        delta_mean = (subset["rt_price"] - subset["da_price"]).mean()
        delta_std = (subset["rt_price"] - subset["da_price"]).std()

        # SGDFNet sMAPE if available
        sgdf_smape = float("nan")
        if sgdf_df is not None and not sgdf_df.empty:
            sgdf_hour_col = "hour_business" if "hour_business" in sgdf_df.columns else "hour"
            sgdf_aligned = subset.merge(
                sgdf_df[["business_day", sgdf_hour_col, "teacher_pred"]].rename(
                    columns={sgdf_hour_col: "hour"}
                ),
                on=["business_day", "hour"],
                how="inner",
            )
            if not sgdf_aligned.empty:
                sgdf_smape = smape_floor50(
                    sgdf_aligned["rt_price"].values,
                    sgdf_aligned["teacher_pred"].values,
                )

        hourly_stats.append({
            "hour": hour,
            "count": len(subset),
            "rt_mean": rt_mean,
            "rt_std": rt_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "sgdf_smape": sgdf_smape,
        })
        logger.info("  Hour %d: n=%d, rt_mean=%.1f, delta_std=%.1f, sgdf_smape=%.2f",
                     hour, len(subset), rt_mean, delta_std, sgdf_smape)

    # ── Analysis 2: By bucket ────────────────────────────────────────
    logger.info("\n── Analysis 2: By bucket ──")
    bucket_stats = []
    for bucket in ["normal", "spike", "negative"]:
        subset = df_916[df_916["bucket"] == bucket]
        if len(subset) == 0:
            continue
        rt_mean = subset["rt_price"].mean()
        delta_std = (subset["rt_price"] - subset["da_price"]).std()

        sgdf_smape = float("nan")
        if sgdf_df is not None and not sgdf_df.empty:
            sgdf_hour_col = "hour_business" if "hour_business" in sgdf_df.columns else "hour"
            sgdf_aligned = subset.merge(
                sgdf_df[["business_day", sgdf_hour_col, "teacher_pred"]].rename(
                    columns={sgdf_hour_col: "hour"}
                ),
                on=["business_day", "hour"],
                how="inner",
            )
            if not sgdf_aligned.empty:
                sgdf_smape = smape_floor50(
                    sgdf_aligned["rt_price"].values,
                    sgdf_aligned["teacher_pred"].values,
                )

        bucket_stats.append({
            "bucket": bucket,
            "count": len(subset),
            "rt_mean": rt_mean,
            "delta_std": delta_std,
            "sgdf_smape": sgdf_smape,
        })
        logger.info("  %s: n=%d, rt_mean=%.1f, delta_std=%.1f, sgdf_smape=%.2f",
                     bucket, len(subset), rt_mean, delta_std, sgdf_smape)

    # ── Analysis 3: By month ─────────────────────────────────────────
    logger.info("\n── Analysis 3: By month ──")
    df_916["month"] = df_916["ds"].dt.to_period("M").astype(str)
    monthly_stats = []
    for month in sorted(df_916["month"].unique()):
        subset = df_916[df_916["month"] == month]
        if len(subset) == 0:
            continue
        rt_mean = subset["rt_price"].mean()
        delta_std = (subset["rt_price"] - subset["da_price"]).std()

        sgdf_smape = float("nan")
        if sgdf_df is not None and not sgdf_df.empty:
            sgdf_hour_col = "hour_business" if "hour_business" in sgdf_df.columns else "hour"
            sgdf_aligned = subset.merge(
                sgdf_df[["business_day", sgdf_hour_col, "teacher_pred"]].rename(
                    columns={sgdf_hour_col: "hour"}
                ),
                on=["business_day", "hour"],
                how="inner",
            )
            if not sgdf_aligned.empty:
                sgdf_smape = smape_floor50(
                    sgdf_aligned["rt_price"].values,
                    sgdf_aligned["teacher_pred"].values,
                )

        monthly_stats.append({
            "month": month,
            "count": len(subset),
            "rt_mean": rt_mean,
            "delta_std": delta_std,
            "sgdf_smape": sgdf_smape,
        })
        logger.info("  %s: n=%d, rt_mean=%.1f, delta_std=%.1f, sgdf_smape=%.2f",
                     month, len(subset), rt_mean, delta_std, sgdf_smape)

    # ── Analysis 4: Feature correlation ──────────────────────────────
    logger.info("\n── Analysis 4: Feature correlation with delta ──")
    feature_cols = []
    for c in ["光伏预测", "新能源预测", "竞价空间", "净负荷", "系统负荷"]:
        if c in df_916.columns:
            feature_cols.append(c)

    feature_corr = {}
    delta = (df_916["rt_price"] - df_916["da_price"]).values
    for col in feature_cols:
        feat = df_916[col].astype(float).values
        if np.std(feat) > 1e-9 and np.std(delta) > 1e-9:
            corr = np.corrcoef(feat, delta)[0, 1]
            if not np.isnan(corr):
                feature_corr[col] = corr
                logger.info("  %s: corr(delta) = %.3f", col, corr)

    # ── Build report ─────────────────────────────────────────────────
    report_lines = [
        "# 9_16 Segment Failure Diagnosis",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Period:** {start_date.date()} to {end_date.date()}",
        f"**9_16 rows:** {len(df_916)}",
        "",
        "## Analysis 1: By Hour",
        "",
        "| Hour | Count | RT Mean | Delta Std | SGDFNet sMAPE |",
        "|------|-------|---------|-----------|---------------|",
    ]
    for row in hourly_stats:
        report_lines.append(
            f"| {row['hour']} | {row['count']} | {row['rt_mean']:.1f} | "
            f"{row['delta_std']:.1f} | {row['sgdf_smape']:.2f} |"
        )

    # Find hardest hour
    if hourly_stats:
        hardest_hour = max(hourly_stats, key=lambda x: x["sgdf_smape"] if not np.isnan(x["sgdf_smape"]) else 0)
        report_lines.extend([
            "",
            f"**Hardest hour:** {hardest_hour['hour']} (sMAPE = {hardest_hour['sgdf_smape']:.2f})",
        ])

    report_lines.extend([
        "",
        "## Analysis 2: By Bucket",
        "",
        "| Bucket | Count | RT Mean | Delta Std | SGDFNet sMAPE |",
        "|--------|-------|---------|-----------|---------------|",
    ])
    for row in bucket_stats:
        report_lines.append(
            f"| {row['bucket']} | {row['count']} | {row['rt_mean']:.1f} | "
            f"{row['delta_std']:.1f} | {row['sgdf_smape']:.2f} |"
        )

    report_lines.extend([
        "",
        "## Analysis 3: By Month",
        "",
        "| Month | Count | RT Mean | Delta Std | SGDFNet sMAPE |",
        "|-------|-------|---------|-----------|---------------|",
    ])
    for row in monthly_stats:
        report_lines.append(
            f"| {row['month']} | {row['count']} | {row['rt_mean']:.1f} | "
            f"{row['delta_std']:.1f} | {row['sgdf_smape']:.2f} |"
        )

    # Find hardest month
    if monthly_stats:
        hardest_month = max(monthly_stats, key=lambda x: x["sgdf_smape"] if not np.isnan(x["sgdf_smape"]) else 0)
        report_lines.extend([
            "",
            f"**Hardest month:** {hardest_month['month']} (sMAPE = {hardest_month['sgdf_smape']:.2f})",
        ])

    report_lines.extend([
        "",
        "## Analysis 4: Feature Correlation with Delta",
        "",
        "| Feature | Correlation |",
        "|---------|-------------|",
    ])
    for feat, corr in sorted(feature_corr.items(), key=lambda x: abs(x[1]), reverse=True):
        report_lines.append(f"| {feat} | {corr:.3f} |")

    report_lines.extend([
        "",
        "## Conclusion",
        "",
    ])

    # Determine if separate TrendKnightSolar916 is needed
    worst_smape = max((r["sgdf_smape"] for r in hourly_stats if not np.isnan(r["sgdf_smape"])), default=0)
    if worst_smape > 40:
        report_lines.append(
            f"**RECOMMENDATION: Consider training a dedicated `TrendKnightSolar916` model.** "
            f"Worst hour sMAPE = {worst_smape:.2f} > 40, indicating severe solar-volatility mismatch."
        )
    else:
        report_lines.append(
            f"**RECOMMENDATION: SGDFNet 9_16 performance is acceptable (worst hour sMAPE = {worst_smape:.2f}). "
            f"A dedicated TrendKnightSolar916 is NOT urgently needed."
        )

    report_path = out_dir / "DIAGNOSE_916_FAILURE.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("\n" + "=" * 60)
    logger.info("Diagnosis complete. Report: %s", report_path)


if __name__ == "__main__":
    main()
