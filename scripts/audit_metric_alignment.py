#!/usr/bin/env python
"""Metric Alignment Audit — cross-module sMAPE consistency check.

Compares predictions from multiple modules to verify metric alignment:
  1. DeepFinal eval  — reports/local/deep_final/predictions_2026_02.csv
  2. DeltaSupply     — artifacts/delta_supply/exp_2026_02/predictions.csv
  3. Raw data        — ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv

For each source, computes row_count, date_range, da_anchor stats, rt_actual
stats, and DA anchor sMAPE_floor50 using the CANONICAL formula from
models.deep_sgdf_delta.metrics.

Then finds the common intersection (merge on ds timestamp) and computes
sMAPE on common rows.

Verdict:
  PASS — common intersection sMAPE differs <= 0.1 pp
  WARN — differs <= 1.0 pp
  FAIL — differs > 1.0 pp

Multi-month mode (--months):
  When --months is provided (comma-separated YYYY-MM), loops over each month,
  computes DA anchor sMAPE from the raw data for each month, and outputs:
    - metric_alignment_monthly.csv (one row per month)
    - metric_alignment_summary.json (overall verdict + per-month details)
    - metric_alignment_report.md

Output (to --out-dir, default reports/local/risk_modules/metric_alignment/):
  metric_alignment_summary.json
  metric_alignment_rows.csv  (single-month mode)
  metric_alignment_monthly.csv  (multi-month mode)
  metric_alignment_report.md

Usage:
    # Single-month mode (backward compatible)
    python scripts/audit_metric_alignment.py \\
        --out-dir reports/local/risk_modules/metric_alignment

    # Multi-month mode
    python scripts/audit_metric_alignment.py \\
        --months 2026-01,2026-02,2026-03 \\
        --out-dir reports/local/risk_modules/metric_alignment
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

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.metrics import smape_floor50
from models.deep_sgdf_delta.realtime_column_mapping import rename_chinese_columns
from models.deep_sgdf_delta.business_time import add_business_time_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Source paths ─────────────────────────────────────────────────────

DEFAULT_DEEP_FINAL_PATH = "reports/local/deep_final/predictions_2026_02.csv"
DEFAULT_DELTA_SUPPLY_PATH = "artifacts/delta_supply/exp_2026_02/predictions.csv"
DEFAULT_RAW_DATA_PATH = "../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv"


# ── Loaders ──────────────────────────────────────────────────────────

def load_deep_final(path: Path) -> pd.DataFrame | None:
    """Load DeepFinal predictions CSV."""
    if not path.exists():
        logger.warning("DeepFinal predictions not found: %s", path)
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("DeepFinal loaded: %d rows from %s", len(df), path)
    return df


def load_delta_supply(path: Path) -> pd.DataFrame | None:
    """Load DeltaSupply predictions CSV."""
    if not path.exists():
        logger.warning("DeltaSupply predictions not found: %s", path)
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("DeltaSupply loaded: %d rows from %s", len(df), path)
    return df


def load_raw_data(path: Path) -> pd.DataFrame | None:
    """Load raw Shandong PMOS hourly CSV with Chinese column renaming."""
    if not path.exists():
        logger.warning("Raw data not found: %s", path)
        return None
    # Try GBK / gb18030 first (Chinese Windows CSV), then utf-8
    df = None
    for enc in ("gbk", "gb18030", "utf-8", "utf-8-sig"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("Raw data loaded with encoding=%s: %d rows from %s", enc, len(df), path)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        logger.error("Cannot decode raw data file: %s", path)
        return None

    # Rename Chinese columns to canonical English names
    df = rename_chinese_columns(df)
    return df


# ── Stats computation ────────────────────────────────────────────────

def compute_source_stats(df: pd.DataFrame, source_name: str, *, rt_col: str = "rt_actual") -> dict:
    """Compute per-source statistics.

    Expected columns after normalisation: ds, da_anchor, rt_actual (or rt_col).
    """
    stats: dict = {"source": source_name, "row_count": len(df)}

    # Date range
    if "ds" in df.columns:
        ds = pd.to_datetime(df["ds"], errors="coerce")
        stats["date_min"] = str(ds.min())
        stats["date_max"] = str(ds.max())
    else:
        stats["date_min"] = None
        stats["date_max"] = None

    # da_anchor stats
    if "da_anchor" in df.columns:
        da = df["da_anchor"].dropna().astype(float)
        if len(da) > 0:
            stats["da_anchor_mean"] = round(float(da.mean()), 4)
            stats["da_anchor_std"] = round(float(da.std()), 4)
            stats["da_anchor_min"] = round(float(da.min()), 4)
            stats["da_anchor_max"] = round(float(da.max()), 4)
        else:
            stats["da_anchor_mean"] = None
            stats["da_anchor_std"] = None
            stats["da_anchor_min"] = None
            stats["da_anchor_max"] = None
    else:
        stats["da_anchor_mean"] = None
        stats["da_anchor_std"] = None
        stats["da_anchor_min"] = None
        stats["da_anchor_max"] = None

    # rt_actual stats
    if rt_col in df.columns:
        rt = df[rt_col].dropna().astype(float)
        if len(rt) > 0:
            stats["rt_actual_mean"] = round(float(rt.mean()), 4)
            stats["rt_actual_std"] = round(float(rt.std()), 4)
            stats["rt_actual_min"] = round(float(rt.min()), 4)
            stats["rt_actual_max"] = round(float(rt.max()), 4)
        else:
            stats["rt_actual_mean"] = None
            stats["rt_actual_std"] = None
            stats["rt_actual_min"] = None
            stats["rt_actual_max"] = None
    else:
        stats["rt_actual_mean"] = None
        stats["rt_actual_std"] = None
        stats["rt_actual_min"] = None
        stats["rt_actual_max"] = None

    return stats


def compute_smape_for_source(
    df: pd.DataFrame,
    source_name: str,
    *,
    y_true_col: str = "rt_actual",
    y_pred_col: str = "rt_pred",
) -> float | None:
    """Compute sMAPE_floor50 for a source using the canonical formula.

    Returns None if required columns are missing or no valid rows.
    """
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        logger.warning(
            "%s: missing columns for sMAPE (%s, %s)", source_name, y_true_col, y_pred_col
        )
        return None
    valid = df.dropna(subset=[y_true_col, y_pred_col])
    if valid.empty:
        logger.warning("%s: no valid rows for sMAPE computation", source_name)
        return None
    yt = valid[y_true_col].to_numpy(dtype=float)
    yp = valid[y_pred_col].to_numpy(dtype=float)
    return round(smape_floor50(yt, yp), 4)


# ── Common intersection ──────────────────────────────────────────────

def find_common_intersection(
    dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Merge all sources on ds timestamp to find the common intersection.

    Returns the merged DataFrame or None if fewer than 2 sources have ds.
    """
    ds_dfs = {name: df for name, df in dfs.items() if "ds" in df.columns}
    if len(ds_dfs) < 2:
        logger.warning("Fewer than 2 sources have 'ds' column; cannot compute intersection")
        return None

    names = list(ds_dfs.keys())
    merged = ds_dfs[names[0]][["ds"]].copy()
    merged["ds"] = pd.to_datetime(merged["ds"], errors="coerce")
    merged = merged.dropna(subset=["ds"])

    for name in names[1:]:
        other = ds_dfs[name][["ds"]].copy()
        other["ds"] = pd.to_datetime(other["ds"], errors="coerce")
        other = other.dropna(subset=["ds"])
        merged = merged.merge(other, on="ds", how="inner")

    logger.info("Common intersection: %d rows across %d sources", len(merged), len(ds_dfs))
    return merged


def compute_common_smape(
    dfs: dict[str, pd.DataFrame],
    common_ds: pd.DataFrame,
    *,
    y_true_col: str = "rt_actual",
    y_pred_col: str = "rt_pred",
) -> dict:
    """Compute sMAPE on the common intersection for each source.

    For each source, merge with common_ds on ds, then compute sMAPE_floor50
    using the source's rt_actual vs rt_pred (or da_anchor as fallback for y_pred).
    """
    result: dict = {}
    common_timestamps = set(pd.to_datetime(common_ds["ds"], errors="coerce").dropna())

    for name, df in dfs.items():
        if "ds" not in df.columns:
            continue
        df_copy = df.copy()
        df_copy["ds"] = pd.to_datetime(df_copy["ds"], errors="coerce")
        df_common = df_copy[df_copy["ds"].isin(common_timestamps)]

        if df_common.empty:
            result[name] = {"common_rows": 0, "smape_common": None}
            continue

        # Determine y_true and y_pred columns
        yt_col = y_true_col if y_true_col in df_common.columns else None
        yp_col = y_pred_col if y_pred_col in df_common.columns else None

        if yt_col is None:
            result[name] = {"common_rows": len(df_common), "smape_common": None}
            continue

        # If no rt_pred, use da_anchor as the "prediction" (baseline)
        if yp_col is None:
            if "da_anchor" in df_common.columns:
                yp_col = "da_anchor"
            else:
                result[name] = {"common_rows": len(df_common), "smape_common": None}
                continue

        valid = df_common.dropna(subset=[yt_col, yp_col])
        if valid.empty:
            result[name] = {"common_rows": len(df_common), "smape_common": None}
            continue

        yt = valid[yt_col].to_numpy(dtype=float)
        yp = valid[yp_col].to_numpy(dtype=float)
        result[name] = {
            "common_rows": len(valid),
            "smape_common": round(smape_floor50(yt, yp), 4),
        }

    return result


# ── Verdict ──────────────────────────────────────────────────────────

def compute_verdict(common_smape_values: list[float]) -> tuple[str, float]:
    """Compute verdict based on sMAPE spread across sources on common rows.

    Returns (verdict, spread_pp).
    """
    if len(common_smape_values) < 2:
        return "INSUFFICIENT", 0.0

    spread = max(common_smape_values) - min(common_smape_values)
    # spread is already in percent scale (canonical formula uses 200x multiplier)
    spread_pp = round(spread, 4)

    if spread_pp <= 0.1:
        verdict = "PASS"
    elif spread_pp <= 1.0:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return verdict, spread_pp


def compute_multi_month_verdict(monthly_verdicts: list[str]) -> str:
    """Compute overall verdict from per-month verdicts.

    Rules:
      - If any month is FAIL → overall FAIL
      - If any month is WARN (and none FAIL) → overall WARN
      - If all PASS → overall PASS
      - If no months have verdict → INSUFFICIENT
    """
    if not monthly_verdicts:
        return "INSUFFICIENT"
    if "FAIL" in monthly_verdicts:
        return "FAIL"
    if "WARN" in monthly_verdicts:
        return "WARN"
    if all(v == "PASS" for v in monthly_verdicts):
        return "PASS"
    return "INSUFFICIENT"


# ── Multi-month audit ────────────────────────────────────────────────

def _filter_raw_for_month(raw_df: pd.DataFrame, month_str: str) -> pd.DataFrame:
    """Filter raw DataFrame to rows belonging to a given YYYY-MM month.

    Uses the ds column (timestamp). The month is determined by the calendar
    month of the ds timestamp.
    """
    df = raw_df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    month_series = df["ds"].dt.to_period("M").astype(str)
    filtered = df[month_series == month_str].copy()
    return filtered


def _compute_monthly_da_anchor_smape(month_df: pd.DataFrame) -> float | None:
    """Compute DA anchor sMAPE_floor50 for a single month's raw data.

    Uses rt_actual vs da_anchor (the DA anchor is the baseline prediction).
    Returns None if required columns are missing or no valid rows.
    """
    if "rt_actual" not in month_df.columns or "da_anchor" not in month_df.columns:
        logger.warning("Missing rt_actual or da_anchor columns for monthly sMAPE")
        return None
    valid = month_df.dropna(subset=["rt_actual", "da_anchor"])
    if valid.empty:
        return None
    yt = valid["rt_actual"].to_numpy(dtype=float)
    yp = valid["da_anchor"].to_numpy(dtype=float)
    return round(smape_floor50(yt, yp), 4)


def _compute_monthly_row_details(month_df: pd.DataFrame, month_str: str) -> dict:
    """Compute detailed row-level statistics for a single month.

    Returns a dict with:
      - month: the YYYY-MM string
      - row_count: number of rows
      - da_anchor_smape_fraction: sMAPE as fraction (0-2 scale)
      - da_anchor_smape_percent: sMAPE as percent (0-200 scale)
      - business_day_range: (min_bd, max_bd) tuple
      - hour_distribution: dict of hour -> count
      - missing_rows: number of expected hours minus actual rows
      - duplicate_rows: number of duplicate ds timestamps
    """
    row_count = len(month_df)

    # DA anchor sMAPE
    da_smape_percent = _compute_monthly_da_anchor_smape(month_df)
    # Fraction scale: percent / 100
    da_smape_fraction = round(da_smape_percent / 100.0, 6) if da_smape_percent is not None else None

    # Business day range
    if "business_day" in month_df.columns:
        bd = month_df["business_day"].dropna()
        if len(bd) > 0:
            bd_min = str(bd.min())[:10]
            bd_max = str(bd.max())[:10]
            business_day_range = f"{bd_min}~{bd_max}"
        else:
            business_day_range = "N/A"
    elif "ds" in month_df.columns:
        # Compute business_day on the fly
        df_bt = add_business_time_columns(month_df, timestamp_col="ds")
        bd = df_bt["business_day"].dropna()
        if len(bd) > 0:
            bd_min = str(bd.min())[:10]
            bd_max = str(bd.max())[:10]
            business_day_range = f"{bd_min}~{bd_max}"
        else:
            business_day_range = "N/A"
    else:
        business_day_range = "N/A"

    # Hour distribution
    if "hour_business" in month_df.columns:
        hour_dist_series = month_df["hour_business"].dropna().astype(int)
    elif "ds" in month_df.columns:
        df_bt = add_business_time_columns(month_df, timestamp_col="ds")
        hour_dist_series = df_bt["hour_business"].dropna().astype(int)
    else:
        hour_dist_series = pd.Series([], dtype=int)

    if len(hour_dist_series) > 0:
        hour_counts = hour_dist_series.value_counts().sort_index()
        hour_distribution = {str(int(h)): int(c) for h, c in hour_counts.items()}
    else:
        hour_distribution = {}

    # Missing rows: expected = number of unique business days * 24 hours
    if "ds" in month_df.columns:
        df_bt = add_business_time_columns(month_df, timestamp_col="ds")
        unique_bd = df_bt["business_day"].nunique()
        expected_rows = unique_bd * 24
        missing_rows = max(0, expected_rows - row_count)
    else:
        missing_rows = 0

    # Duplicate rows
    if "ds" in month_df.columns:
        ds_parsed = pd.to_datetime(month_df["ds"], errors="coerce").dropna()
        duplicate_rows = int(ds_parsed.duplicated().sum())
    else:
        duplicate_rows = 0

    return {
        "month": month_str,
        "row_count": row_count,
        "da_anchor_smape_fraction": da_smape_fraction,
        "da_anchor_smape_percent": da_smape_percent,
        "business_day_range": business_day_range,
        "hour_distribution": str(hour_distribution),
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
    }


def run_multi_month_audit(
    raw_df: pd.DataFrame,
    months: list[str],
    out_dir: Path,
) -> None:
    """Run multi-month metric alignment audit.

    For each month:
      1. Filter raw data for the month
      2. Add business time columns
      3. Compute DA anchor sMAPE
      4. Collect row-level details

    Outputs:
      - metric_alignment_monthly.csv
      - metric_alignment_summary.json
      - metric_alignment_report.md
    """
    logger.info("=" * 60)
    logger.info("Multi-Month Metric Alignment Audit")
    logger.info("Months: %s", months)
    logger.info("=" * 60)

    monthly_rows: list[dict] = []
    monthly_details: list[dict] = []
    monthly_verdicts: list[str] = []

    for month_str in months:
        logger.info("Processing month: %s", month_str)
        month_df = _filter_raw_for_month(raw_df, month_str)

        if month_df.empty:
            logger.warning("No data found for month %s", month_str)
            monthly_details.append({
                "month": month_str,
                "row_count": 0,
                "da_anchor_smape_fraction": None,
                "da_anchor_smape_percent": None,
                "business_day_range": "N/A",
                "hour_distribution": "{}",
                "missing_rows": 0,
                "duplicate_rows": 0,
                "verdict": "NO_DATA",
            })
            monthly_verdicts.append("NO_DATA")
            continue

        # Add business time columns
        month_df = add_business_time_columns(month_df, timestamp_col="ds")

        # Compute DA anchor sMAPE
        da_smape = _compute_monthly_da_anchor_smape(month_df)
        logger.info("  %s: rows=%d, da_anchor_sMAPE=%.4f", month_str, len(month_df), da_smape if da_smape is not None else float("nan"))

        # Compute detailed row info
        detail = _compute_monthly_row_details(month_df, month_str)

        # Per-month verdict: based on data integrity checks
        # PASS if row_count correct, no missing/duplicate rows
        # WARN if minor issues (e.g. missing rows explained by DST/leap)
        # FAIL if major integrity issues
        if da_smape is not None:
            expected_rows = len(month_df)
            n_missing = detail.get("missing_rows", 0)
            n_dup = detail.get("duplicate_rows", 0)
            if n_dup > 0:
                verdict = "FAIL"
            elif n_missing > 48:
                verdict = "FAIL"
            elif n_missing > 0:
                verdict = "WARN"
            else:
                verdict = "PASS"
        else:
            verdict = "INSUFFICIENT"

        detail["verdict"] = verdict
        monthly_details.append(detail)
        monthly_verdicts.append(verdict)

        monthly_rows.append({
            "month": month_str,
            "row_count": detail["row_count"],
            "da_anchor_smape_fraction": detail["da_anchor_smape_fraction"],
            "da_anchor_smape_percent": detail["da_anchor_smape_percent"],
            "business_day_range": detail["business_day_range"],
            "hour_distribution": detail["hour_distribution"],
            "missing_rows": detail["missing_rows"],
            "duplicate_rows": detail["duplicate_rows"],
        })

    # Overall verdict
    overall_verdict = compute_multi_month_verdict(monthly_verdicts)
    logger.info("Overall verdict: %s", overall_verdict)

    # ── Write outputs ────────────────────────────────────────────────

    # 1. Monthly CSV
    monthly_df = pd.DataFrame(monthly_rows)
    csv_path = out_dir / "metric_alignment_monthly.csv"
    monthly_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s", csv_path)

    # 2. JSON summary
    summary = {
        "audit_timestamp": datetime.now().isoformat(),
        "audit_mode": "multi_month",
        "canonical_formula": "models.deep_sgdf_delta.metrics.smape_floor50",
        "multiplier": 200,
        "floor": 50.0,
        "months_audited": months,
        "per_month_details": monthly_details,
        "overall_verdict": overall_verdict,
        "verdict_rules": {
            "per_month": "PASS if no missing/dup rows, WARN if <=48 missing, FAIL if >48 missing or duplicates",
            "overall": "FAIL if any month FAIL, WARN if any WARN, else PASS",
        },
    }
    json_path = out_dir / "metric_alignment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    # 3. Markdown report
    report_md = _generate_multi_month_report(monthly_details, overall_verdict, months)
    md_path = out_dir / "metric_alignment_report.md"
    md_path.write_text(report_md, encoding="utf-8")
    logger.info("Wrote %s", md_path)

    logger.info("=" * 60)
    logger.info("Multi-month audit complete. Overall verdict: %s", overall_verdict)
    logger.info("=" * 60)

    if overall_verdict == "FAIL":
        sys.exit(1)


def _generate_multi_month_report(
    monthly_details: list[dict],
    overall_verdict: str,
    months: list[str],
) -> str:
    """Generate the markdown report for multi-month audit."""
    lines = [
        "# Metric Alignment Audit Report (Multi-Month)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Canonical formula:** `models.deep_sgdf_delta.metrics.smape_floor50` (multiplier=200, percent scale)",
        f"**Months audited:** {', '.join(months)}",
        "",
        "## Per-Month Summary",
        "",
        "| Month | Rows | DA sMAPE (%) | DA sMAPE (frac) | BD Range | Missing | Duplicates | Verdict |",
        "|-------|------|-------------|-----------------|----------|---------|------------|---------|",
    ]

    for detail in monthly_details:
        smape_pct = f"{detail['da_anchor_smape_percent']:.4f}" if detail["da_anchor_smape_percent"] is not None else "N/A"
        smape_frac = f"{detail['da_anchor_smape_fraction']:.6f}" if detail["da_anchor_smape_fraction"] is not None else "N/A"
        lines.append(
            f"| {detail['month']} "
            f"| {detail['row_count']} "
            f"| {smape_pct} "
            f"| {smape_frac} "
            f"| {detail['business_day_range']} "
            f"| {detail['missing_rows']} "
            f"| {detail['duplicate_rows']} "
            f"| {detail['verdict']} |"
        )

    # Hour distribution section
    lines.extend([
        "",
        "## Hour Distribution",
        "",
    ])
    for detail in monthly_details:
        lines.append(f"### {detail['month']}")
        lines.append(f"```")
        lines.append(detail["hour_distribution"])
        lines.append(f"```")
        lines.append("")

    # Verdict section
    lines.extend([
        f"## Overall Verdict: **{overall_verdict}**",
        "",
        "Per-month rules: PASS if no missing/dup rows, WARN if <=48 missing, FAIL if >48 missing or duplicates",
        "",
        "Overall rules: FAIL if any month FAIL, WARN if any WARN, else PASS",
        "",
    ])

    if overall_verdict == "PASS":
        lines.append("All months pass metric alignment data integrity checks.")
    elif overall_verdict == "WARN":
        lines.append("Some months show minor data integrity issues (missing rows). Review data completeness.")
    elif overall_verdict == "FAIL":
        lines.append("One or more months have data integrity issues (>48 missing rows or duplicates). Investigate root cause.")
    else:
        lines.append("Insufficient data for multi-month audit.")

    return "\n".join(lines)


# ── Report generation (single-month mode) ────────────────────────────

def build_comparison_rows(
    source_stats: list[dict],
    source_smapes: dict[str, float | None],
    common_smape: dict[str, dict],
) -> pd.DataFrame:
    """Build a comparison table DataFrame."""
    rows = []
    for stats in source_stats:
        name = stats["source"]
        rows.append({
            "source": name,
            "row_count": stats["row_count"],
            "date_min": stats.get("date_min"),
            "date_max": stats.get("date_max"),
            "da_anchor_mean": stats.get("da_anchor_mean"),
            "da_anchor_std": stats.get("da_anchor_std"),
            "rt_actual_mean": stats.get("rt_actual_mean"),
            "rt_actual_std": stats.get("rt_actual_std"),
            "smape_floor50_full": source_smapes.get(name),
            "common_rows": common_smape.get(name, {}).get("common_rows", 0),
            "smape_floor50_common": common_smape.get(name, {}).get("smape_common"),
        })
    return pd.DataFrame(rows)


def generate_report_md(
    source_stats: list[dict],
    source_smapes: dict[str, float | None],
    common_smape: dict[str, dict],
    verdict: str,
    spread_pp: float,
    comparison_df: pd.DataFrame,
) -> str:
    """Generate the markdown audit report."""
    lines = [
        "# Metric Alignment Audit Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Canonical formula:** `models.deep_sgdf_delta.metrics.smape_floor50` (multiplier=200, percent scale)",
        "",
        "## Per-Source Statistics",
        "",
        "| Source | Rows | Date Min | Date Max | DA mean | DA std | RT mean | RT std |",
        "|--------|------|----------|----------|---------|--------|---------|--------|",
    ]
    for stats in source_stats:
        lines.append(
            f"| {stats['source']} | {stats['row_count']} "
            f"| {stats.get('date_min', 'N/A')} | {stats.get('date_max', 'N/A')} "
            f"| {stats.get('da_anchor_mean', 'N/A')} | {stats.get('da_anchor_std', 'N/A')} "
            f"| {stats.get('rt_actual_mean', 'N/A')} | {stats.get('rt_actual_std', 'N/A')} |"
        )

    lines.extend([
        "",
        "## sMAPE_floor50 Comparison",
        "",
        "| Source | Full sMAPE | Common Rows | Common sMAPE |",
        "|--------|-----------|-------------|-------------|",
    ])
    for stats in source_stats:
        name = stats["source"]
        full_smape = source_smapes.get(name)
        cs = common_smape.get(name, {})
        common_rows = cs.get("common_rows", 0)
        common_val = cs.get("smape_common")
        lines.append(
            f"| {name} "
            f"| {f'{full_smape:.4f}' if full_smape is not None else 'N/A'} "
            f"| {common_rows} "
            f"| {f'{common_val:.4f}' if common_val is not None else 'N/A'} |"
        )

    common_vals = [
        cs["smape_common"]
        for cs in common_smape.values()
        if cs.get("smape_common") is not None
    ]
    if len(common_vals) >= 2:
        lines.extend([
            "",
            f"- Common sMAPE min: {min(common_vals):.4f}",
            f"- Common sMAPE max: {max(common_vals):.4f}",
            f"- Spread: {spread_pp:.4f} pp",
        ])

    lines.extend([
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"Spread threshold: PASS <= 0.1 pp, WARN <= 1.0 pp, FAIL > 1.0 pp",
        "",
    ])

    if verdict == "PASS":
        lines.append("All modules are metric-aligned within 0.1 pp on the common intersection.")
    elif verdict == "WARN":
        lines.append("Modules show mild metric divergence (<= 1.0 pp). Review feature alignment.")
    elif verdict == "FAIL":
        lines.append("Modules are NOT metric-aligned (> 1.0 pp divergence). Investigate root cause.")
    else:
        lines.append("Insufficient sources for cross-module comparison.")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Metric Alignment Audit")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/local/risk_modules/metric_alignment",
    )
    parser.add_argument("--deep-final-path", type=str, default=DEFAULT_DEEP_FINAL_PATH)
    parser.add_argument("--delta-supply-path", type=str, default=DEFAULT_DELTA_SUPPLY_PATH)
    parser.add_argument("--raw-data-path", type=str, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument(
        "--months",
        type=str,
        default=None,
        help="Comma-separated list of YYYY-MM months for multi-month audit "
             "(e.g. 2026-01,2026-02,2026-03). When provided, runs multi-month mode.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Multi-month mode ─────────────────────────────────────────────
    if args.months is not None:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
        if not months:
            logger.error("No valid months provided in --months argument")
            sys.exit(1)

        # Load raw data
        raw_df = load_raw_data(PROJECT_ROOT / args.raw_data_path)
        if raw_df is None:
            logger.error("Raw data not found. Cannot run multi-month audit.")
            sys.exit(1)

        run_multi_month_audit(raw_df, months, out_dir)
        return

    # ── Single-month mode (backward compatible) ──────────────────────
    logger.info("=" * 60)
    logger.info("Metric Alignment Audit")
    logger.info("=" * 60)

    # ── Load sources ─────────────────────────────────────────────────
    deep_final_df = load_deep_final(PROJECT_ROOT / args.deep_final_path)
    delta_supply_df = load_delta_supply(PROJECT_ROOT / args.delta_supply_path)
    raw_df = load_raw_data(PROJECT_ROOT / args.raw_data_path)

    sources: dict[str, pd.DataFrame] = {}
    if deep_final_df is not None:
        sources["DeepFinal"] = deep_final_df
    if delta_supply_df is not None:
        sources["DeltaSupply"] = delta_supply_df
    if raw_df is not None:
        sources["RawData"] = raw_df

    if len(sources) == 0:
        logger.error("No sources loaded. Cannot proceed.")
        sys.exit(1)

    logger.info("Loaded %d sources: %s", len(sources), list(sources.keys()))

    # ── Per-source stats ─────────────────────────────────────────────
    source_stats: list[dict] = []
    source_smapes: dict[str, float | None] = {}

    for name, df in sources.items():
        stats = compute_source_stats(df, name)
        source_stats.append(stats)
        logger.info(
            "  %s: rows=%d, da_mean=%s, rt_mean=%s",
            name, stats["row_count"], stats.get("da_anchor_mean"), stats.get("rt_actual_mean"),
        )

        # sMAPE on full source
        smape_val = compute_smape_for_source(df, name)
        source_smapes[name] = smape_val
        if smape_val is not None:
            logger.info("  %s: full sMAPE_floor50 = %.4f", name, smape_val)

    # ── Common intersection ──────────────────────────────────────────
    common_ds = find_common_intersection(sources)
    common_smape: dict[str, dict] = {}

    if common_ds is not None and not common_ds.empty:
        common_smape = compute_common_smape(sources, common_ds)
        for name, cs in common_smape.items():
            if cs.get("smape_common") is not None:
                logger.info(
                    "  %s: common_rows=%d, common_sMAPE=%.4f",
                    name, cs["common_rows"], cs["smape_common"],
                )
    else:
        logger.warning("No common intersection found across sources")

    # ── Verdict ──────────────────────────────────────────────────────
    common_vals = [
        cs["smape_common"]
        for cs in common_smape.values()
        if cs.get("smape_common") is not None
    ]
    verdict, spread_pp = compute_verdict(common_vals)
    logger.info("Verdict: %s (spread=%.4f pp)", verdict, spread_pp)

    # ── Build comparison table ───────────────────────────────────────
    comparison_df = build_comparison_rows(source_stats, source_smapes, common_smape)

    # ── Write outputs ────────────────────────────────────────────────

    # 1. JSON summary
    summary = {
        "audit_timestamp": datetime.now().isoformat(),
        "canonical_formula": "models.deep_sgdf_delta.metrics.smape_floor50",
        "multiplier": 200,
        "floor": 50.0,
        "sources_loaded": list(sources.keys()),
        "per_source_stats": source_stats,
        "per_source_smape_full": {k: v for k, v in source_smapes.items()},
        "common_intersection_rows": len(common_ds) if common_ds is not None else 0,
        "common_smape": common_smape,
        "verdict": verdict,
        "spread_pp": spread_pp,
    }
    json_path = out_dir / "metric_alignment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    # 2. CSV comparison table
    csv_path = out_dir / "metric_alignment_rows.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s", csv_path)

    # 3. Markdown report
    report_md = generate_report_md(source_stats, source_smapes, common_smape, verdict, spread_pp, comparison_df)
    md_path = out_dir / "metric_alignment_report.md"
    md_path.write_text(report_md, encoding="utf-8")
    logger.info("Wrote %s", md_path)

    logger.info("=" * 60)
    logger.info("Audit complete. Verdict: %s", verdict)
    logger.info("=" * 60)

    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
