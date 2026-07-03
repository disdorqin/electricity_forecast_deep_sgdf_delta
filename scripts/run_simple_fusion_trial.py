#!/usr/bin/env python
"""Simple Fusion Trial — Phase 5 Task E.

Tests 6 fusion schemes combining SGDFNet with TrendKnight-X variants:
  1. SGDFNet only
  2. 0.9 SGDFNet + 0.1 v3_multiscale
  3. 0.8 SGDFNet + 0.2 v3_multiscale
  4. 0.7 SGDFNet + 0.3 v3_multiscale
  5. period-aware: 1_8=0.8/0.2, 9_16=0.6/0.4, 17_24=0.8/0.2
  6. bucket-aware: normal=SGDFNet heavy, spike=more TrendKnight

Output (under --out-dir, default reports/local/phase5/fusion_trial/):
  leaderboard.csv
  period_metrics.csv
  bucket_metrics.csv
  monthly_metrics.csv
  fusion_gain_report.md

Usage:
    python scripts/run_simple_fusion_trial.py \\
        --start-date 2026-02-01 --end-date 2026-02-28 \\
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


# ── Data loading helpers ─────────────────────────────────────────────

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


def build_ground_truth(raw_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Build ground truth DataFrame for evaluation period."""
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
    df = df[(df[ts_col] >= start) & (df[ts_col] < end)]

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

    result = pd.DataFrame()
    result["ds"] = df[ts_col]
    result["da_price"] = df[da_col].astype(float)
    result["rt_price"] = df[rt_col].astype(float)

    # Business day alignment using unified module
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    result = add_business_time_columns(result, timestamp_col="ds", hour_col="hour")

    # Bucket
    result["bucket"] = "normal"
    result.loc[result["rt_price"].abs() > 500, "bucket"] = "spike"
    result.loc[result["rt_price"] < 0, "bucket"] = "negative"

    return result


def load_sgdfnet_predictions(source_repo_root: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Load SGDFNet predictions from teacher adapter."""
    try:
        from models.deep_sgdf_delta.teacher_adapters import sgdfnet_teacher
        df = sgdfnet_teacher.load_predictions(
            source_repo_root=source_repo_root,
            start_date=start_date,
            end_date=end_date,
        )
        return df
    except Exception as exc:
        logger.warning("Failed to load SGDFNet predictions: %s", exc)
        return None


def load_trendknight_predictions(report_dir: Path) -> pd.DataFrame | None:
    """Load TrendKnight predictions from ablation output if available."""
    # Look for v3_multiscale_tcn predictions
    pred_file = report_dir / "v3_multiscale_tcn_predictions.csv"
    if pred_file.exists():
        return pd.read_csv(pred_file)
    return None


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 50.0) -> float:
    """Compute sMAPE with floor-50 capping."""
    yt = np.clip(np.abs(y_true), floor, None)
    yp = np.clip(np.abs(y_pred), floor, None)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


# ── Fusion schemes ───────────────────────────────────────────────────

def fuse_sgdfnet_only(sgdf_pred: np.ndarray, tk_pred: np.ndarray, **kwargs) -> np.ndarray:
    """Scheme 1: SGDFNet only."""
    return sgdf_pred


def fuse_weighted(sgdf_pred: np.ndarray, tk_pred: np.ndarray, w_sgdf: float = 0.9) -> np.ndarray:
    """Scheme 2-4: weighted blend."""
    return w_sgdf * sgdf_pred + (1 - w_sgdf) * tk_pred


def fuse_period_aware(sgdf_pred: np.ndarray, tk_pred: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Scheme 5: period-aware blend."""
    result = np.zeros_like(sgdf_pred)
    for period, w_sgdf in [("1_8", 0.8), ("9_16", 0.6), ("17_24", 0.8)]:
        mask = periods == period
        if mask.any():
            result[mask] = w_sgdf * sgdf_pred[mask] + (1 - w_sgdf) * tk_pred[mask]
    return result


def fuse_bucket_aware(sgdf_pred: np.ndarray, tk_pred: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    """Scheme 6: bucket-aware blend."""
    result = np.zeros_like(sgdf_pred)
    for bucket, w_sgdf in [("normal", 0.9), ("spike", 0.6), ("negative", 0.7)]:
        mask = buckets == bucket
        if mask.any():
            result[mask] = w_sgdf * sgdf_pred[mask] + (1 - w_sgdf) * tk_pred[mask]
    return result


# ── Main evaluation ──────────────────────────────────────────────────

def evaluate_fusion_scheme(
    name: str,
    fused_pred: np.ndarray,
    gt_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate a fusion scheme and return metrics."""
    rt_true = gt_df["rt_price"].values.astype(float)
    rt_pred = fused_pred

    overall_smape = smape_floor50(rt_true, rt_pred)

    # Per-period
    period_metrics = {}
    for period in ["1_8", "9_16", "17_24"]:
        mask = gt_df["period"].values == period
        if mask.any():
            period_metrics[period] = smape_floor50(rt_true[mask], rt_pred[mask])

    # Per-bucket
    bucket_metrics = {}
    for bucket in ["normal", "spike", "negative"]:
        mask = gt_df["bucket"].values == bucket
        if mask.any():
            bucket_metrics[bucket] = smape_floor50(rt_true[mask], rt_pred[mask])

    return {
        "name": name,
        "overall_sMAPE_floor50": overall_smape,
        "period_metrics": period_metrics,
        "bucket_metrics": bucket_metrics,
    }


def _load_and_standardize_tk_predictions(path: Path) -> pd.DataFrame | None:
    """Load TrendKnight predictions CSV and standardize columns.

    Expected columns: business_day, hour_business (or hour), trend_pred (or rt_pred/y_pred).
    Returns standardized DataFrame with columns: business_day, hour_business, trend_pred.
    """
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return None

    if df.empty:
        return None

    # Standardize column names
    col_map = {}
    # Hour column
    for c in ["hour_business", "hour"]:
        if c in df.columns:
            col_map[c] = "hour_business"
            break
    # Prediction column
    for c in ["trend_pred", "rt_pred", "y_pred", "prediction"]:
        if c in df.columns:
            col_map[c] = "trend_pred"
            break

    if "trend_pred" not in col_map.values():
        logger.warning("No prediction column found in TK predictions")
        return None

    df = df.rename(columns=col_map)

    # Ensure business_day is datetime
    if "business_day" in df.columns:
        df["business_day"] = pd.to_datetime(df["business_day"])

    # Ensure hour_business is int
    if "hour_business" in df.columns:
        df["hour_business"] = df["hour_business"].astype(int)

    keep = ["business_day", "hour_business", "trend_pred"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def _write_skip_report(out_dir: Path, message: str):
    """Write a skip report when fusion cannot be evaluated."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# Simple Fusion Trial Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Status:** SKIPPED",
        "",
        f"**Reason:** {message}",
        "",
        "```",
        "synthetic_tk_proxy=false",
        "verdict=NO_DECISION",
        "```",
    ]
    (out_dir / "fusion_gain_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("Skip report written to %s", out_dir / "fusion_gain_report.md")


def main():
    parser = argparse.ArgumentParser(description="Simple Fusion Trial")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--source-repo-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="reports/local/phase6/fusion_trial")
    parser.add_argument("--trendknight-predictions", type=str, default=None,
                        help="Path to real TrendKnight predictions CSV (required for formal evaluation)")
    parser.add_argument("--allow-synthetic-tk", action="store_true",
                        help="Allow synthetic TK proxy (smoke test only, not for formal decision)")
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Simple Fusion Trial — Phase 6")
    logger.info("=" * 60)
    logger.info("  Period: %s to %s", start_date.date(), end_date.date())
    logger.info("  Output: %s", out_dir)
    logger.info("  TK predictions: %s", args.trendknight_predictions or "NOT PROVIDED")
    logger.info("  Allow synthetic TK: %s", args.allow_synthetic_tk)

    # Load data
    raw_df = load_raw_data(args.data_path)
    logger.info("Raw data: %d rows", len(raw_df))

    gt_df = build_ground_truth(raw_df, start_date, end_date)
    logger.info("Ground truth: %d rows", len(gt_df))

    # Load SGDFNet predictions
    sgdf_df = load_sgdfnet_predictions(args.source_repo_root, args.start_date, args.end_date)
    if sgdf_df is None or sgdf_df.empty:
        logger.error("SGDFNet predictions not available. Cannot run fusion trial.")
        sys.exit(1)
    logger.info("SGDFNet predictions: %d rows", len(sgdf_df))

    # Align SGDFNet to ground truth
    sgdf_hour_col = "hour_business" if "hour_business" in sgdf_df.columns else "hour"
    sgdf_aligned = gt_df.merge(
        sgdf_df[["business_day", sgdf_hour_col, "teacher_pred"]].rename(
            columns={sgdf_hour_col: "hour", "teacher_pred": "sgdf_pred"}
        ),
        on=["business_day", "hour"],
        how="inner",
    )
    if sgdf_aligned.empty:
        logger.error("No overlap between SGDFNet predictions and ground truth.")
        sys.exit(1)
    logger.info("Aligned SGDFNet: %d rows", len(sgdf_aligned))

    # Load TrendKnight predictions
    tk_pred = None
    synthetic_tk_proxy = False

    if args.trendknight_predictions:
        tk_path = Path(args.trendknight_predictions)
        if tk_path.exists():
            tk_df = _load_and_standardize_tk_predictions(tk_path)
            if tk_df is not None and not tk_df.empty:
                # Align TK to ground truth
                tk_hour_col = "hour_business" if "hour_business" in tk_df.columns else "hour"
                tk_aligned = sgdf_aligned.merge(
                    tk_df[["business_day", tk_hour_col, "trend_pred"]].rename(
                        columns={tk_hour_col: "hour"}
                    ),
                    on=["business_day", "hour"],
                    how="inner",
                )
                if not tk_aligned.empty:
                    tk_pred = tk_aligned["trend_pred"].values
                    sgdf_pred = tk_aligned["sgdf_pred"].values
                    periods = tk_aligned["period"].values
                    buckets = tk_aligned["bucket"].values
                    sgdf_aligned = tk_aligned
                    logger.info("Real TrendKnight predictions: %d rows", len(tk_aligned))
                else:
                    logger.warning("TrendKnight predictions loaded but no overlap with SGDFNet+GT")
            else:
                logger.warning("Failed to load TrendKnight predictions from %s", tk_path)
        else:
            logger.warning("TrendKnight predictions file not found: %s", tk_path)

    if tk_pred is None:
        if args.allow_synthetic_tk:
            logger.warning("Using synthetic TK proxy (SGDFNet + noise). SMOKE TEST ONLY.")
            np.random.seed(42)
            noise = np.random.normal(0, 20, len(sgdf_aligned))
            tk_pred = sgdf_aligned["sgdf_pred"].values + noise
            synthetic_tk_proxy = True
        else:
            # No real TK, no synthetic allowed → skip formal evaluation
            skip_msg = "TrendKnight predictions unavailable; fusion not evaluated"
            logger.warning(skip_msg)
            _write_skip_report(out_dir, skip_msg)
            return

    sgdf_pred = sgdf_aligned["sgdf_pred"].values
    periods = sgdf_aligned["period"].values
    buckets = sgdf_aligned["bucket"].values
    rt_true = sgdf_aligned["rt_price"].values

    # Evaluate all fusion schemes
    schemes = [
        ("sgdfnet_only", lambda: fuse_sgdfnet_only(sgdf_pred, tk_pred)),
        ("blend_09_01", lambda: fuse_weighted(sgdf_pred, tk_pred, 0.9)),
        ("blend_08_02", lambda: fuse_weighted(sgdf_pred, tk_pred, 0.8)),
        ("blend_07_03", lambda: fuse_weighted(sgdf_pred, tk_pred, 0.7)),
        ("period_aware", lambda: fuse_period_aware(sgdf_pred, tk_pred, periods)),
        ("bucket_aware", lambda: fuse_bucket_aware(sgdf_pred, tk_pred, buckets)),
    ]

    results = []
    for name, fuse_fn in schemes:
        fused = fuse_fn()
        metrics = evaluate_fusion_scheme(name, fused, sgdf_aligned)
        results.append(metrics)
        logger.info("  %-15s  overall_sMAPE = %.4f", name, metrics["overall_sMAPE_floor50"])

    # Build leaderboard
    leaderboard = pd.DataFrame([
        {
            "rank": i + 1,
            "name": r["name"],
            "overall_sMAPE_floor50": r["overall_sMAPE_floor50"],
            **{f"{p}_sMAPE": r["period_metrics"].get(p, float("nan")) for p in ["1_8", "9_16", "17_24"]},
        }
        for i, r in enumerate(sorted(results, key=lambda x: x["overall_sMAPE_floor50"]))
    ])
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    # Period metrics
    period_rows = []
    for r in results:
        for period, smape in r["period_metrics"].items():
            period_rows.append({"candidate": r["name"], "period": period, "sMAPE_floor50": smape})
    pd.DataFrame(period_rows).to_csv(out_dir / "period_metrics.csv", index=False)

    # Bucket metrics
    bucket_rows = []
    for r in results:
        for bucket, smape in r["bucket_metrics"].items():
            bucket_rows.append({"candidate": r["name"], "bucket": bucket, "sMAPE_floor50": smape})
    pd.DataFrame(bucket_rows).to_csv(out_dir / "bucket_metrics.csv", index=False)

    # Monthly metrics (single month for now)
    monthly_rows = [{"candidate": r["name"], "month": start_date.strftime("%Y-%m"), "sMAPE_floor50": r["overall_sMAPE_floor50"]} for r in results]
    pd.DataFrame(monthly_rows).to_csv(out_dir / "monthly_metrics.csv", index=False)

    # Fusion gain report
    baseline_smape = results[0]["overall_sMAPE_floor50"]  # sgdfnet_only
    best_result = min(results, key=lambda x: x["overall_sMAPE_floor50"])
    gain = baseline_smape - best_result["overall_sMAPE_floor50"]

    report_lines = [
        "# Simple Fusion Trial Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Period:** {start_date.date()} to {end_date.date()}",
        "",
        "## Results",
        "",
        "| Scheme | Overall sMAPE | Gain vs Baseline |",
        "|--------|---------------|------------------|",
    ]
    for r in sorted(results, key=lambda x: x["overall_sMAPE_floor50"]):
        g = baseline_smape - r["overall_sMAPE_floor50"]
        report_lines.append(f"| {r['name']} | {r['overall_sMAPE_floor50']:.4f} | {g:+.4f} |")

    report_lines.extend([
        "",
        "## Verdict",
        "",
    ])

    if synthetic_tk_proxy:
        report_lines.append("**SMOKE_ONLY_NOT_FOR_DECISION**: Using synthetic TK proxy. "
                           "Real TrendKnight predictions required for formal evaluation.")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("synthetic_tk_proxy=true")
        report_lines.append("verdict=SMOKE_ONLY_NOT_FOR_DECISION")
        report_lines.append("```")
    elif gain >= 0.3:
        report_lines.append(f"**GO**: Best fusion ({best_result['name']}) improves sMAPE by {gain:.4f} >= 0.3. Enter main fusion pipeline.")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("synthetic_tk_proxy=false")
        report_lines.append("verdict=GO")
        report_lines.append("```")
    elif gain >= 0.05:
        report_lines.append(f"**LOW-WEIGHT DIVERSITY**: Best fusion ({best_result['name']}) improves sMAPE by {gain:.4f} (0.05~0.3). Use as low-weight diversity model only.")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("synthetic_tk_proxy=false")
        report_lines.append("verdict=LOW-WEIGHT DIVERSITY ONLY")
        report_lines.append("```")
    else:
        report_lines.append(f"**NO-GO**: Best fusion ({best_result['name']}) improves sMAPE by {gain:.4f} < 0.05. Do not integrate. Fall back to SGDFNet + spike/negative modules.")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("synthetic_tk_proxy=false")
        report_lines.append("verdict=NO-GO")
        report_lines.append("```")

    report_path = out_dir / "fusion_gain_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("Fusion trial complete. Output: %s", out_dir)
    logger.info("Baseline (SGDFNet only): %.4f", baseline_smape)
    logger.info("Best fusion: %s (%.4f), gain = %.4f", best_result["name"], best_result["overall_sMAPE_floor50"], gain)
    logger.info("Verdict: %s", report_lines[-1])


if __name__ == "__main__":
    main()
