#!/usr/bin/env python
"""Phase 2 evaluation entry point for TrendKnight.

Computes comprehensive metrics from a predictions CSV and produces:
  - metrics_summary.json    (overall + segment + bucket metrics)
  - monthly_metrics.csv     (per-month breakdown)
  - segment_metrics.csv     (per-segment breakdown: 1_8, 9_16, 17_24)
  - bucket_metrics.csv      (per-bucket: normal, high_price, negative)
  - go_nogo.md              (Go / No-Go verdict report)

Usage:
    python scripts/evaluate_phase2_trendknight.py --predictions reports/local/phase2/xxx/predictions.csv
    python scripts/evaluate_phase2_trendknight.py --help
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

from models.deep_sgdf_delta.metrics import (  # noqa: E402
    compute_full_metrics,
    compute_monthly_metrics,
    compute_period_mask,
    smape_floor50,
    classify_spike,
    classify_negative,
    delta_mae,
)
from models.deep_sgdf_delta.evaluate import (  # noqa: E402
    PASS_THRESHOLD,
    SOFT_PASS_THRESHOLD,
    BASELINE_SGDFNET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_phase2_trendknight")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 evaluation entry point for TrendKnight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs:
  metrics_summary.json    Overall, segment, and bucket metrics
  monthly_metrics.csv     Per-month sMAPE_floor50 breakdown
  segment_metrics.csv     Per-segment metrics (1_8, 9_16, 17_24)
  bucket_metrics.csv      Per-bucket metrics (normal, high_price, negative)
  go_nogo.md              Go/No-Go verdict report

Examples:
  python scripts/evaluate_phase2_trendknight.py --predictions reports/local/phase2/xxx/predictions.csv
  python scripts/evaluate_phase2_trendknight.py --predictions pred.csv --sgdfnet-baseline 16.5 --run-id my_run
""",
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions.csv with columns: "
                             "business_day, hour, rt_actual, rt_pred, delta_target, delta_pred, da_anchor, hour")
    parser.add_argument("--sgdfnet-baseline", type=float, default=BASELINE_SGDFNET,
                        help=f"SGDFNet baseline sMAPE_floor50 for comparison (default: {BASELINE_SGDFNET:.4f})")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: same directory as predictions file)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID for report headers (default: derived from predictions path)")
    return parser.parse_args()


# ── Column standardisation ───────────────────────────────────────────

def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to the canonical set expected by metrics."""
    col_map = {}
    if "rt_actual" not in df.columns and "y_true" in df.columns:
        col_map["y_true"] = "rt_actual"
    if "rt_pred" not in df.columns and "y_pred" in df.columns:
        col_map["y_pred"] = "rt_pred"
    if "hour" not in df.columns and "target_hour" in df.columns:
        col_map["target_hour"] = "hour"
    if "hour" not in df.columns and "hour_business" in df.columns:
        col_map["hour_business"] = "hour"
    if "delta_target" not in df.columns and "delta_true" in df.columns:
        col_map["delta_true"] = "delta_target"
    if "delta_target" not in df.columns and "rt_actual" in df.columns and "da_anchor" in df.columns:
        # Derive delta_target from rt_actual - da_anchor
        df = df.copy()
        df["delta_target"] = pd.to_numeric(df.get("rt_actual", df.get("y_true")), errors="coerce") - \
                             pd.to_numeric(df["da_anchor"], errors="coerce")
    if "delta_pred" not in df.columns and "deep_delta_pred" in df.columns:
        col_map["deep_delta_pred"] = "delta_pred"
    if "delta_pred" not in df.columns and "rt_pred" in df.columns and "da_anchor" in df.columns:
        df = df.copy()
        df["delta_pred"] = pd.to_numeric(df.get("rt_pred", df.get("y_pred")), errors="coerce") - \
                           pd.to_numeric(df["da_anchor"], errors="coerce")

    df = df.rename(columns=col_map)
    return df


# ── Segment metrics ──────────────────────────────────────────────────

def _compute_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics per time-of-day segment (1_8, 9_16, 17_24)."""
    valid = df.dropna(subset=["rt_actual", "rt_pred"]).copy()
    if valid.empty:
        return pd.DataFrame()

    hours = valid["hour"].to_numpy(dtype=int)
    yt = valid["rt_actual"].to_numpy(dtype=float)
    yp = valid["rt_pred"].to_numpy(dtype=float)

    has_delta = "delta_target" in valid.columns and "delta_pred" in valid.columns
    if has_delta:
        dt = valid["delta_target"].to_numpy(dtype=float)
        dp = valid["delta_pred"].to_numpy(dtype=float)

    rows = []
    for period in ("1_8", "9_16", "17_24"):
        mask = compute_period_mask(hours, period)
        n = int(mask.sum())
        if n == 0:
            continue
        row = {
            "segment": period,
            "count": n,
            "sMAPE_floor50": smape_floor50(yt[mask], yp[mask]),
        }
        if has_delta:
            row["delta_mae"] = delta_mae(dt[mask], dp[mask])
        rows.append(row)

    return pd.DataFrame(rows)


# ── Bucket metrics ───────────────────────────────────────────────────

def _compute_bucket_metrics(
    df: pd.DataFrame,
    spike_threshold: float = 500.0,
) -> pd.DataFrame:
    """Compute metrics per price bucket (normal, high_price, negative)."""
    valid = df.dropna(subset=["rt_actual", "rt_pred"]).copy()
    if valid.empty:
        return pd.DataFrame()

    yt = valid["rt_actual"].to_numpy(dtype=float)
    yp = valid["rt_pred"].to_numpy(dtype=float)

    spike_mask = classify_spike(yt, spike_threshold)
    neg_mask = classify_negative(yt)
    normal_mask = ~spike_mask & ~neg_mask

    has_delta = "delta_target" in valid.columns and "delta_pred" in valid.columns
    if has_delta:
        dt = valid["delta_target"].to_numpy(dtype=float)
        dp = valid["delta_pred"].to_numpy(dtype=float)

    rows = []
    for label, mask in [("normal", normal_mask), ("high_price", spike_mask), ("negative", neg_mask)]:
        n = int(mask.sum())
        if n == 0:
            continue
        row = {
            "bucket": label,
            "count": n,
            "sMAPE_floor50": smape_floor50(yt[mask], yp[mask]),
        }
        if has_delta:
            row["delta_mae"] = delta_mae(dt[mask], dp[mask])
        rows.append(row)

    return pd.DataFrame(rows)


# ── Go/No-Go verdict ─────────────────────────────────────────────────

def _compute_verdict(
    overall_smape: float,
    sgdfnet_baseline: float,
) -> tuple[str, str]:
    """Return (verdict, detail) based on overall sMAPE and baseline."""
    if overall_smape < PASS_THRESHOLD:
        return "PASS", f"Overall sMAPE_floor50={overall_smape:.4f} < {PASS_THRESHOLD}"
    if overall_smape <= SOFT_PASS_THRESHOLD:
        return "SOFT_PASS", (
            f"Overall sMAPE_floor50={overall_smape:.4f} <= {SOFT_PASS_THRESHOLD}, "
            f"awaiting spike/negative module fusion"
        )
    if overall_smape < sgdfnet_baseline:
        return "BASELINE_PASS", (
            f"Overall sMAPE_floor50={overall_smape:.4f} < SGDFNet baseline {sgdfnet_baseline:.4f}"
        )
    return "NO-GO", (
        f"Overall sMAPE_floor50={overall_smape:.4f} >= SGDFNet baseline {sgdfnet_baseline:.4f}"
    )


# ── Report generation ────────────────────────────────────────────────

def _write_go_nogo(
    output_dir: Path,
    run_id: str,
    full_metrics: dict,
    monthly_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    verdict: str,
    verdict_detail: str,
    sgdfnet_baseline: float,
) -> None:
    """Write the go_nogo.md report."""
    overall = full_metrics.get("overall_sMAPE_floor50", float("nan"))
    monthly_avg = full_metrics.get("monthly_avg_sMAPE_floor50", float("nan"))

    lines = [
        f"# Go/No-Go Report: {run_id}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Verdict:** {verdict}",
        "",
        f"{verdict_detail}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall sMAPE_floor50 | {overall:.4f} |",
        f"| Monthly Avg sMAPE_floor50 | {monthly_avg:.4f} |",
        f"| Delta MAE | {full_metrics.get('delta_mae', float('nan')):.4f} |",
        f"| Rows Total | {full_metrics.get('rows_total', 0)} |",
        f"| Rows Missing | {full_metrics.get('rows_missing', 0)} |",
        "",
    ]

    # Segment table
    if not segment_df.empty:
        lines.append("## Segment Metrics")
        lines.append("")
        lines.append("| Segment | Count | sMAPE_floor50 |")
        lines.append("|---------|-------|---------------|")
        for _, row in segment_df.iterrows():
            lines.append(f"| {row['segment']} | {row['count']} | {row['sMAPE_floor50']:.4f} |")
        lines.append("")

    # Bucket table
    if not bucket_df.empty:
        lines.append("## Bucket Metrics")
        lines.append("")
        lines.append("| Bucket | Count | sMAPE_floor50 |")
        lines.append("|--------|-------|---------------|")
        for _, row in bucket_df.iterrows():
            lines.append(f"| {row['bucket']} | {row['count']} | {row['sMAPE_floor50']:.4f} |")
        lines.append("")

    # Monthly table
    if not monthly_df.empty:
        lines.append("## Monthly Breakdown")
        lines.append("")
        lines.append("| Month | sMAPE_floor50 | Count |")
        lines.append("|-------|---------------|-------|")
        for _, row in monthly_df.iterrows():
            lines.append(f"| {row['month']} | {row['sMAPE_floor50']:.4f} | {row['count']} |")
        lines.append("")

    lines.extend([
        "## Thresholds",
        "",
        f"- PASS: overall sMAPE_floor50 < {PASS_THRESHOLD}",
        f"- SOFT PASS: overall sMAPE_floor50 <= {SOFT_PASS_THRESHOLD}",
        f"- BASELINE PASS: better than SGDFNet corrected baseline {sgdfnet_baseline:.4f}",
        f"- NO-GO: worse than SGDFNet baseline or leakage detected",
        "",
    ])

    (output_dir / "go_nogo.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"go_nogo.md written — verdict: {verdict}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(pred_df)} predictions from {pred_path}")

    # Standardise columns
    pred_df = _standardise_columns(pred_df)

    # Ensure hour is integer
    if "hour" in pred_df.columns:
        pred_df["hour"] = pd.to_numeric(pred_df["hour"], errors="coerce").astype("Int64")

    # Ensure numeric columns
    for col in ["rt_actual", "rt_pred", "delta_target", "delta_pred", "da_anchor"]:
        if col in pred_df.columns:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

    # Output directory
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = pred_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ID
    run_id = args.run_id or output_dir.name or "eval_run"

    # ── Compute all metrics ──────────────────────────────────────
    logger.info("Computing full metrics...")
    full_metrics = compute_full_metrics(pred_df)
    logger.info(f"Overall sMAPE_floor50: {full_metrics.get('overall_sMAPE_floor50', float('nan')):.4f}")

    # Monthly metrics
    logger.info("Computing monthly metrics...")
    monthly_df = compute_monthly_metrics(pred_df)
    if not monthly_df.empty:
        monthly_avg = float(monthly_df["sMAPE_floor50"].mean())
        full_metrics["monthly_avg_sMAPE_floor50"] = monthly_avg
        logger.info(f"Monthly average sMAPE_floor50: {monthly_avg:.4f}")

    # Segment metrics
    logger.info("Computing segment metrics...")
    segment_df = _compute_segment_metrics(pred_df)

    # Bucket metrics
    logger.info("Computing bucket metrics...")
    bucket_df = _compute_bucket_metrics(pred_df)

    # Go/No-Go verdict
    overall = full_metrics.get("overall_sMAPE_floor50", float("nan"))
    verdict, verdict_detail = _compute_verdict(overall, args.sgdfnet_baseline)
    full_metrics["verdict"] = verdict
    full_metrics["verdict_detail"] = verdict_detail

    # ── Write outputs ────────────────────────────────────────────

    # metrics_summary.json
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"metrics_summary.json written to {output_dir}")

    # monthly_metrics.csv
    if not monthly_df.empty:
        monthly_df.to_csv(output_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info(f"monthly_metrics.csv written ({len(monthly_df)} months)")

    # segment_metrics.csv
    if not segment_df.empty:
        segment_df.to_csv(output_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info(f"segment_metrics.csv written ({len(segment_df)} segments)")

    # bucket_metrics.csv
    if not bucket_df.empty:
        bucket_df.to_csv(output_dir / "bucket_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info(f"bucket_metrics.csv written ({len(bucket_df)} buckets)")

    # go_nogo.md
    _write_go_nogo(
        output_dir, run_id, full_metrics,
        monthly_df, segment_df, bucket_df,
        verdict, verdict_detail, args.sgdfnet_baseline,
    )

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Phase 2 Evaluation Complete")
    print(f"  Run ID:    {run_id}")
    print(f"  Verdict:   {verdict}")
    print(f"  Overall sMAPE_floor50: {overall:.4f}")
    print(f"  SGDFNet baseline:      {args.sgdfnet_baseline:.4f}")
    print(f"  Improvement:           {args.sgdfnet_baseline - overall:+.4f}")
    if not monthly_df.empty:
        print(f"  Monthly avg:           {monthly_avg:.4f}")
    print(f"  Rows total:            {full_metrics.get('rows_total', 0)}")
    print(f"  Rows missing:          {full_metrics.get('rows_missing', 0)}")
    print("=" * 60)
    print()
    print("Artifacts:")
    for fname in ["metrics_summary.json", "monthly_metrics.csv",
                   "segment_metrics.csv", "bucket_metrics.csv", "go_nogo.md"]:
        fpath = output_dir / fname
        if fpath.exists():
            print(f"  {fpath}")


if __name__ == "__main__":
    main()
