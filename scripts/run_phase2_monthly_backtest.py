#!/usr/bin/env python
"""Phase 2 monthly backtest runner for TrendKnight.

Walks through each month in the evaluation window:
  1. Train a fresh model on all data before the month
  2. Predict every day within the month
  3. Collect predictions across all months
  4. Run full evaluation on the combined predictions

Produces:
  - predictions.csv          (all monthly predictions with ground truth)
  - metrics_summary.json     (overall metrics)
  - monthly_metrics.csv      (per-month breakdown)
  - segment_metrics.csv      (per-segment: 1_8, 9_16, 17_24)
  - bucket_metrics.csv       (per-bucket: normal, high_price, negative)
  - go_nogo.md               (verdict report)
  - runtime_report.json      (per-month timing and status)

Usage:
    python scripts/run_phase2_monthly_backtest.py --profile v2_day_tcn --start-date 2026-01-01 --end-date 2026-05-11
    python scripts/run_phase2_monthly_backtest.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

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
logger = logging.getLogger("run_phase2_monthly_backtest")

# ── Profile registry (same as training script) ──────────────────────

PROFILES: dict[str, dict] = {
    "v1_hourly_tcn": {
        "version": "v1", "backbone": "tcn", "blend": "deep_only",
        "description": "V1 per-hour model, TCN backbone",
    },
    "v1_hourly_gru": {
        "version": "v1", "backbone": "gru", "blend": "deep_only",
        "description": "V1 per-hour model, GRU backbone",
    },
    "v2_day_tcn": {
        "version": "v2", "backbone": "tcn", "blend": "deep_only",
        "description": "V2 day-level model, TCN backbone",
    },
    "v2_day_gru": {
        "version": "v2", "backbone": "gru", "blend": "deep_only",
        "description": "V2 day-level model, GRU backbone",
    },
    "v2_day_transformer_tiny": {
        "version": "v2", "backbone": "transformer_tiny", "blend": "deep_only",
        "description": "V2 day-level model, Transformer-tiny backbone",
    },
    "v2_residual_sgdfnet": {
        "version": "v2", "backbone": "tcn", "blend": "sgdfnet_residual",
        "description": "V2 day-level, TCN, SGDFNet residual blend",
    },
    "v2_blend_sgdfnet": {
        "version": "v2", "backbone": "tcn", "blend": "sgdfnet_blend",
        "description": "V2 day-level, TCN, SGDFNet weighted blend",
    },
}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 monthly backtest runner for TrendKnight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Enumerate all months in [start_date, end_date]
  2. For each month:
     a. Train a model on all data before the month starts
     b. Predict every business day within the month
     c. Merge predictions with ground truth
  3. Combine all monthly predictions
  4. Run full evaluation (metrics, segments, buckets, go/no-go)

Examples:
  python scripts/run_phase2_monthly_backtest.py --profile v2_day_tcn --start-date 2026-01-01 --end-date 2026-05-11
  python scripts/run_phase2_monthly_backtest.py --profile v1_hourly_gru --start-date 2026-03-01 --end-date 2026-04-30 --fast-dev-run
  python scripts/run_phase2_monthly_backtest.py --profile v2_day_transformer_tiny --amp --device cuda
""",
    )
    parser.add_argument("--profile", type=str, required=True, choices=list(PROFILES.keys()),
                        help="Model profile to use")
    parser.add_argument("--start-date", type=str, default="2026-01-01",
                        help="Start of evaluation period (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2026-05-11",
                        help="End of evaluation period (YYYY-MM-DD)")
    parser.add_argument("--sgdfnet-root", type=str, default=None,
                        help="Path to SGDFNet project root (contains src/sgdfnet/)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to raw data file (default: auto-detect)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: reports/local/phase2/{run_id})")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP (automatic mixed precision)")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Quick sanity run: max 2 months, 3 epochs, tiny data")
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────

def _resolve_data_path(args: argparse.Namespace) -> Path:
    """Find the raw data file."""
    if args.data_path:
        p = Path(args.data_path)
        if p.exists():
            return p

    candidates = [
        PROJECT_ROOT / "data" / "shandong_pmos_hourly.xlsx",
        PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "data" / "shandong_pmos_hourly.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find data file. Use --data-path to specify.\n"
        f"Tried: {candidates}"
    )


def _load_raw_data(data_path: Path, sgdfnet_root: str | None) -> pd.DataFrame:
    """Load the raw hourly dataset."""
    if sgdfnet_root:
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root
        find_sgdfnet_root(sgdfnet_root)

    from models.deep_sgdf_delta.sgdfnet_bridge import load_dataset
    logger.info(f"Loading data from {data_path}")
    raw_df = load_dataset(str(data_path))
    logger.info(f"Data loaded: {len(raw_df)} rows")
    return raw_df


# ── Month enumeration ───────────────────────────────────────────────

def _enumerate_months(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (month_start, month_end) tuples covering the date range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    months = []
    current = start.replace(day=1)
    while current <= end:
        month_start = max(current, start)
        # Last day of this month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        month_end = min(next_month - pd.Timedelta(days=1), end)

        months.append((month_start, month_end))
        current = next_month

    return months


def _enumerate_business_days(month_start: pd.Timestamp, month_end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return business days within a month range."""
    all_days = pd.date_range(start=month_start, end=month_end, freq="D")
    # Business days only (Mon-Fri)
    return [d for d in all_days if d.dayofweek < 5]


# ── V1 backtest per month ───────────────────────────────────────────

def _backtest_month_v1(
    raw_df: pd.DataFrame,
    profile: dict,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    device_str: str,
    amp: bool,
    fast_dev_run: bool,
    val_days: int,
) -> pd.DataFrame:
    """Train on pre-month data, predict every business day in the month.

    Returns a DataFrame with predictions merged with ground truth.
    """
    import torch
    from models.deep_sgdf_delta.train import TrainConfig, train_model
    from models.deep_sgdf_delta.predict import predict_delta
    from models.deep_sgdf_delta.dataset import DEFAULT_FEATURE_CONFIG, build_predict_dataset, _collate_fn

    decision_day = month_start
    logger.info(f"[V1] Training for decision_day={decision_day.date()}")

    # Training config
    train_config = TrainConfig(
        hidden_dim=64, num_layers=2, dropout=0.1,
        backbone=profile["backbone"],
        tcn_kernel_size=3, tcn_dilation_base=2,
        segment_embed_dim=8,
        use_global_residual=True, global_residual_weight=0.3,
        amp_enabled=amp,
        epochs=3 if fast_dev_run else 30,
        batch_size=8 if fast_dev_run else 256,
        learning_rate=1e-3, weight_decay=1e-4,
        early_stopping_patience=2 if fast_dev_run else 5,
        device=device_str,
        window_days=7, seed=42,
        val_days=min(val_days, 7) if fast_dev_run else val_days,
        train_min_rows=100 if fast_dev_run else 2160,
    )

    # Train
    result = train_model(
        raw_df, DEFAULT_FEATURE_CONFIG, train_config,
        decision_day=decision_day,
    )
    model = result["model"]
    feature_cols = result["feature_cols"]

    # Resolve device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Predict each day
    business_days = _enumerate_business_days(month_start, month_end)
    if fast_dev_run:
        business_days = business_days[:3]  # limit to 3 days

    all_preds = []
    for day in business_days:
        try:
            pred_ds, _ = build_predict_dataset(
                raw_df, DEFAULT_FEATURE_CONFIG, target_day=day,
            )
            pred_df = predict_delta(model, pred_ds, device, batch_size=256)
            pred_df["business_day"] = day
            all_preds.append(pred_df)
        except Exception as e:
            logger.warning(f"[V1] Failed to predict day {day.date()}: {e}")
            continue

    if not all_preds:
        return pd.DataFrame()

    combined = pd.concat(all_preds, ignore_index=True)

    # Merge with ground truth
    combined = _merge_ground_truth_v1(combined, raw_df)
    return combined


# ── V2 backtest per month ───────────────────────────────────────────

def _backtest_month_v2(
    raw_df: pd.DataFrame,
    profile: dict,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    device_str: str,
    amp: bool,
    fast_dev_run: bool,
    val_days: int,
) -> pd.DataFrame:
    """Train on pre-month data, predict every business day in the month.

    Returns a DataFrame with predictions merged with ground truth.
    """
    import torch
    from models.deep_sgdf_delta.train_v2 import TrainV2Config, train_model_v2
    from models.deep_sgdf_delta.predict_v2 import predict_delta_v2
    from models.deep_sgdf_delta.dataset_v2 import DEFAULT_FEATURE_CONFIG, build_predict_dataset_v2

    decision_day = month_start
    logger.info(f"[V2] Training for decision_day={decision_day.date()}")

    # Training config
    train_config = TrainV2Config(
        hidden_dim=64, num_layers=2, dropout=0.1,
        backbone=profile["backbone"],
        tcn_kernel_size=3, tcn_dilation_base=2,
        transformer_nhead=4, transformer_dim_ff=128,
        hour_embed_dim=8, segment_embed_dim=8,
        use_residual_head=True, residual_weight=0.3,
        epochs=3 if fast_dev_run else 30,
        batch_size=8 if fast_dev_run else 64,
        learning_rate=1e-3, weight_decay=1e-4,
        early_stopping_patience=2 if fast_dev_run else 5,
        amp_enabled=amp,
        device=device_str,
        val_days=min(val_days, 7) if fast_dev_run else val_days,
        train_min_days=10 if fast_dev_run else 90,
        seed=42,
    )

    # Train
    result = train_model_v2(
        raw_df, DEFAULT_FEATURE_CONFIG, train_config,
        decision_day=decision_day,
        fast_dev_run=fast_dev_run,
    )
    model = result["model"]

    # Resolve device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Predict each day
    business_days = _enumerate_business_days(month_start, month_end)
    if fast_dev_run:
        business_days = business_days[:3]

    all_preds = []
    for day in business_days:
        try:
            pred_ds, _ = build_predict_dataset_v2(
                raw_df, DEFAULT_FEATURE_CONFIG, target_day=day,
            )
            pred_df = predict_delta_v2(model, pred_ds, device, batch_size=64)
            # Filter to valid hours
            pred_df = pred_df[pred_df["valid"] == True].copy()
            pred_df = pred_df.drop(columns=["valid"], errors="ignore")
            all_preds.append(pred_df)
        except Exception as e:
            logger.warning(f"[V2] Failed to predict day {day.date()}: {e}")
            continue

    if not all_preds:
        return pd.DataFrame()

    combined = pd.concat(all_preds, ignore_index=True)

    # Merge with ground truth
    combined = _merge_ground_truth_v2(combined, raw_df)
    return combined


# ── Ground truth merge ───────────────────────────────────────────────

def _merge_ground_truth_v1(pred_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Merge V1 predictions with ground truth from raw data."""
    from models.deep_sgdf_delta.sgdfnet_bridge import (
        add_business_time_columns, TIMESTAMP_COL, RT_COL, DA_COL,
    )

    # Build ground truth frame
    gt = raw_df.copy()
    gt = add_business_time_columns(gt)
    gt["rt_actual"] = pd.to_numeric(gt[RT_COL], errors="coerce")
    gt["da_anchor_gt"] = pd.to_numeric(gt[DA_COL], errors="coerce")
    gt["delta_target"] = gt["rt_actual"] - gt["da_anchor_gt"]
    gt["hour"] = gt["target_hour"].astype(int)
    gt["business_day"] = pd.to_datetime(gt["business_day"]).dt.normalize()

    gt_cols = ["business_day", "hour", "rt_actual", "delta_target", "da_anchor_gt"]
    gt = gt[gt_cols].dropna(subset=["rt_actual"])

    # Merge
    pred_df["business_day"] = pd.to_datetime(pred_df["business_day"]).dt.normalize()
    pred_df["hour"] = pred_df["hour"].astype(int)

    merged = pred_df.merge(gt, on=["business_day", "hour"], how="left", suffixes=("", "_gt"))

    # Use ground truth da_anchor if model's da_anchor is missing
    if "da_anchor" in merged.columns:
        merged["da_anchor"] = merged["da_anchor"].fillna(merged.get("da_anchor_gt"))
    else:
        merged["da_anchor"] = merged.get("da_anchor_gt")

    merged = merged.drop(columns=["da_anchor_gt"], errors="ignore")
    return merged


def _merge_ground_truth_v2(pred_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Merge V2 predictions with ground truth from raw data."""
    from models.deep_sgdf_delta.sgdfnet_bridge import (
        add_business_time_columns, RT_COL, DA_COL,
    )

    gt = raw_df.copy()
    gt = add_business_time_columns(gt)
    gt["rt_actual"] = pd.to_numeric(gt[RT_COL], errors="coerce")
    gt["da_anchor_gt"] = pd.to_numeric(gt[DA_COL], errors="coerce")
    gt["delta_target"] = gt["rt_actual"] - gt["da_anchor_gt"]
    gt["hour"] = gt["target_hour"].astype(int)
    gt["business_day"] = pd.to_datetime(gt["business_day"]).dt.normalize()

    gt_cols = ["business_day", "hour", "rt_actual", "delta_target", "da_anchor_gt"]
    gt = gt[gt_cols].dropna(subset=["rt_actual"])

    pred_df["business_day"] = pd.to_datetime(pred_df["business_day"]).dt.normalize()
    pred_df["hour"] = pred_df["hour"].astype(int)

    merged = pred_df.merge(gt, on=["business_day", "hour"], how="left", suffixes=("", "_gt"))

    if "da_anchor" in merged.columns:
        merged["da_anchor"] = merged["da_anchor"].fillna(merged.get("da_anchor_gt"))
    else:
        merged["da_anchor"] = merged.get("da_anchor_gt")

    merged = merged.drop(columns=["da_anchor_gt"], errors="ignore")
    return merged


# ── Evaluation (inline, mirrors evaluate_phase2_trendknight.py) ──────

def _run_evaluation(
    pred_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    sgdfnet_baseline: float,
) -> dict:
    """Run full evaluation and write all report files.

    Returns the full metrics dict.
    """
    from models.deep_sgdf_delta.metrics import (
        compute_full_metrics,
        compute_monthly_metrics,
        compute_period_mask,
        smape_floor50,
        classify_spike,
        classify_negative,
        delta_mae,
    )
    from models.deep_sgdf_delta.evaluate import PASS_THRESHOLD, SOFT_PASS_THRESHOLD

    # Full metrics
    full_metrics = compute_full_metrics(pred_df)

    # Monthly
    monthly_df = compute_monthly_metrics(pred_df)
    if not monthly_df.empty:
        full_metrics["monthly_avg_sMAPE_floor50"] = float(monthly_df["sMAPE_floor50"].mean())

    # Segment metrics
    valid = pred_df.dropna(subset=["rt_actual", "rt_pred"]).copy()
    hours = valid["hour"].to_numpy(dtype=int) if "hour" in valid.columns else np.array([])
    yt = valid["rt_actual"].to_numpy(dtype=float) if not valid.empty else np.array([])
    yp = valid["rt_pred"].to_numpy(dtype=float) if not valid.empty else np.array([])

    has_delta = "delta_target" in valid.columns and "delta_pred" in valid.columns
    if has_delta and not valid.empty:
        dt_arr = valid["delta_target"].to_numpy(dtype=float)
        dp_arr = valid["delta_pred"].to_numpy(dtype=float)

    seg_rows = []
    for period in ("1_8", "9_16", "17_24"):
        if len(hours) == 0:
            break
        mask = compute_period_mask(hours, period)
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"segment": period, "count": n, "sMAPE_floor50": smape_floor50(yt[mask], yp[mask])}
        if has_delta:
            row["delta_mae"] = delta_mae(dt_arr[mask], dp_arr[mask])
        seg_rows.append(row)
    segment_df = pd.DataFrame(seg_rows)

    # Bucket metrics
    bucket_rows = []
    if len(yt) > 0:
        spike_mask = classify_spike(yt)
        neg_mask = classify_negative(yt)
        normal_mask = ~spike_mask & ~neg_mask
        for label, mask in [("normal", normal_mask), ("high_price", spike_mask), ("negative", neg_mask)]:
            n = int(mask.sum())
            if n == 0:
                continue
            row = {"bucket": label, "count": n, "sMAPE_floor50": smape_floor50(yt[mask], yp[mask])}
            if has_delta:
                row["delta_mae"] = delta_mae(dt_arr[mask], dp_arr[mask])
            bucket_rows.append(row)
    bucket_df = pd.DataFrame(bucket_rows)

    # Verdict
    overall = full_metrics.get("overall_sMAPE_floor50", float("nan"))
    if overall < PASS_THRESHOLD:
        verdict = "PASS"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} < {PASS_THRESHOLD}"
    elif overall <= SOFT_PASS_THRESHOLD:
        verdict = "SOFT_PASS"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} <= {SOFT_PASS_THRESHOLD}"
    elif overall < sgdfnet_baseline:
        verdict = "BASELINE_PASS"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} < SGDFNet baseline {sgdfnet_baseline:.4f}"
    else:
        verdict = "NO-GO"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} >= SGDFNet baseline {sgdfnet_baseline:.4f}"

    full_metrics["verdict"] = verdict
    full_metrics["verdict_detail"] = verdict_detail

    # Write outputs
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, ensure_ascii=False, indent=2)

    if not monthly_df.empty:
        monthly_df.to_csv(output_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    if not segment_df.empty:
        segment_df.to_csv(output_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    if not bucket_df.empty:
        bucket_df.to_csv(output_dir / "bucket_metrics.csv", index=False, encoding="utf-8-sig")

    # go_nogo.md
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

    if not segment_df.empty:
        lines.extend(["## Segment Metrics", "", "| Segment | Count | sMAPE_floor50 |", "|---------|-------|---------------|"])
        for _, row in segment_df.iterrows():
            lines.append(f"| {row['segment']} | {row['count']} | {row['sMAPE_floor50']:.4f} |")
        lines.append("")

    if not bucket_df.empty:
        lines.extend(["## Bucket Metrics", "", "| Bucket | Count | sMAPE_floor50 |", "|--------|-------|---------------|"])
        for _, row in bucket_df.iterrows():
            lines.append(f"| {row['bucket']} | {row['count']} | {row['sMAPE_floor50']:.4f} |")
        lines.append("")

    if not monthly_df.empty:
        lines.extend(["## Monthly Breakdown", "", "| Month | sMAPE_floor50 | Count |", "|-------|---------------|-------|"])
        for _, row in monthly_df.iterrows():
            lines.append(f"| {row['month']} | {row['sMAPE_floor50']:.4f} | {row['count']} |")
        lines.append("")

    lines.extend([
        "## Thresholds", "",
        f"- PASS: overall sMAPE_floor50 < {PASS_THRESHOLD}",
        f"- SOFT PASS: overall sMAPE_floor50 <= {SOFT_PASS_THRESHOLD}",
        f"- BASELINE PASS: better than SGDFNet corrected baseline {sgdfnet_baseline:.4f}",
        f"- NO-GO: worse than SGDFNet baseline or leakage detected", "",
    ])
    (output_dir / "go_nogo.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"Evaluation complete: {verdict} — {verdict_detail}")
    return full_metrics


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_total_start = time.time()

    # Resolve profile
    profile_name = args.profile
    profile = PROFILES[profile_name]
    logger.info(f"Profile: {profile_name} — {profile['description']}")
    logger.info(f"Version: {profile['version']}, Backbone: {profile['backbone']}, Blend: {profile['blend']}")

    # Output directory
    run_id = f"backtest_{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = PROJECT_ROOT / "reports" / "local" / "phase2" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Enumerate months
    months = _enumerate_months(args.start_date, args.end_date)
    if args.fast_dev_run:
        months = months[:2]  # limit to 2 months
    logger.info(f"Backtest period: {args.start_date} to {args.end_date} ({len(months)} months)")

    # Load data
    data_path = _resolve_data_path(args)
    raw_df = _load_raw_data(data_path, args.sgdfnet_root)

    # Monthly backtest loop
    all_predictions: list[pd.DataFrame] = []
    runtime_records: list[dict] = []

    for i, (month_start, month_end) in enumerate(months):
        month_label = month_start.strftime("%Y-%m")
        logger.info(f"{'='*60}")
        logger.info(f"Month {i+1}/{len(months)}: {month_label} ({month_start.date()} to {month_end.date()})")
        logger.info(f"{'='*60}")

        t_month_start = time.time()
        record = {
            "month": month_label,
            "month_start": str(month_start.date()),
            "month_end": str(month_end.date()),
            "status": "pending",
            "rows_predicted": 0,
            "elapsed_seconds": 0.0,
            "error": None,
        }

        try:
            if profile["version"] == "v1":
                month_preds = _backtest_month_v1(
                    raw_df, profile, month_start, month_end,
                    args.device, args.amp, args.fast_dev_run, val_days=30,
                )
            else:
                month_preds = _backtest_month_v2(
                    raw_df, profile, month_start, month_end,
                    args.device, args.amp, args.fast_dev_run, val_days=30,
                )

            if month_preds.empty:
                record["status"] = "empty"
                logger.warning(f"Month {month_label}: no predictions generated")
            else:
                record["status"] = "success"
                record["rows_predicted"] = len(month_preds)
                all_predictions.append(month_preds)
                logger.info(f"Month {month_label}: {len(month_preds)} predictions collected")

        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)
            logger.error(f"Month {month_label} failed: {e}")
            logger.debug(traceback.format_exc())

        elapsed = time.time() - t_month_start
        record["elapsed_seconds"] = round(elapsed, 1)
        runtime_records.append(record)

        logger.info(f"Month {month_label} completed in {elapsed:.1f}s — status: {record['status']}")

    # Combine all predictions
    if not all_predictions:
        logger.error("No predictions collected from any month. Cannot run evaluation.")
        # Still write runtime report
        with open(output_dir / "runtime_report.json", "w", encoding="utf-8") as f:
            json.dump({"months": runtime_records, "total_elapsed_seconds": round(time.time() - t_total_start, 1)},
                      f, ensure_ascii=False, indent=2)
        print("\nBacktest failed: no predictions were generated.")
        print(f"Runtime report: {output_dir / 'runtime_report.json'}")
        return

    combined_preds = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"Total combined predictions: {len(combined_preds)} rows across {combined_preds['business_day'].nunique()} days")

    # Save predictions
    combined_preds.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Predictions saved to {output_dir / 'predictions.csv'}")

    # Run evaluation
    logger.info("Running full evaluation...")
    sgdfnet_baseline = 16.5902
    full_metrics = _run_evaluation(combined_preds, output_dir, run_id, sgdfnet_baseline)

    # Write runtime report
    total_elapsed = time.time() - t_total_start
    runtime_report = {
        "run_id": run_id,
        "profile": profile_name,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "total_months": len(months),
        "successful_months": sum(1 for r in runtime_records if r["status"] == "success"),
        "failed_months": sum(1 for r in runtime_records if r["status"] == "error"),
        "empty_months": sum(1 for r in runtime_records if r["status"] == "empty"),
        "total_predictions": len(combined_preds),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "fast_dev_run": args.fast_dev_run,
        "device": args.device,
        "amp": args.amp,
        "verdict": full_metrics.get("verdict"),
        "overall_sMAPE_floor50": full_metrics.get("overall_sMAPE_floor50"),
        "months": runtime_records,
    }
    with open(output_dir / "runtime_report.json", "w", encoding="utf-8") as f:
        json.dump(runtime_report, f, ensure_ascii=False, indent=2)

    # Print summary
    overall = full_metrics.get("overall_sMAPE_floor50", float("nan"))
    verdict = full_metrics.get("verdict", "N/A")
    monthly_avg = full_metrics.get("monthly_avg_sMAPE_floor50", float("nan"))

    print()
    print("=" * 60)
    print(f"  Phase 2 Monthly Backtest Complete")
    print(f"  Profile:          {profile_name}")
    print(f"  Period:           {args.start_date} to {args.end_date}")
    print(f"  Months:           {len(months)} ({runtime_report['successful_months']} ok, "
          f"{runtime_report['failed_months']} failed)")
    print(f"  Total predictions: {len(combined_preds)}")
    print(f"  Overall sMAPE:     {overall:.4f}")
    print(f"  Monthly avg:       {monthly_avg:.4f}")
    print(f"  Verdict:           {verdict}")
    print(f"  Elapsed:           {total_elapsed:.1f}s")
    print(f"  Output:            {output_dir}")
    print("=" * 60)
    print()
    print("Artifacts:")
    for fname in ["predictions.csv", "metrics_summary.json", "monthly_metrics.csv",
                   "segment_metrics.csv", "bucket_metrics.csv", "go_nogo.md", "runtime_report.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            print(f"  {fpath}")


if __name__ == "__main__":
    main()
