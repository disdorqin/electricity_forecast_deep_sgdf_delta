#!/usr/bin/env python
"""Build a leaderboard comparing different baseline methods for realtime price forecasting.

Evaluates multiple baselines on the same data/month and produces a comparison
leaderboard with per-baseline metrics, prediction CSVs, and a formatted report.

Baselines:
  1. DA anchor only          — rt_pred = da_anchor (= forecast_price)
  2. SGDFNet prediction      — rt_pred = sgdfnet_pred (fallback: da_anchor)
  3. SGDFNet + rolling bias  — rt_pred = sgdfnet_pred + rolling_mean(residual, 7d)
  4. Deep model (TrendKnightRT) — if --model-dir provided
  5. Teacher models          — rt916_pred, timemixer_pred, timesfm_pred (if columns exist)

Usage:
    python scripts/build_realtime_baseline_leaderboard.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-month 2026-02 \
        --out reports/local/deep_final/leaderboard_2026_02

    # With deep model:
    python scripts/build_realtime_baseline_leaderboard.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-month 2026-02 \
        --model-dir artifacts/trendknight_rt/champion \
        --out reports/local/deep_final/leaderboard_2026_02
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
logger = logging.getLogger("build_realtime_baseline_leaderboard")


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build baseline leaderboard for realtime price forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to hourly CSV data file",
    )
    parser.add_argument(
        "--target-month", type=str, required=True,
        help="Target month to evaluate (YYYY-MM)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to trained TrendKnightRT model directory (optional)",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output directory for leaderboard files",
    )
    parser.add_argument(
        "--spike-threshold", type=float, default=500.0,
        help="Price threshold for spike classification (default: 500)",
    )
    parser.add_argument(
        "--rolling-window", type=int, default=7,
        help="Rolling window in days for bias correction (default: 7)",
    )
    return parser.parse_args()


# -- Data loading -------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load hourly CSV data with GBK encoding fallback."""
    path = Path(data_path)
    if not path.exists():
        # Try relative to project root
        alt = PROJECT_ROOT / data_path
        if alt.exists():
            path = alt
        else:
            # Try sibling project
            alt2 = PROJECT_ROOT.parent / data_path
            if alt2.exists():
                path = alt2
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading data from %s", path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError):
            logger.info("utf-8-sig failed, retrying with gbk encoding")
            df = pd.read_csv(path, encoding="gbk")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info("Data loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with standard column names and business time columns."""
    df = df.copy()

    # Chinese column mapping
    cn_map = {
        "时刻": "ds",
        "日前电价": "da_anchor",
        "实时电价": "rt_actual",
        "forecast_price": "da_anchor",
        "rt_price": "rt_actual",
        "day_ahead_price": "da_anchor",
    }
    rename = {k: v for k, v in cn_map.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(columns=rename)
        logger.info("Renamed columns: %s", rename)

    # Parse timestamp
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError("Data must contain a timestamp column (ds or 时刻)")

    # Ensure required columns
    if "da_anchor" not in df.columns:
        raise ValueError("Data must contain da_anchor / forecast_price / 日前电价")
    if "rt_actual" not in df.columns:
        raise ValueError("Data must contain rt_actual / rt_price / 实时电价")

    # Ensure sgdfnet_pred exists
    if "sgdfnet_pred" not in df.columns:
        logger.info("sgdfnet_pred not found — using da_anchor as fallback")
        df["sgdfnet_pred"] = df["da_anchor"]
    else:
        mask = df["sgdfnet_pred"].isna()
        if mask.any():
            df.loc[mask, "sgdfnet_pred"] = df.loc[mask, "da_anchor"]

    # Add business time columns
    df = add_business_time_columns(df, timestamp_col="ds")

    return df


def filter_target_month(df: pd.DataFrame, target_month: str) -> pd.DataFrame:
    """Filter DataFrame to the target month based on business_day."""
    target_period = pd.Period(target_month, freq="M")
    bd_month = pd.to_datetime(df["business_day"]).dt.to_period("M")
    mask = bd_month == target_period
    filtered = df[mask].copy().reset_index(drop=True)
    logger.info(
        "Filtered to %s: %d rows (from %d total)",
        target_month, len(filtered), len(df),
    )
    return filtered


# -- Baseline builders --------------------------------------------------------

def build_da_anchor_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline 1: rt_pred = da_anchor (forecast_price)."""
    result = df[["business_day", "hour_business", "ds", "rt_actual", "da_anchor"]].copy()
    result["rt_pred"] = result["da_anchor"]
    result["delta_target"] = result["rt_actual"] - result["da_anchor"]
    result["delta_pred"] = result["rt_pred"] - result["da_anchor"]
    result["baseline"] = "da_anchor"
    return result


def build_sgdfnet_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Baseline 2: rt_pred = sgdfnet_pred.

    Returns (result_df, is_approximated).
    If sgdfnet_pred was originally missing (all equal to da_anchor), mark as APPROXIMATED.
    """
    # Check if sgdfnet_pred is actually different from da_anchor
    has_real_sgdfnet = not np.allclose(
        df["sgdfnet_pred"].values,
        df["da_anchor"].values,
        equal_nan=True,
    )

    result = df[["business_day", "hour_business", "ds", "rt_actual", "da_anchor"]].copy()
    result["rt_pred"] = df["sgdfnet_pred"].values
    result["delta_target"] = result["rt_actual"] - result["da_anchor"]
    result["delta_pred"] = result["rt_pred"] - result["da_anchor"]
    result["baseline"] = "sgdfnet_pred"

    is_approximated = not has_real_sgdfnet
    if is_approximated:
        logger.warning("SGDFNet predictions not available — using da_anchor as APPROXIMATED fallback")

    return result, is_approximated


def build_sgdfnet_rolling_bias_baseline(
    df: pd.DataFrame,
    rolling_window: int = 7,
) -> tuple[pd.DataFrame, bool]:
    """Baseline 3: rt_pred = sgdfnet_pred + rolling_mean(sgdfnet_residual, window).

    The rolling bias is computed from historical data BEFORE the target month
    to avoid lookahead.

    Returns (result_df, is_approximated).
    """
    df = df.copy()

    # Compute sgdfnet residual
    df["sgdfnet_residual"] = df["rt_actual"] - df["sgdfnet_pred"]

    # Check if sgdfnet_pred is real
    has_real_sgdfnet = not np.allclose(
        df["sgdfnet_pred"].values,
        df["da_anchor"].values,
        equal_nan=True,
    )

    # Compute rolling mean of residual (shifted to avoid lookahead)
    # Use a rolling window on the full data, then shift by 1 day
    df = df.sort_values("ds").reset_index(drop=True)
    df["rolling_bias"] = (
        df["sgdfnet_residual"]
        .rolling(window=rolling_window * 24, min_periods=1)
        .mean()
        .shift(24)  # shift by 1 day to avoid lookahead
    )
    df["rolling_bias"] = df["rolling_bias"].fillna(0.0)

    result = df[["business_day", "hour_business", "ds", "rt_actual", "da_anchor"]].copy()
    result["rt_pred"] = df["sgdfnet_pred"].values + df["rolling_bias"].values
    result["delta_target"] = result["rt_actual"] - result["da_anchor"]
    result["delta_pred"] = result["rt_pred"] - result["da_anchor"]
    result["baseline"] = "sgdfnet_rolling_bias"

    is_approximated = not has_real_sgdfnet
    if is_approximated:
        logger.warning("SGDFNet+rolling: sgdfnet_pred not real — result is APPROXIMATED")

    return result, is_approximated


def build_teacher_baseline(
    df: pd.DataFrame,
    teacher_col: str,
) -> pd.DataFrame | None:
    """Build baseline for a teacher model column.

    Returns None if the column doesn't exist.
    """
    if teacher_col not in df.columns:
        return None

    # Check if column has non-null, non-zero data
    col_data = df[teacher_col]
    if col_data.isna().all() or (col_data == 0).all():
        return None

    result = df[["business_day", "hour_business", "ds", "rt_actual", "da_anchor"]].copy()
    result["rt_pred"] = col_data.values
    # Fill NaN predictions with da_anchor
    mask = result["rt_pred"].isna()
    if mask.any():
        result.loc[mask, "rt_pred"] = result.loc[mask, "da_anchor"]
    result["delta_target"] = result["rt_actual"] - result["da_anchor"]
    result["delta_pred"] = result["rt_pred"] - result["da_anchor"]
    result["baseline"] = teacher_col
    return result


def build_deep_model_baseline(
    df: pd.DataFrame,
    model_dir: Path,
    target_month: str,
) -> pd.DataFrame | None:
    """Baseline 4: TrendKnightRT deep model predictions.

    Loads the trained model and runs prediction on the target month.
    Returns None if model loading fails.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from models.deep_sgdf_delta.trendknight_rt import (
            TrendKnightRTConfig,
            build_trendknight_rt,
        )
        from models.deep_sgdf_delta.realtime_dataset_final import (
            build_training_datasets_final,
            collate_fn_final,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading deep model from %s on %s", model_dir, device)

        # Load checkpoint
        ckpt_path = model_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.error("best_model.pt not found in %s", model_dir)
            return None

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Load config
        config_path = model_dir / "config.yaml"
        config_dict = {}
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f) or {}
            config_dict = full_config.get("model", {})

        # Build model
        model_config = TrendKnightRTConfig(
            input_dim=config_dict.get("input_dim", 40),
            hidden_dim=config_dict.get("hidden_dim", 64),
            num_layers=config_dict.get("num_layers", 2),
            dropout=config_dict.get("dropout", 0.1),
            backbone=config_dict.get("backbone", "tcn"),
            tcn_kernel_size=config_dict.get("tcn_kernel_size", 3),
            tcn_dilation_base=config_dict.get("tcn_dilation_base", 2),
            transformer_nhead=config_dict.get("transformer_nhead", 4),
            transformer_dim_ff=config_dict.get("transformer_dim_ff", 128),
            use_sgdfnet_residual_head=config_dict.get("use_sgdfnet_residual_head", True),
            use_delta_head=config_dict.get("use_delta_head", True),
            use_confidence_head=config_dict.get("use_confidence_head", True),
            use_period_bias=config_dict.get("use_period_bias", True),
            fusion_mode=config_dict.get("fusion_mode", "C"),
            hour_embed_dim=config_dict.get("hour_embed_dim", 8),
            segment_embed_dim=config_dict.get("segment_embed_dim", 8),
            multiscale=config_dict.get("multiscale", True),
            teacher_input_dim=config_dict.get("teacher_input_dim", 0),
        )

        model = build_trendknight_rt(model_config).to(device)

        # Load state dict
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        # Build test dataset for the target month
        # We need the full data (not just target month) for context
        train_ds, val_ds, test_ds, manifest = build_training_datasets_final(
            df, target_month=target_month, val_days=30, train_min_days=90,
        )

        if test_ds.n_days == 0:
            logger.warning("No test data for target month %s", target_month)
            return None

        test_loader = DataLoader(
            test_ds, batch_size=128, shuffle=False,
            collate_fn=collate_fn_final, drop_last=False, num_workers=0,
        )

        # Run prediction
        all_rows: list[dict] = []
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features_24h"].to(device)
                segment_id = batch["segment_id"].to(device)
                da_anchor = batch["da_anchor_24"].to(device)
                sgdfnet_pred = batch["sgdfnet_pred_24"].to(device)
                hour_ids = batch["hour_ids"].to(device)
                mask = batch["mask_24"].to(device)

                out = model(features, segment_id, da_anchor, sgdfnet_pred, hour_ids)
                rt_pred = out["trend_rt_pred_24"]
                delta_pred = out["delta_pred_24"]

                rt_true = da_anchor + batch["delta_target_24"].to(device)

                B = features.size(0)
                business_days_batch = batch.get("business_days", None)

                for i in range(B):
                    for h in range(24):
                        if mask[i, h].item() == 0:
                            continue
                        hour = h + 1
                        row = {
                            "rt_actual": float(rt_true[i, h].item()),
                            "rt_pred": float(rt_pred[i, h].item()),
                            "da_anchor": float(da_anchor[i, h].item()),
                            "delta_target": float(batch["delta_target_24"][i, h].item()),
                            "delta_pred": float(delta_pred[i, h].item()),
                            "hour_business": hour,
                        }
                        all_rows.append(row)

        if not all_rows:
            logger.warning("No predictions generated by deep model")
            return None

        result = pd.DataFrame(all_rows)

        # Reconstruct business_day from test dataset info
        # The test dataset provides business days in order
        test_bdays = []
        for day_idx in range(test_ds.n_days):
            sample = test_ds[day_idx]
            bd = sample.get("business_day", None)
            if bd is not None:
                if isinstance(bd, torch.Tensor):
                    bd = pd.Timestamp(bd.item() if bd.dim() == 0 else bd[0].item())
                test_bdays.append(pd.Timestamp(bd))
            else:
                test_bdays.append(pd.NaT)

        # Assign business days to rows
        rows_per_day = 24
        bday_list = []
        for i, bd in enumerate(test_bdays):
            for h in range(rows_per_day):
                bday_list.append(bd)

        if len(bday_list) >= len(result):
            result["business_day"] = bday_list[:len(result)]
        else:
            # Fallback: try to reconstruct from manifest
            logger.warning("Could not align business days — using sequential assignment")
            result["business_day"] = pd.NaT

        result["ds"] = result["business_day"]
        result["baseline"] = "trendknight_rt"

        logger.info("Deep model baseline: %d rows", len(result))
        return result

    except Exception as e:
        logger.error("Failed to build deep model baseline: %s", e)
        import traceback
        traceback.print_exc()
        return None


# -- Metrics computation ------------------------------------------------------

def compute_baseline_metrics(
    pred_df: pd.DataFrame,
    spike_threshold: float = 500.0,
) -> dict:
    """Compute comprehensive metrics for a baseline prediction DataFrame."""
    df = pred_df.copy()

    # Ensure required columns
    if "hour" not in df.columns:
        df["hour"] = df["hour_business"].astype(int)

    # Ensure delta columns exist
    if "delta_target" not in df.columns:
        df["delta_target"] = df["rt_actual"] - df["da_anchor"]
    if "delta_pred" not in df.columns:
        df["delta_pred"] = df["rt_pred"] - df["da_anchor"]

    # Drop NaN rows
    valid = df.dropna(subset=["rt_actual", "rt_pred"]).copy()
    if valid.empty:
        return {"overall_sMAPE_floor50": float("nan"), "rows_total": 0}

    hours = valid["hour"].to_numpy(dtype=int)
    yt = valid["rt_actual"].to_numpy(dtype=float)
    yp = valid["rt_pred"].to_numpy(dtype=float)

    metrics: dict = {}
    metrics["overall_sMAPE_floor50"] = smape_floor50(yt, yp)
    metrics["rows_total"] = len(valid)

    # Period metrics
    for period in ("1_8", "9_16", "17_24"):
        mask = compute_period_mask(hours, period)
        if mask.sum() > 0:
            metrics[f"{period}_sMAPE_floor50"] = smape_floor50(yt[mask], yp[mask])
        else:
            metrics[f"{period}_sMAPE_floor50"] = float("nan")

    # Bucket metrics
    spike_mask = np.abs(yt) > spike_threshold
    neg_mask = yt < 0.0
    normal_mask = ~spike_mask & ~neg_mask

    for bucket_name, mask in [("normal", normal_mask), ("negative", neg_mask), ("spike", spike_mask)]:
        if mask.sum() > 0:
            metrics[f"{bucket_name}_sMAPE_floor50"] = smape_floor50(yt[mask], yp[mask])
            metrics[f"{bucket_name}_count"] = int(mask.sum())
        else:
            metrics[f"{bucket_name}_sMAPE_floor50"] = float("nan")
            metrics[f"{bucket_name}_count"] = 0

    # Delta MAE
    if "delta_target" in valid.columns and "delta_pred" in valid.columns:
        dt = valid["delta_target"].to_numpy(dtype=float)
        dp = valid["delta_pred"].to_numpy(dtype=float)
        valid_delta = ~(np.isnan(dt) | np.isnan(dp))
        if valid_delta.sum() > 0:
            metrics["delta_mae"] = float(np.mean(np.abs(dp[valid_delta] - dt[valid_delta])))
        else:
            metrics["delta_mae"] = float("nan")
    else:
        metrics["delta_mae"] = float("nan")

    return metrics


# -- Leaderboard assembly -----------------------------------------------------

def build_leaderboard(
    baseline_results: list[dict],
) -> pd.DataFrame:
    """Build leaderboard DataFrame from baseline results.

    Each entry in baseline_results is a dict with:
      - baseline: str (name)
      - metrics: dict
      - status: str ("OK", "APPROXIMATED", "NOT_AVAILABLE", "FAILED")
      - runtime_s: float
      - notes: str
    """
    rows = []
    for entry in baseline_results:
        row = {"baseline": entry["baseline"], "status": entry["status"]}
        row["runtime_s"] = entry.get("runtime_s", 0.0)
        row["notes"] = entry.get("notes", "")

        m = entry.get("metrics", {})
        row["overall_sMAPE_floor50"] = m.get("overall_sMAPE_floor50", float("nan"))
        row["1_8_sMAPE_floor50"] = m.get("1_8_sMAPE_floor50", float("nan"))
        row["9_16_sMAPE_floor50"] = m.get("9_16_sMAPE_floor50", float("nan"))
        row["17_24_sMAPE_floor50"] = m.get("17_24_sMAPE_floor50", float("nan"))
        row["normal_sMAPE_floor50"] = m.get("normal_sMAPE_floor50", float("nan"))
        row["negative_sMAPE_floor50"] = m.get("negative_sMAPE_floor50", float("nan"))
        row["spike_sMAPE_floor50"] = m.get("spike_sMAPE_floor50", float("nan"))
        row["delta_mae"] = m.get("delta_mae", float("nan"))
        row["rows_total"] = m.get("rows_total", 0)
        row["negative_count"] = m.get("negative_count", 0)
        row["spike_count"] = m.get("spike_count", 0)

        rows.append(row)

    leaderboard = pd.DataFrame(rows)

    # Sort by overall sMAPE (ascending, NaN last)
    leaderboard = leaderboard.sort_values(
        "overall_sMAPE_floor50", ascending=True, na_position="last",
    ).reset_index(drop=True)

    return leaderboard


# -- Report generation --------------------------------------------------------

def write_leaderboard_report(
    out_dir: Path,
    leaderboard: pd.DataFrame,
    target_month: str,
    baseline_results: list[dict],
) -> None:
    """Write a formatted leaderboard report as Markdown."""
    lines = [
        f"# Baseline Leaderboard — {target_month}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Target month:** {target_month}",
        "",
        "## Summary",
        "",
        "| Baseline | Status | Overall sMAPE | 1_8 | 9_16 | 17_24 | Normal | Negative | Spike | Delta MAE | Runtime |",
        "|----------|--------|---------------|-----|------|-------|--------|----------|-------|-----------|---------|",
    ]

    for _, row in leaderboard.iterrows():
        def fmt(val):
            if pd.isna(val):
                return "N/A"
            return f"{val:.2f}"

        lines.append(
            f"| {row['baseline']} "
            f"| {row['status']} "
            f"| {fmt(row['overall_sMAPE_floor50'])} "
            f"| {fmt(row['1_8_sMAPE_floor50'])} "
            f"| {fmt(row['9_16_sMAPE_floor50'])} "
            f"| {fmt(row['17_24_sMAPE_floor50'])} "
            f"| {fmt(row['normal_sMAPE_floor50'])} "
            f"| {fmt(row['negative_sMAPE_floor50'])} "
            f"| {fmt(row['spike_sMAPE_floor50'])} "
            f"| {fmt(row['delta_mae'])} "
            f"| {row['runtime_s']:.1f}s "
            f"|"
        )

    lines.extend([
        "",
        "## Baseline Descriptions",
        "",
        "| Baseline | Description |",
        "|----------|-------------|",
        "| `da_anchor` | rt_pred = day-ahead forecast price (simplest baseline) |",
        "| `sgdfnet_pred` | rt_pred = SGDFNet prediction (if available) |",
        "| `sgdfnet_rolling_bias` | rt_pred = SGDFNet + 7-day rolling bias correction |",
        "| `trendknight_rt` | TrendKnightRT deep model prediction |",
        "| `rt916_pred` | RT916 teacher model prediction |",
        "| `timemixer_pred` | TimeMixer teacher model prediction |",
        "| `timesfm_pred` | TimesFM teacher model prediction |",
        "",
        "## Status Legend",
        "",
        "| Status | Meaning |",
        "|--------|---------|",
        "| OK | Baseline computed from actual data |",
        "| APPROXIMATED | Required input not available; used fallback |",
        "| NOT_AVAILABLE | Required columns/data not present |",
        "| FAILED | Computation failed with error |",
        "",
    ])

    # Add notes for special baselines
    notes_lines = []
    for entry in baseline_results:
        if entry.get("notes"):
            notes_lines.append(f"- **{entry['baseline']}**: {entry['notes']}")

    if notes_lines:
        lines.append("## Notes")
        lines.append("")
        lines.extend(notes_lines)
        lines.append("")

    # Verdict section
    lines.extend([
        "## Verdict Thresholds (for deep model)",
        "",
        "| Level | Condition |",
        "|-------|-----------|",
        "| STRONG | overall < 17 AND 9_16 < 22 |",
        "| PASS | overall < 15 |",
        "| ACCEPTABLE | overall < 20 |",
        "| NO-GO | overall >= 20 |",
        "",
    ])

    report_path = out_dir / "leaderboard_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Leaderboard report written to %s", report_path)


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "baseline_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Baseline Leaderboard Builder")
    logger.info("=" * 60)
    logger.info("  Data:         %s", args.data_path)
    logger.info("  Target month: %s", args.target_month)
    logger.info("  Model dir:    %s", args.model_dir or "(none)")
    logger.info("  Output:       %s", out_dir)
    logger.info("=" * 60)

    # Load and prepare data
    raw_df = load_data(args.data_path)
    full_df = prepare_data(raw_df)
    test_df = filter_target_month(full_df, args.target_month)

    if test_df.empty:
        logger.error("No data found for target month %s", args.target_month)
        sys.exit(1)

    logger.info("Test data: %d rows for %s", len(test_df), args.target_month)

    # Evaluate each baseline
    baseline_results: list[dict] = []

    # --- Baseline 1: DA anchor ---
    logger.info("")
    logger.info("--- Baseline 1: DA anchor ---")
    t0 = time.time()
    try:
        da_pred = build_da_anchor_baseline(test_df)
        da_metrics = compute_baseline_metrics(da_pred, args.spike_threshold)
        runtime = time.time() - t0
        baseline_results.append({
            "baseline": "da_anchor",
            "metrics": da_metrics,
            "status": "OK",
            "runtime_s": runtime,
            "notes": "Simplest baseline: rt_pred = forecast_price",
        })
        da_pred.to_csv(pred_dir / "da_anchor.csv", index=False, encoding="utf-8-sig")
        logger.info("DA anchor: overall_sMAPE=%.4f (%.1fs)", da_metrics["overall_sMAPE_floor50"], runtime)
    except Exception as e:
        logger.error("DA anchor failed: %s", e)
        baseline_results.append({
            "baseline": "da_anchor", "metrics": {}, "status": "FAILED",
            "runtime_s": time.time() - t0, "notes": str(e),
        })

    # --- Baseline 2: SGDFNet prediction ---
    logger.info("")
    logger.info("--- Baseline 2: SGDFNet prediction ---")
    t0 = time.time()
    try:
        sgdf_pred, is_approx = build_sgdfnet_baseline(test_df)
        sgdf_metrics = compute_baseline_metrics(sgdf_pred, args.spike_threshold)
        runtime = time.time() - t0
        status = "APPROXIMATED" if is_approx else "OK"
        notes = "SGDFNet not available, used da_anchor fallback" if is_approx else "SGDFNet predictions from bridge"
        baseline_results.append({
            "baseline": "sgdfnet_pred",
            "metrics": sgdf_metrics,
            "status": status,
            "runtime_s": runtime,
            "notes": notes,
        })
        sgdf_pred.to_csv(pred_dir / "sgdfnet_pred.csv", index=False, encoding="utf-8-sig")
        logger.info("SGDFNet: overall_sMAPE=%.4f [%s] (%.1fs)", sgdf_metrics["overall_sMAPE_floor50"], status, runtime)
    except Exception as e:
        logger.error("SGDFNet baseline failed: %s", e)
        baseline_results.append({
            "baseline": "sgdfnet_pred", "metrics": {}, "status": "FAILED",
            "runtime_s": time.time() - t0, "notes": str(e),
        })

    # --- Baseline 3: SGDFNet + rolling bias ---
    logger.info("")
    logger.info("--- Baseline 3: SGDFNet + rolling bias ---")
    t0 = time.time()
    try:
        roll_pred, is_approx = build_sgdfnet_rolling_bias_baseline(test_df, args.rolling_window)
        roll_metrics = compute_baseline_metrics(roll_pred, args.spike_threshold)
        runtime = time.time() - t0
        status = "APPROXIMATED" if is_approx else "OK"
        notes = f"Rolling {args.rolling_window}d bias correction (approximated)" if is_approx else f"Rolling {args.rolling_window}d bias correction"
        baseline_results.append({
            "baseline": "sgdfnet_rolling_bias",
            "metrics": roll_metrics,
            "status": status,
            "runtime_s": runtime,
            "notes": notes,
        })
        roll_pred.to_csv(pred_dir / "sgdfnet_rolling_bias.csv", index=False, encoding="utf-8-sig")
        logger.info("SGDFNet+rolling: overall_sMAPE=%.4f [%s] (%.1fs)", roll_metrics["overall_sMAPE_floor50"], status, runtime)
    except Exception as e:
        logger.error("SGDFNet+rolling baseline failed: %s", e)
        baseline_results.append({
            "baseline": "sgdfnet_rolling_bias", "metrics": {}, "status": "FAILED",
            "runtime_s": time.time() - t0, "notes": str(e),
        })

    # --- Baseline 4: Deep model (TrendKnightRT) ---
    if args.model_dir:
        logger.info("")
        logger.info("--- Baseline 4: TrendKnightRT deep model ---")
        t0 = time.time()
        try:
            model_dir = Path(args.model_dir)
            # For deep model we need the full data for context
            deep_pred = build_deep_model_baseline(full_df, model_dir, args.target_month)
            if deep_pred is not None and not deep_pred.empty:
                deep_metrics = compute_baseline_metrics(deep_pred, args.spike_threshold)
                runtime = time.time() - t0

                # Get model size
                model_size_mb = 0.0
                model_pt = model_dir / "best_model.pt"
                if model_pt.exists():
                    model_size_mb = model_pt.stat().st_size / (1024 * 1024)

                baseline_results.append({
                    "baseline": "trendknight_rt",
                    "metrics": deep_metrics,
                    "status": "OK",
                    "runtime_s": runtime,
                    "notes": f"Model size: {model_size_mb:.1f} MB",
                })
                deep_pred.to_csv(pred_dir / "trendknight_rt.csv", index=False, encoding="utf-8-sig")
                logger.info("Deep model: overall_sMAPE=%.4f (%.1fs)", deep_metrics["overall_sMAPE_floor50"], runtime)
            else:
                runtime = time.time() - t0
                baseline_results.append({
                    "baseline": "trendknight_rt",
                    "metrics": {},
                    "status": "NOT_AVAILABLE",
                    "runtime_s": runtime,
                    "notes": "Model prediction returned empty result",
                })
                logger.warning("Deep model returned no predictions")
        except Exception as e:
            logger.error("Deep model baseline failed: %s", e)
            baseline_results.append({
                "baseline": "trendknight_rt", "metrics": {}, "status": "FAILED",
                "runtime_s": time.time() - t0, "notes": str(e),
            })
    else:
        logger.info("No --model-dir provided — skipping deep model baseline")

    # --- Baseline 5: Teacher models ---
    teacher_cols = ["rt916_pred", "timemixer_pred", "timesfm_pred"]
    for teacher_col in teacher_cols:
        logger.info("")
        logger.info("--- Teacher: %s ---", teacher_col)
        t0 = time.time()
        try:
            teacher_pred = build_teacher_baseline(test_df, teacher_col)
            if teacher_pred is not None and not teacher_pred.empty:
                teacher_metrics = compute_baseline_metrics(teacher_pred, args.spike_threshold)
                runtime = time.time() - t0
                baseline_results.append({
                    "baseline": teacher_col,
                    "metrics": teacher_metrics,
                    "status": "OK",
                    "runtime_s": runtime,
                    "notes": f"Teacher model: {teacher_col}",
                })
                teacher_pred.to_csv(
                    pred_dir / f"{teacher_col}.csv", index=False, encoding="utf-8-sig",
                )
                logger.info(
                    "%s: overall_sMAPE=%.4f (%.1fs)",
                    teacher_col, teacher_metrics["overall_sMAPE_floor50"], runtime,
                )
            else:
                runtime = time.time() - t0
                baseline_results.append({
                    "baseline": teacher_col,
                    "metrics": {},
                    "status": "NOT_AVAILABLE",
                    "runtime_s": runtime,
                    "notes": f"Column '{teacher_col}' not found in data",
                })
                logger.info("%s: NOT_AVAILABLE (column not in data)", teacher_col)
        except Exception as e:
            logger.error("%s baseline failed: %s", teacher_col, e)
            baseline_results.append({
                "baseline": teacher_col, "metrics": {}, "status": "FAILED",
                "runtime_s": time.time() - t0, "notes": str(e),
            })

    # -- Build leaderboard ---
    logger.info("")
    logger.info("Building leaderboard ...")
    leaderboard = build_leaderboard(baseline_results)

    # Save leaderboard CSV
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    logger.info("Leaderboard CSV saved to %s", out_dir / "leaderboard.csv")

    # Save leaderboard report
    write_leaderboard_report(out_dir, leaderboard, args.target_month, baseline_results)

    # Save raw results as JSON
    results_json = []
    for entry in baseline_results:
        e = dict(entry)
        # Convert NaN to None for JSON
        for k, v in e.get("metrics", {}).items():
            if isinstance(v, float) and np.isnan(v):
                e["metrics"][k] = None
        results_json.append(e)

    with open(out_dir / "leaderboard_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    # -- Summary ---
    print()
    print("=" * 80)
    print("  Baseline Leaderboard — " + args.target_month)
    print("=" * 80)
    print()
    print(f"  {'Baseline':<25s} {'Status':<15s} {'Overall sMAPE':>14s} {'9_16':>8s} {'Runtime':>8s}")
    print("  " + "-" * 72)
    for _, row in leaderboard.iterrows():
        smape_str = f"{row['overall_sMAPE_floor50']:.2f}" if not pd.isna(row["overall_sMAPE_floor50"]) else "N/A"
        p916_str = f"{row['9_16_sMAPE_floor50']:.2f}" if not pd.isna(row["9_16_sMAPE_floor50"]) else "N/A"
        print(
            f"  {row['baseline']:<25s} {row['status']:<15s} {smape_str:>14s} {p916_str:>8s} {row['runtime_s']:>7.1f}s"
        )
    print()
    print(f"  Output: {out_dir}")
    print("  Files:")
    for fpath in sorted(out_dir.rglob("*")):
        if fpath.is_file():
            rel = fpath.relative_to(out_dir)
            print(f"    {rel}")
    print("=" * 80)


if __name__ == "__main__":
    main()
