#!/usr/bin/env python
"""Export trend prediction pack for mainline system integration.

Produces:
  reports/local/phase3/export/
    trend_prediction_pack.csv      Online prediction pack (no y_true)
    trend_prediction_manifest.json Metadata and summary statistics
    integration_notes.md           Human-readable integration notes

Usage:
    python scripts/export_trend_prediction_pack.py \
        --predictions reports/local/phase3/month_2026_03/champion_predictions.csv \
        --champion-model v2_day_tcn \
        --out-dir reports/local/phase3/export

    python scripts/export_trend_prediction_pack.py --help
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

from models.deep_sgdf_delta.integration_contract import (  # noqa: E402
    ONLINE_PACK_COLUMNS,
    EVAL_EXTRA_COLUMNS,
    validate_online_pack,
    validate_eval_pack,
    hour_to_period,
    strip_eval_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_trend_prediction_pack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trend prediction pack for mainline system integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  trend_prediction_pack.csv      Online-safe prediction pack (no y_true)
  trend_prediction_manifest.json Metadata and summary statistics
  integration_notes.md           Human-readable integration notes

The online pack contains only columns safe for production use.
Eval columns (y_true, residuals) are stripped automatically.
""",
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to champion_predictions.csv (from champion search or backtest)",
    )
    parser.add_argument(
        "--champion-model", type=str, default="unknown",
        help="Name of the champion model (for metadata)",
    )
    parser.add_argument(
        "--metrics-json", type=str, default=None,
        help="Path to champion_metrics_summary.json (optional, for manifest)",
    )
    parser.add_argument(
        "--blend-weights-json", type=str, default=None,
        help="Path to blend_weights.json (optional, for manifest)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="reports/local/phase3/export",
        help="Output directory (default: reports/local/phase3/export)",
    )
    parser.add_argument(
        "--include-eval", action="store_true",
        help="Also export eval pack (with y_true and residuals)",
    )
    return parser.parse_args()


def build_pack_from_predictions(
    pred_df: pd.DataFrame,
    champion_model: str,
) -> pd.DataFrame:
    """Build the online prediction pack from champion predictions.

    Maps column names from the champion search output format to the
    integration contract format.
    """
    df = pred_df.copy()

    # Map column names
    col_map = {}

    # business_day
    if "business_day" in df.columns:
        col_map["business_day"] = "business_day"

    # hour_business (from 'hour' or 'target_hour')
    for h_col in ("hour", "target_hour", "hour_business"):
        if h_col in df.columns:
            col_map[h_col] = "hour_business"
            break

    # ds (from 'ds' or construct from business_day + hour)
    if "ds" in df.columns:
        col_map["ds"] = "ds"
    elif "business_day" in df.columns:
        # Construct ds from business_day + hour_business
        h_col = None
        for c in ("hour", "target_hour", "hour_business"):
            if c in df.columns:
                h_col = c
                break
        if h_col:
            df["ds"] = pd.to_datetime(df["business_day"]) + (
                pd.to_timedelta(df[h_col].astype(int), unit="h")
            )
            col_map["ds"] = "ds"

    # trend_pred = blend_pred or rt_pred
    for tp_col in ("blend_pred", "rt_pred", "y_pred"):
        if tp_col in df.columns:
            col_map[tp_col] = "trend_pred"
            break

    # da_anchor
    for da_col in ("da_anchor", "da_price", "日前电价"):
        if da_col in df.columns:
            col_map[da_col] = "da_anchor"
            break

    # deep_rt_pred
    for drt_col in ("deep_rt_pred", "rt_pred"):
        if drt_col in df.columns:
            col_map[drt_col] = "deep_rt_pred"
            break

    # sgdfnet_pred
    for sgd_col in ("sgdfnet_pred", "sgdfnet_rt", "rt_hat"):
        if sgd_col in df.columns:
            col_map[sgd_col] = "sgdfnet_pred"
            break

    # blend_pred
    for bp_col in ("blend_pred", "rt_pred"):
        if bp_col in df.columns:
            col_map[bp_col] = "blend_pred"
            break

    # Apply column mapping
    rename_map = {src: dst for src, dst in col_map.items() if src != dst}
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    df["business_day"] = pd.to_datetime(df["business_day"]).dt.normalize()
    if "hour_business" not in df.columns:
        df["hour_business"] = 1  # fallback

    df["hour_business"] = df["hour_business"].astype(int)
    df["period"] = df["hour_business"].apply(hour_to_period)

    # Add metadata columns
    df["trend_model_name"] = champion_model
    if "trend_confidence" not in df.columns:
        df["trend_confidence"] = 1.0  # default full confidence

    # Flags
    if "normal_trend_flag" not in df.columns:
        rt_vals = df.get("trend_pred", df.get("rt_pred", pd.Series(dtype=float)))
        if len(rt_vals) > 0:
            df["normal_trend_flag"] = (
                (rt_vals.abs() <= 500) & (rt_vals >= 0)
            ).astype(int)
        else:
            df["normal_trend_flag"] = 1

    if "high_price_bucket_flag" not in df.columns:
        rt_vals = df.get("trend_pred", df.get("rt_pred", pd.Series(dtype=float)))
        df["high_price_bucket_flag"] = (rt_vals.abs() > 500).astype(int) if len(rt_vals) > 0 else 0

    if "negative_bucket_flag" not in df.columns:
        rt_vals = df.get("trend_pred", df.get("rt_pred", pd.Series(dtype=float)))
        df["negative_bucket_flag"] = (rt_vals < 0).astype(int) if len(rt_vals) > 0 else 0

    # Select online pack columns (only those that exist)
    available = [c for c in ONLINE_PACK_COLUMNS if c in df.columns]
    pack = df[available].copy()

    return pack


def write_manifest(
    out_dir: Path,
    pack_df: pd.DataFrame,
    champion_model: str,
    metrics: dict | None,
    blend_weights: dict | None,
) -> None:
    """Write trend_prediction_manifest.json."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "champion_model": champion_model,
        "n_rows": len(pack_df),
        "date_range": {
            "start": str(pack_df["business_day"].min().date()) if "business_day" in pack_df.columns else None,
            "end": str(pack_df["business_day"].max().date()) if "business_day" in pack_df.columns else None,
        },
        "columns": list(pack_df.columns),
        "column_types": {col: str(pack_df[col].dtype) for col in pack_df.columns},
        "missing_values": {col: int(pack_df[col].isna().sum()) for col in pack_df.columns},
    }

    if metrics:
        manifest["metrics"] = {
            k: v for k, v in metrics.items()
            if k not in ("monthly_smape",)  # Skip large nested dicts
        }

    if blend_weights:
        manifest["blend_weights"] = blend_weights

    with open(out_dir / "trend_prediction_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Manifest -> {out_dir / 'trend_prediction_manifest.json'}")


def write_integration_notes(
    out_dir: Path,
    pack_df: pd.DataFrame,
    champion_model: str,
    metrics: dict | None,
) -> None:
    """Write integration_notes.md."""
    lines = [
        "# Trend Prediction Integration Notes",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Champion Model:** {champion_model}",
        f"**Rows:** {len(pack_df)}",
        "",
        "## Column Descriptions",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| business_day | Business day (00:00 of calendar day D = business_day D-1) |",
        "| hour_business | Hour of business day (1-24) |",
        "| period | Segment: 1_8, 9_16, or 17_24 |",
        "| ds | Full timestamp (business_day + hour) |",
        "| trend_pred | Main trend prediction (final output) |",
        "| trend_model_name | Name of the trend model |",
        "| trend_confidence | Confidence score (0-1) |",
        "| deep_rt_pred | Deep model realtime prediction |",
        "| sgdfnet_pred | SGDFNet baseline prediction |",
        "| blend_pred | Blended prediction (deep + SGDFNet) |",
        "| da_anchor | Day-ahead anchor price |",
        "| normal_trend_flag | 1 if normal trend, 0 if outlier |",
        "| high_price_bucket_flag | 1 if |price| > 500 |",
        "| negative_bucket_flag | 1 if price < 0 |",
        "",
        "## Usage by Downstream Modules",
        "",
        "- **Spike module**: Uses `trend_pred` + `normal_trend_flag`. "
        "When `normal_trend_flag=0`, spike module should override.",
        "- **Negative price module**: Uses `trend_pred` + `negative_bucket_flag`. "
        "When `negative_bucket_flag=1`, negative module should override.",
        "- **Ledger fusion**: Uses `trend_pred` as the base trend, "
        "then applies spike/negative corrections.",
        "",
        "## Important Notes",
        "",
        "1. This pack does NOT contain `y_true` — it is safe for online use.",
        "2. `residual_for_spike_module` and `residual_for_negative_module` "
        "are only in the eval pack, not the online pack.",
        "3. `high_price_bucket_flag` and `negative_bucket_flag` in the online "
        "pack are based on prediction values, not actual values.",
        "4. All prices are in CNY/MWh.",
        "",
    ]

    if metrics:
        overall = metrics.get("overall_sMAPE_floor50", "N/A")
        monthly_avg = metrics.get("monthly_avg_sMAPE_floor50", "N/A")
        verdict = metrics.get("verdict", "N/A")
        lines.extend([
            "## Model Performance",
            "",
            f"- Overall sMAPE_floor50: {overall}",
            f"- Monthly avg sMAPE_floor50: {monthly_avg}",
            f"- Verdict: {verdict}",
            "",
        ])

    (out_dir / "integration_notes.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Integration notes -> {out_dir / 'integration_notes.md'}")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / pred_path
    if not pred_path.exists():
        logger.error(f"Predictions file not found: {pred_path}")
        sys.exit(1)

    pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(pred_df)} prediction rows from {pred_path}")

    # Build online pack
    pack_df = build_pack_from_predictions(pred_df, args.champion_model)
    logger.info(f"Online pack: {len(pack_df)} rows, {len(pack_df.columns)} columns")

    # Validate
    is_valid, errors = validate_online_pack(pack_df)
    if not is_valid:
        logger.warning(f"Online pack validation errors: {errors}")
    else:
        logger.info("Online pack validation: PASSED")

    # Write online pack
    pack_df.to_csv(out_dir / "trend_prediction_pack.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Online pack -> {out_dir / 'trend_prediction_pack.csv'}")

    # Load optional metadata
    metrics = None
    if args.metrics_json:
        mp = Path(args.metrics_json)
        if not mp.is_absolute():
            mp = PROJECT_ROOT / mp
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
                metrics = json.load(f)

    blend_weights = None
    if args.blend_weights_json:
        bp = Path(args.blend_weights_json)
        if not bp.is_absolute():
            bp = PROJECT_ROOT / bp
        if bp.exists():
            with open(bp, "r", encoding="utf-8") as f:
                blend_weights = json.load(f)

    # Write manifest
    write_manifest(out_dir, pack_df, args.champion_model, metrics, blend_weights)

    # Write integration notes
    write_integration_notes(out_dir, pack_df, args.champion_model, metrics)

    # Optional: eval pack
    if args.include_eval and "y_true" in pred_df.columns:
        from models.deep_sgdf_delta.integration_contract import add_eval_columns
        eval_pack = pack_df.copy()
        eval_pack["y_true"] = pred_df["y_true"].values[:len(eval_pack)] if "y_true" in pred_df.columns else np.nan
        eval_pack = add_eval_columns(eval_pack)
        eval_pack.to_csv(out_dir / "trend_prediction_pack_eval.csv", index=False, encoding="utf-8-sig")
        logger.info(f"Eval pack -> {out_dir / 'trend_prediction_pack_eval.csv'}")

    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
