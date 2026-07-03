#!/usr/bin/env python
"""Prediction entry point for TrendKnightRT.

Loads a trained TrendKnightRT model and produces realtime price predictions
for one or more decision days.  Output is an online-safe CSV (no y_true).

Usage:
    # Single day
    python scripts/predict_realtime_deep_model.py \
        --model-dir artifacts/trendknight_rt/champion \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --decision-day 2026-02-01 \
        --out outputs/trendknight_rt_predictions.csv

    # Batch mode
    python scripts/predict_realtime_deep_model.py \
        --model-dir artifacts/trendknight_rt/champion \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --decision-days 2026-02-01,2026-02-02,2026-02-03 \
        --out outputs/trendknight_rt_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.realtime_dataset_final import (  # noqa: E402
    build_predict_dataset_final,
    collate_fn_final,
)
from models.deep_sgdf_delta.trendknight_rt import (  # noqa: E402
    TrendKnightRT,
    TrendKnightRTConfig,
    build_trendknight_rt,
)
from models.deep_sgdf_delta.realtime_feature_contract import (  # noqa: E402
    FEATURE_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict_realtime_deep_model")

MODEL_VERSION = "trendknight_rt_v1"


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrendKnightRT realtime price prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Directory containing best_model.pt and optional model_config.json",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to the hourly CSV data file",
    )
    parser.add_argument(
        "--decision-day", type=str, default=None,
        help="Single decision day to predict (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--decision-days", type=str, default=None,
        help="Comma-separated list of decision days for batch prediction",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for DataLoader",
    )
    return parser.parse_args()


# -- Data loading -------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV data, trying utf-8-sig first then gbk."""
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
                raise FileNotFoundError(
                    f"Data file not found: {data_path}\n"
                    f"Tried: {Path(data_path).resolve()}, {alt}, {alt2}"
                )

    logger.info("Loading data from %s", path)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        logger.info("utf-8-sig failed, retrying with gbk encoding")
        df = pd.read_csv(path, encoding="gbk")

    logger.info("Data loaded: %d rows, columns: %s", len(df), list(df.columns))
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw column names to internal convention.

    Handles both Chinese (Shandong spot market CSV) and English column names.
    """
    rename_map = {}
    # Chinese → English
    _CN_MAP = {
        "时刻": "ds",
        "日前电价": "da_anchor",
        "实时电价": "rt_actual",
    }
    for cn, en in _CN_MAP.items():
        if cn in df.columns:
            rename_map[cn] = en
    # English aliases
    if "rt_price" in df.columns and "rt_actual" not in rename_map.values():
        rename_map["rt_price"] = "rt_actual"
    if "forecast_price" in df.columns and "da_anchor" not in rename_map.values():
        rename_map["forecast_price"] = "da_anchor"
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("Renamed %d columns: %s", len(rename_map), rename_map)
    # Ensure forecast_price and sgdfnet_pred exist (matching training script)
    if "forecast_price" not in df.columns and "da_anchor" in df.columns:
        df["forecast_price"] = df["da_anchor"]
    if "sgdfnet_pred" not in df.columns and "da_anchor" in df.columns:
        df["sgdfnet_pred"] = df["da_anchor"]
    return df


# -- Model loading ------------------------------------------------------------

def load_model(
    model_dir: Path,
    device: torch.device,
) -> tuple[TrendKnightRT, dict]:
    """Load TrendKnightRT model from checkpoint.

    Supports two checkpoint formats:
      1. Dict with 'model_state_dict' and 'model_config' (rich checkpoint)
      2. Plain state_dict (requires model_config.json alongside)

    Returns:
        (model, config_dict) -- model in eval mode on the given device
    """
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    # Load config
    config_dict = {}
    config_path = model_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        logger.info("Loaded model config from %s", config_path)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # Rich checkpoint may carry its own config
        if "model_config" in ckpt and not config_dict:
            config_dict = ckpt["model_config"]
            if isinstance(config_dict, dict):
                logger.info("Using model_config from checkpoint")
        feature_cols = ckpt.get("feature_cols")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Format from train_realtime_deep_model.py
        state_dict = ckpt["state_dict"]
        if "config" in ckpt and not config_dict:
            config_dict = ckpt["config"]
            if isinstance(config_dict, dict):
                logger.info("Using config from checkpoint")
        feature_cols = ckpt.get("feature_cols")
    else:
        state_dict = ckpt
        feature_cols = None

    # Also try config.yaml if config_dict is still empty
    if not config_dict:
        config_yaml_path = model_dir / "config.yaml"
        if config_yaml_path.exists():
            import yaml
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
            logger.info("Loaded model config from %s", config_yaml_path)

    # Build model config from dict
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
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded TrendKnightRT model (%d params, fusion=%s, backbone=%s)",
        n_params, model_config.fusion_mode, model_config.backbone,
    )

    return model, config_dict


# -- Prediction ---------------------------------------------------------------

@torch.no_grad()
def predict_day(
    model: TrendKnightRT,
    raw_df: pd.DataFrame,
    decision_day: str,
    device: torch.device,
    batch_size: int = 64,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run prediction for a single decision day.

    Returns a DataFrame with 24 rows (hours 1-24) containing model outputs.
    """
    target_day = pd.Timestamp(decision_day)

    # Build prediction dataset
    pred_ds, manifest = build_predict_dataset_final(
        raw_df, target_day=target_day, feature_cols=feature_cols,
    )
    logger.info(
        "Prediction dataset for %s: %d days, %d features",
        target_day.date(), pred_ds.n_days, pred_ds.input_dim,
    )

    if pred_ds.n_days == 0:
        logger.warning("No data found for decision day %s", decision_day)
        return pd.DataFrame()

    loader = DataLoader(
        pred_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_final,
    )

    all_rows: list[dict] = []
    business_days = pred_ds.business_days

    batch_offset = 0
    for batch in loader:
        features_24h = batch["features_24h"].to(device)        # [B, 24, F]
        segment_id = batch["segment_id"].to(device)             # [B]
        da_anchor_24 = batch["da_anchor_24"].to(device)         # [B, 24]
        sgdfnet_pred_24 = batch["sgdfnet_pred_24"].to(device)   # [B, 24]
        hour_ids = batch["hour_ids"].to(device)                 # [B, 24]
        mask_24 = batch["mask_24"]                              # [B, 24]
        period_24 = batch["period_24"]                          # [B, 24]

        out = model(
            features_24h=features_24h,
            segment_id=segment_id,
            da_anchor_24=da_anchor_24,
            sgdfnet_pred_24=sgdfnet_pred_24,
            hour_ids=hour_ids,
        )

        delta_pred_24 = out["delta_pred_24"].cpu()                # [B, 24]
        residual_24 = out["residual_to_sgdfnet_24"].cpu()         # [B, 24]
        trend_rt_24 = out["trend_rt_pred_24"].cpu()               # [B, 24]
        confidence_24 = out["confidence_24"].cpu()                 # [B, 24]

        da_anchor_cpu = batch["da_anchor_24"]                      # [B, 24]
        sgdfnet_cpu = batch["sgdfnet_pred_24"]                     # [B, 24]

        B = features_24h.size(0)
        for i in range(B):
            day_idx = batch_offset + i
            bd = business_days[day_idx] if day_idx < len(business_days) else pd.NaT

            for h in range(24):
                hour = h + 1  # 1-24
                period_labels = {0: "1_8", 1: "9_16", 2: "17_24"}
                period_val = period_labels.get(int(period_24[i, h].item()), "1_8")

                all_rows.append({
                    "business_day": bd,
                    "hour_business": hour,
                    "ds": bd + pd.Timedelta(hours=hour) if hour < 24 else bd + pd.Timedelta(days=1),
                    "model_name": MODEL_VERSION,
                    "rt_pred": float(trend_rt_24[i, h].item()),
                    "delta_pred": float(delta_pred_24[i, h].item()),
                    "da_anchor": float(da_anchor_cpu[i, h].item()),
                    "sgdfnet_pred": float(sgdfnet_cpu[i, h].item()),
                    "residual_to_sgdfnet": float(residual_24[i, h].item()),
                    "confidence": float(confidence_24[i, h].item()),
                    "period": period_val,
                    "prediction_mode": "realtime",
                    "feature_version": FEATURE_VERSION,
                    "model_version": MODEL_VERSION,
                })

        batch_offset += B

    result = pd.DataFrame(all_rows)

    # Fix ds for hour 24: should be next day 00:00
    mask_24h = result["hour_business"] == 24
    if mask_24h.any():
        result.loc[mask_24h, "ds"] = (
            pd.to_datetime(result.loc[mask_24h, "business_day"]) + pd.Timedelta(days=1)
        )

    return result


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve decision days
    decision_days: list[str] = []
    if args.decision_days:
        decision_days = [d.strip() for d in args.decision_days.split(",") if d.strip()]
    elif args.decision_day:
        decision_days = [args.decision_day.strip()]
    else:
        print("ERROR: Must specify --decision-day or --decision-days", file=sys.stderr)
        sys.exit(1)

    logger.info("Decision days: %s", decision_days)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load data
    raw_df = load_data(args.data_path)
    raw_df = rename_columns(raw_df)

    # Ensure ds column is datetime
    if "ds" in raw_df.columns:
        raw_df["ds"] = pd.to_datetime(raw_df["ds"])

    # Load model
    model_dir = Path(args.model_dir)
    model, config_dict = load_model(model_dir, device)

    # Load feature columns from manifest (to match training)
    feature_cols = None
    manifest_path = model_dir / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        feature_cols = manifest.get("feature_columns")
        if feature_cols:
            logger.info("Loaded %d feature columns from manifest", len(feature_cols))

    # Predict for each decision day
    all_predictions: list[pd.DataFrame] = []
    for day_str in decision_days:
        logger.info("Predicting for decision day: %s", day_str)
        try:
            day_pred = predict_day(model, raw_df, day_str, device, args.batch_size, feature_cols=feature_cols)
            if not day_pred.empty:
                all_predictions.append(day_pred)
                logger.info(
                    "  -> %d rows predicted for %s",
                    len(day_pred), day_str,
                )
            else:
                logger.warning("  -> No predictions for %s", day_str)
        except Exception as e:
            logger.error("  -> Failed for %s: %s", day_str, e)
            raise

    if not all_predictions:
        logger.error("No predictions generated for any decision day")
        sys.exit(1)

    # Concatenate all predictions
    output_df = pd.concat(all_predictions, ignore_index=True)

    # Ensure output columns are in the right order
    output_cols = [
        "business_day", "hour_business", "ds", "model_name",
        "rt_pred", "delta_pred", "da_anchor", "sgdfnet_pred",
        "residual_to_sgdfnet", "confidence", "period",
        "prediction_mode", "feature_version", "model_version",
    ]
    output_df = output_df[output_cols]

    # Format business_day as date string
    output_df["business_day"] = pd.to_datetime(output_df["business_day"]).dt.strftime("%Y-%m-%d")
    output_df["ds"] = pd.to_datetime(output_df["ds"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Summary
    n_days = output_df["business_day"].nunique()
    n_rows = len(output_df)
    mean_conf = output_df["confidence"].mean()

    print()
    print("=" * 60)
    print("  TrendKnightRT Prediction Complete")
    print("=" * 60)
    print(f"  Decision days:    {n_days}")
    print(f"  Total rows:       {n_rows} (expected {n_days * 24})")
    print(f"  Mean confidence:  {mean_conf:.4f}")
    print(f"  Output:           {out_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
