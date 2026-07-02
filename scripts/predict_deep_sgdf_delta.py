#!/usr/bin/env python
"""Predict with trained DeepSGDFDelta model.

Usage:
    python scripts/predict_deep_sgdf_delta.py --date 2026-03-01 --model-dir reports/local/deep_sgdf_delta/run_xxx
    python scripts/predict_deep_sgdf_delta.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ORIG_SGDFNET = Path(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\SGDFNet\src")
if _ORIG_SGDFNET.exists() and str(_ORIG_SGDFNET) not in sys.path:
    sys.path.insert(0, str(_ORIG_SGDFNET))

from sgdfnet.data_contract import FeatureConfig, load_dataset  # noqa: E402
from sgdfnet.protocol_b_cutoff import (  # noqa: E402
    _build_protocol_b_visible_frame,
    _build_inference_frame,
    load_protocol_b_cutoff_config,
)

from models.deep_sgdf_delta.dataset import build_predict_dataset  # noqa: E402
from models.deep_sgdf_delta.model import DeepSGDFDeltaConfig, build_model  # noqa: E402
from models.deep_sgdf_delta.predict import predict_delta, predict_with_blend, BlendMode  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("predict_deep_sgdf_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict realtime price delta with trained DeepSGDFDelta model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", type=str, required=True,
                        help="Target prediction date (YYYY-MM-DD)")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing trained model (best_model.pt + run_config.json)")
    parser.add_argument("--config", type=str, default="models/deep_sgdf_delta/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override data path")
    parser.add_argument("--blend-mode", type=str, choices=["deep_only", "sgdfnet_blend", "sgdfnet_residual"],
                        default=None, help="Override blend mode")
    parser.add_argument("--blend-weight", type=float, default=0.5,
                        help="SGDFNet weight for sgdfnet_blend mode")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for inference")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Load model config
    model_dir = Path(args.model_dir)
    run_config_path = model_dir / "run_config.json"
    if run_config_path.exists():
        with open(run_config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
    else:
        run_config = {}

    # Build model config
    model_cfg = cfg.get("model", {})
    train_cfg = run_config.get("train_config", {})
    feature_cols = run_config.get("feature_cols", None)

    model_config = DeepSGDFDeltaConfig(
        input_dim=train_cfg.get("input_dim", 40),  # Will be overridden by actual feature count
        hidden_dim=train_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 64)),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        backbone=model_cfg.get("backbone", "tcn"),
        tcn_kernel_size=model_cfg.get("tcn_kernel_size", 3),
        tcn_dilation_base=model_cfg.get("tcn_dilation_base", 2),
        segment_embed_dim=model_cfg.get("segment_embed_dim", 8),
        use_global_residual=model_cfg.get("use_global_residual", True),
        global_residual_weight=model_cfg.get("global_residual_weight", 0.3),
    )

    # Load data
    data_path = args.data_path or cfg.get("data_path", "data/shandong_pmos_hourly.xlsx")
    data_file = PROJECT_ROOT / data_path
    if not data_file.exists():
        data_file = Path(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp") / data_path

    raw_df = load_dataset(data_file)
    target_day = pd.Timestamp(args.date)

    # Build visible frame (cutoff-safe)
    decision_day = target_day - pd.Timedelta(days=1)
    decision_hour = cfg.get("decision_hour", 15)

    sgdfnet_cfg_path = cfg.get("sgdfnet_config_path", "SGDFNet/configs/cutoff_recovery_2026_diag_a_prune_actualside.yaml")
    sgdfnet_cfg_file = PROJECT_ROOT / sgdfnet_cfg_path
    if not sgdfnet_cfg_file.exists():
        sgdfnet_cfg_file = Path(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp") / sgdfnet_cfg_path

    visible_df = _build_protocol_b_visible_frame(raw_df, decision_day, decision_hour)

    # Build predict dataset
    feat_cfg_dict = cfg.get("feature_config", {})
    feature_config = FeatureConfig(**feat_cfg_dict)
    window_days = train_cfg.get("window_days", cfg.get("training", {}).get("window_days", 7))

    pred_ds, actual_feature_cols = build_predict_dataset(
        raw_df, feature_config,
        target_day=target_day,
        window_days=window_days,
        visible_frame=visible_df,
    )

    # Update model config with actual feature count
    model_config.input_dim = len(actual_feature_cols)

    # Load model
    model = build_model(model_config)
    model_path = model_dir / "best_model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = model.to(device)

    # Predict
    deep_pred = predict_delta(model, pred_ds, device, batch_size=args.batch_size)

    # Blend
    blend_mode: BlendMode = args.blend_mode or cfg.get("blend", {}).get("mode", "deep_only")
    final_pred = predict_with_blend(deep_pred, None, mode=blend_mode, blend_weight=args.blend_weight)

    # Add metadata columns
    target_rows = visible_df[
        (pd.to_datetime(visible_df["business_day"]).dt.normalize() == target_day.normalize())
    ].copy() if "business_day" in visible_df.columns else pd.DataFrame()

    # Output
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_pred.to_csv(output_dir / f"predictions_{args.date}.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Predictions saved to {output_dir / f'predictions_{args.date}.csv'}")


if __name__ == "__main__":
    main()
