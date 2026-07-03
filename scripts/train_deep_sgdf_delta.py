#!/usr/bin/env python
"""Train DeepSGDFDelta / TrendKnight model.

Usage:
    python scripts/train_deep_sgdf_delta.py --config models/deep_sgdf_delta/config.yaml
    python scripts/train_deep_sgdf_delta.py --help
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.sgdfnet_bridge import lazy_import as _bridge_lazy  # noqa: E402

from models.deep_sgdf_delta.train import TrainConfig, train_model  # noqa: E402
from models.deep_sgdf_delta.dataset import build_training_datasets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_deep_sgdf_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DeepSGDFDelta / TrendKnight model for realtime price trend prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sgdfnet-root", type=str, default=None,
                        help="Path to SGDFNet project root (contains src/sgdfnet/)")
    parser.add_argument("--config", type=str, default="models/deep_sgdf_delta/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override data path from config")
    parser.add_argument("--start-day", type=str, default=None,
                        help="Override start day (YYYY-MM-DD)")
    parser.add_argument("--end-day", type=str, default=None,
                        help="Override end day (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--backbone", type=str, choices=["tcn", "gru"], default=None,
                        help="Override backbone type")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Override hidden dimension")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Override sequence window days (7~14)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None,
                        help="Override device")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="Enable AMP training")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run ID for output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        cfg = {}

    # Apply overrides
    data_path = args.data_path or cfg.get("data_path", "data/shandong_pmos_hourly.xlsx")
    start_day = args.start_day or cfg.get("start_day", "2026-01-01")
    end_day = args.end_day or cfg.get("end_day", "2026-05-11")

    # Resolve SGDFNet via bridge
    _bridge_lazy(args.sgdfnet_root)
    from models.deep_sgdf_delta.sgdfnet_bridge import load_dataset as _load_dataset

    # Build TrainConfig
    model_cfg = cfg.get("model", {})
    train_cfg_dict = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})

    train_config = TrainConfig(
        hidden_dim=args.hidden_dim or model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        backbone=args.backbone or model_cfg.get("backbone", "tcn"),
        tcn_kernel_size=model_cfg.get("tcn_kernel_size", 3),
        tcn_dilation_base=model_cfg.get("tcn_dilation_base", 2),
        segment_embed_dim=model_cfg.get("segment_embed_dim", 8),
        use_global_residual=model_cfg.get("use_global_residual", True),
        global_residual_weight=model_cfg.get("global_residual_weight", 0.3),
        amp_enabled=args.amp if args.amp is not None else model_cfg.get("amp_enabled", False),
        epochs=args.epochs or train_cfg_dict.get("epochs", 30),
        batch_size=args.batch_size or train_cfg_dict.get("batch_size", 256),
        learning_rate=train_cfg_dict.get("learning_rate", 1e-3),
        weight_decay=train_cfg_dict.get("weight_decay", 1e-4),
        early_stopping_patience=train_cfg_dict.get("early_stopping_patience", 5),
        device=args.device or train_cfg_dict.get("device", "auto"),
        window_days=args.window_days or train_cfg_dict.get("window_days", 7),
        seed=args.seed or train_cfg_dict.get("seed", 42),
        w_smape=loss_cfg.get("w_smape", 0.55),
        w_delta_mae=loss_cfg.get("w_delta_mae", 0.25),
        w_period=loss_cfg.get("w_period", 0.10),
        w_smooth=loss_cfg.get("w_smooth", 0.10),
        period_916_weight=loss_cfg.get("period_916_weight", 2.0),
        val_days=cfg.get("val_days", 30),
        train_min_rows=cfg.get("train_min_rows", 2160),
    )

    # Build FeatureConfig
    from models.deep_sgdf_delta.sgdfnet_bridge import FeatureConfig as _FeatureConfig
    feat_cfg_dict = cfg.get("feature_config", {})
    feature_config = _FeatureConfig(**feat_cfg_dict)

    # Load data — try project root first, then sibling project
    data_file = PROJECT_ROOT / data_path
    if not data_file.exists():
        sibling = PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / data_path
        if sibling.exists():
            data_file = sibling
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_file}")
    raw_df = _load_dataset(data_file)
    logger.info(f"Data loaded: {len(raw_df)} rows, {raw_df['时刻'].min()} to {raw_df['时刻'].max()}")

    # Output directory
    run_id = args.run_id or f"deep_sgdf_delta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_dir or cfg.get("output_root", "reports/local/deep_sgdf_delta"))
    output_dir = PROJECT_ROOT / output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train for each decision day (walk-forward)
    start = pd.Timestamp(start_day)
    end = pd.Timestamp(end_day)
    decision_days = pd.date_range(start=start - pd.Timedelta(days=1), end=end - pd.Timedelta(days=1), freq="D")

    logger.info(f"Training DeepSGDFDelta: {len(decision_days)} decision days, "
                f"backbone={train_config.backbone}, hidden={train_config.hidden_dim}, "
                f"window={train_config.window_days}d, epochs={train_config.epochs}")

    # For walk-forward, we train once on all available data and evaluate
    # Use the last decision day for training
    last_decision_day = decision_days[-1]

    result = train_model(
        raw_df, feature_config, train_config,
        decision_day=last_decision_day,
        output_dir=output_dir,
    )

    # Save training history
    import json
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(result["history"], f, ensure_ascii=False, indent=2)

    # Save run config
    run_meta = {
        "run_id": run_id,
        "start_day": start_day,
        "end_day": end_day,
        "decision_day": str(last_decision_day.date()),
        "train_config": {
            "hidden_dim": train_config.hidden_dim,
            "num_layers": train_config.num_layers,
            "backbone": train_config.backbone,
            "epochs": train_config.epochs,
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "window_days": train_config.window_days,
            "seed": train_config.seed,
        },
        "best_val_smape": result["best_val_smape"],
        "best_epoch": result["best_epoch"],
        "total_params": result["total_params"],
        "device": result["device"],
        "feature_count": len(result["feature_cols"]),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Training complete. Best val sMAPE: {result['best_val_smape']:.4f} at epoch {result['best_epoch']}")
    logger.info(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
