#!/usr/bin/env python
"""Phase 2 unified training entry point for TrendKnight.

Supports all 7 profiles across V1 (per-hour) and V2 (day-level) architectures
with optional SGDFNet blending.  Walk-forward training: one model trained on
all data before the evaluation period, then saved for downstream prediction.

Profiles:
  v1_hourly_tcn           V1 per-hour model, TCN backbone
  v1_hourly_gru           V1 per-hour model, GRU backbone
  v2_day_tcn              V2 day-level model, TCN backbone
  v2_day_gru              V2 day-level model, GRU backbone
  v2_day_transformer_tiny V2 day-level model, Transformer-tiny backbone
  v2_residual_sgdfnet     V2 day-level, TCN, SGDFNet residual blend
  v2_blend_sgdfnet        V2 day-level, TCN, SGDFNet weighted blend

Usage:
    python scripts/train_phase2_trendknight.py --profile v2_day_tcn --start-date 2026-01-01 --end-date 2026-05-11
    python scripts/train_phase2_trendknight.py --help
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
import yaml

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_phase2_trendknight")


# ── Profile registry ─────────────────────────────────────────────────

PROFILES: dict[str, dict] = {
    "v1_hourly_tcn": {
        "version": "v1",
        "backbone": "tcn",
        "blend": "deep_only",
        "description": "V1 per-hour model, TCN backbone",
    },
    "v1_hourly_gru": {
        "version": "v1",
        "backbone": "gru",
        "blend": "deep_only",
        "description": "V1 per-hour model, GRU backbone",
    },
    "v2_day_tcn": {
        "version": "v2",
        "backbone": "tcn",
        "blend": "deep_only",
        "description": "V2 day-level model, TCN backbone",
    },
    "v2_day_gru": {
        "version": "v2",
        "backbone": "gru",
        "blend": "deep_only",
        "description": "V2 day-level model, GRU backbone",
    },
    "v2_day_transformer_tiny": {
        "version": "v2",
        "backbone": "transformer_tiny",
        "blend": "deep_only",
        "description": "V2 day-level model, Transformer-tiny backbone",
    },
    "v2_residual_sgdfnet": {
        "version": "v2",
        "backbone": "tcn",
        "blend": "sgdfnet_residual",
        "description": "V2 day-level, TCN, SGDFNet residual blend",
    },
    "v2_blend_sgdfnet": {
        "version": "v2",
        "backbone": "tcn",
        "blend": "sgdfnet_blend",
        "description": "V2 day-level, TCN, SGDFNet weighted blend",
    },
}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 unified training entry point for TrendKnight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  v1_hourly_tcn             V1 per-hour model, TCN backbone
  v1_hourly_gru             V1 per-hour model, GRU backbone
  v2_day_tcn                V2 day-level model, TCN backbone
  v2_day_gru                V2 day-level model, GRU backbone
  v2_day_transformer_tiny   V2 day-level model, Transformer-tiny backbone
  v2_residual_sgdfnet       V2 day-level, TCN, SGDFNet residual blend
  v2_blend_sgdfnet          V2 day-level, TCN, SGDFNet weighted blend

Examples:
  python scripts/train_phase2_trendknight.py --profile v2_day_tcn --start-date 2026-01-01 --end-date 2026-05-11
  python scripts/train_phase2_trendknight.py --profile v1_hourly_gru --epochs 20 --batch-size 128 --amp
  python scripts/train_phase2_trendknight.py --profile v2_day_transformer_tiny --fast-dev-run
""",
    )
    parser.add_argument("--profile", type=str, required=True, choices=list(PROFILES.keys()),
                        help="Model profile to train")
    parser.add_argument("--start-date", type=str, default="2026-01-01",
                        help="Start of evaluation period (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2026-05-11",
                        help="End of evaluation period (YYYY-MM-DD)")
    parser.add_argument("--train-start", type=str, default=None,
                        help="Override earliest training data date (default: use all available data before start-date)")
    parser.add_argument("--val-days", type=int, default=30,
                        help="Number of validation days before the decision day (default: 30)")
    parser.add_argument("--test-month", type=str, default=None,
                        help="If set, train for a specific test month (YYYY-MM) and use the first day of that month as decision day")
    parser.add_argument("--sgdfnet-root", type=str, default=None,
                        help="Path to SGDFNet project root (contains src/sgdfnet/)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to raw data file (default: data/shandong_pmos_hourly.xlsx)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: reports/local/phase2/{run_id})")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP (automatic mixed precision) training")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Quick sanity run with tiny data and few epochs")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Override sequence window days (V1 only, default: 7)")
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────

def _resolve_data_path(args: argparse.Namespace) -> Path:
    """Find the raw data file."""
    if args.data_path:
        p = Path(args.data_path)
        if p.exists():
            return p

    # Default locations
    candidates = [
        PROJECT_ROOT / "data" / "shandong_pmos_hourly.xlsx",
        Path(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp") / "data" / "shandong_pmos_hourly.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find data file. Use --data-path to specify its location.\n"
        f"Tried: {candidates}"
    )


def _load_raw_data(data_path: Path, sgdfnet_root: str | None = None) -> pd.DataFrame:
    """Load the raw hourly dataset via sgdfnet_bridge."""
    # Ensure sgdfnet is on path
    if sgdfnet_root:
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root
        find_sgdfnet_root(sgdfnet_root)

    from models.deep_sgdf_delta.sgdfnet_bridge import load_dataset
    logger.info(f"Loading data from {data_path}")
    raw_df = load_dataset(str(data_path))
    logger.info(f"Data loaded: {len(raw_df)} rows")
    return raw_df


# ── V1 Training ──────────────────────────────────────────────────────

def _train_v1(
    raw_df: pd.DataFrame,
    profile: dict,
    args: argparse.Namespace,
    decision_day: pd.Timestamp,
    output_dir: Path,
) -> dict:
    """Train a V1 per-hour model."""
    from models.deep_sgdf_delta.train import TrainConfig, train_model
    from models.deep_sgdf_delta.dataset import DEFAULT_FEATURE_CONFIG

    # Build training config
    epochs = args.epochs if args.epochs else 30
    batch_size = args.batch_size if args.batch_size else 256
    window_days = args.window_days if args.window_days else 7

    train_config = TrainConfig(
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        backbone=profile["backbone"],
        tcn_kernel_size=3,
        tcn_dilation_base=2,
        segment_embed_dim=8,
        use_global_residual=True,
        global_residual_weight=0.3,
        amp_enabled=args.amp,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=5,
        device=args.device,
        window_days=window_days,
        seed=42,
        val_days=args.val_days,
        train_min_rows=2160,
    )

    # Fast-dev-run overrides
    if args.fast_dev_run:
        train_config.epochs = min(train_config.epochs, 3)
        train_config.batch_size = min(train_config.batch_size, 8)
        train_config.val_days = min(train_config.val_days, 7)
        train_config.early_stopping_patience = 2
        train_config.amp_enabled = False
        logger.info("Fast-dev-run mode: epochs<=3, batch<=8, val_days<=7")

    result = train_model(
        raw_df,
        DEFAULT_FEATURE_CONFIG,
        train_config,
        decision_day=decision_day,
        output_dir=output_dir,
    )

    # Save the best model with a standard name
    best_path = output_dir / "best_model.pt"
    if not best_path.exists():
        import torch
        torch.save(result["model"].state_dict(), best_path)

    return result


# ── V2 Training ──────────────────────────────────────────────────────

def _train_v2(
    raw_df: pd.DataFrame,
    profile: dict,
    args: argparse.Namespace,
    decision_day: pd.Timestamp,
    output_dir: Path,
) -> dict:
    """Train a V2 day-level model."""
    from models.deep_sgdf_delta.train_v2 import TrainV2Config, train_model_v2
    from models.deep_sgdf_delta.dataset_v2 import DEFAULT_FEATURE_CONFIG

    epochs = args.epochs if args.epochs else 30
    batch_size = args.batch_size if args.batch_size else 64

    train_config = TrainV2Config(
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        backbone=profile["backbone"],
        tcn_kernel_size=3,
        tcn_dilation_base=2,
        transformer_nhead=4,
        transformer_dim_ff=128,
        hour_embed_dim=8,
        segment_embed_dim=8,
        use_residual_head=True,
        residual_weight=0.3,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=5,
        amp_enabled=args.amp,
        device=args.device,
        val_days=args.val_days,
        train_min_days=90,
        seed=42,
    )

    result = train_model_v2(
        raw_df,
        DEFAULT_FEATURE_CONFIG,
        train_config,
        decision_day=decision_day,
        output_dir=output_dir,
        fast_dev_run=args.fast_dev_run,
    )

    # Standardize checkpoint name
    v2_path = output_dir / "best_model_v2.pt"
    best_path = output_dir / "best_model.pt"
    if v2_path.exists() and not best_path.exists():
        import shutil
        shutil.copy2(str(v2_path), str(best_path))

    return result


# ── Config snapshot ──────────────────────────────────────────────────

def _write_config_snapshot(
    output_dir: Path,
    profile_name: str,
    profile: dict,
    args: argparse.Namespace,
    decision_day: pd.Timestamp,
    result: dict,
) -> None:
    """Write a YAML config snapshot of the training run."""
    snapshot = {
        "run_id": output_dir.name,
        "profile": profile_name,
        "profile_description": profile["description"],
        "version": profile["version"],
        "backbone": profile["backbone"],
        "blend_mode": profile["blend"],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "decision_day": str(decision_day.date()),
        "val_days": args.val_days,
        "device": args.device,
        "amp": args.amp,
        "fast_dev_run": args.fast_dev_run,
        "data_path": args.data_path,
        "sgdfnet_root": args.sgdfnet_root,
        "best_val_smape": result.get("best_val_smape"),
        "best_epoch": result.get("best_epoch"),
        "total_params": result.get("total_params"),
        "feature_count": len(result.get("feature_cols", [])),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if args.epochs:
        snapshot["epochs_override"] = args.epochs
    if args.batch_size:
        snapshot["batch_size_override"] = args.batch_size
    if args.window_days:
        snapshot["window_days_override"] = args.window_days

    config_path = output_dir / "config_snapshot.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"Config snapshot written to {config_path}")


# ── Training curve ───────────────────────────────────────────────────

def _write_training_curve(output_dir: Path, result: dict) -> None:
    """Write training history as a CSV."""
    history = result.get("history", [])
    if history:
        curve_df = pd.DataFrame(history)
        curve_df.to_csv(output_dir / "training_curve.csv", index=False, encoding="utf-8-sig")
        logger.info(f"Training curve written ({len(history)} epochs)")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    t_start = time.time()

    # Resolve profile
    profile_name = args.profile
    profile = PROFILES[profile_name]
    logger.info(f"Profile: {profile_name} — {profile['description']}")
    logger.info(f"Version: {profile['version']}, Backbone: {profile['backbone']}, Blend: {profile['blend']}")

    # Resolve output directory
    run_id = f"phase2_{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = PROJECT_ROOT / "reports" / "local" / "phase2" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Resolve decision day
    if args.test_month:
        # Parse YYYY-MM, decision day is first day of that month
        decision_day = pd.Timestamp(args.test_month + "-01")
        logger.info(f"Test month mode: decision_day = {decision_day.date()}")
    else:
        # Decision day = start of evaluation period
        decision_day = pd.Timestamp(args.start_date)
    logger.info(f"Decision day: {decision_day.date()}")
    logger.info(f"Evaluation period: {args.start_date} to {args.end_date}")

    # Load data
    data_path = _resolve_data_path(args)
    raw_df = _load_raw_data(data_path, args.sgdfnet_root)

    # Train
    if profile["version"] == "v1":
        result = _train_v1(raw_df, profile, args, decision_day, output_dir)
    else:
        result = _train_v2(raw_df, profile, args, decision_day, output_dir)

    # Save artifacts
    _write_training_curve(output_dir, result)
    _write_config_snapshot(output_dir, profile_name, profile, args, decision_day, result)

    elapsed = time.time() - t_start

    # Print summary
    print()
    print("=" * 60)
    print(f"  Phase 2 Training Complete")
    print(f"  Profile:      {profile_name}")
    print(f"  Decision day: {decision_day.date()}")
    print(f"  Best val sMAPE_floor50: {result.get('best_val_smape', float('nan')):.4f}")
    print(f"  Best epoch:   {result.get('best_epoch', 'N/A')}")
    print(f"  Total params: {result.get('total_params', 0):,}")
    print(f"  Device:       {result.get('device', 'N/A')}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Output:       {output_dir}")
    print("=" * 60)
    print()
    print("Artifacts:")
    print(f"  {output_dir / 'best_model.pt'}")
    print(f"  {output_dir / 'training_curve.csv'}")
    print(f"  {output_dir / 'config_snapshot.yaml'}")


if __name__ == "__main__":
    main()
