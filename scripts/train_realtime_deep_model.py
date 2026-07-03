#!/usr/bin/env python
"""TrendKnightRT (Phase DeepFinal-1) — main training entry point.

Trains TrendKnightRT models on hourly electricity price data with walk-forward
validation.  Supports three backbone profiles (TCN, GRU, Transformer-tiny),
multi-head loss (sMAPE + delta MAE + residual MAE + smoothness), period-aware
weighting, AMP, early stopping, and multi-month batch runs.

Usage:
    python scripts/train_realtime_deep_model.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --start-date 2022-01-01 --end-date 2026-06-21 \
        --target-month 2026-02 \
        --model-profile trendknight_rt_tcn \
        --out-dir artifacts/trendknight_rt/exp_001 \
        --epochs 50 --batch-size 128 --lr 0.001 --patience 7 --seed 42

    # Multi-month:
    python scripts/train_realtime_deep_model.py \
        --data-path data.csv --target-months 2026-01,2026-02,2026-03 \
        --model-profile trendknight_rt_tcn --out-dir artifacts/rt_multi

    # Smoke test:
    python scripts/train_realtime_deep_model.py \
        --data-path data.csv --target-month 2026-02 \
        --model-profile trendknight_rt_tcn --fast-dev-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yaml

# ── Path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.deep_sgdf_delta.trendknight_rt import (
    TrendKnightRTConfig,
    build_trendknight_rt,
    count_parameters,
)
from models.deep_sgdf_delta.realtime_dataset_final import (
    build_training_datasets_final,
    collate_fn_final,
)
from models.deep_sgdf_delta.losses import (
    SMAPEFloor50Loss,
    DeltaMAELoss,
    SmoothnessLoss,
)
from models.deep_sgdf_delta.metrics import smape_floor50
from models.deep_sgdf_delta.realtime_feature_contract import ALL_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_realtime_deep_model")


# ── Model profiles ───────────────────────────────────────────────────────

MODEL_PROFILES: dict[str, dict] = {
    "trendknight_rt_tcn": {
        "backbone": "tcn",
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "tcn_kernel_size": 3,
        "tcn_dilation_base": 2,
        "fusion_mode": "C",
        "multiscale": True,
        "description": "TrendKnightRT with TCN backbone",
    },
    "trendknight_rt_gru": {
        "backbone": "gru",
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "fusion_mode": "C",
        "multiscale": True,
        "description": "TrendKnightRT with GRU backbone",
    },
    "trendknight_rt_transformer": {
        "backbone": "transformer",
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "transformer_nhead": 4,
        "transformer_dim_ff": 128,
        "fusion_mode": "C",
        "multiscale": True,
        "description": "TrendKnightRT with Transformer-tiny backbone",
    },
}


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrendKnightRT (DeepFinal-1) training entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model profiles:
  trendknight_rt_tcn            TCN backbone
  trendknight_rt_gru            GRU backbone
  trendknight_rt_transformer    Transformer-tiny backbone

Examples:
  python scripts/train_realtime_deep_model.py \\
      --data-path data/shandong.csv --target-month 2026-02 \\
      --model-profile trendknight_rt_tcn --epochs 50

  python scripts/train_realtime_deep_model.py \\
      --data-path data/shandong.csv --target-months 2026-01,2026-02 \\
      --model-profile trendknight_rt_tcn --out-dir artifacts/rt

  python scripts/train_realtime_deep_model.py \\
      --data-path data/shandong.csv --target-month 2026-02 --fast-dev-run
""",
    )
    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to hourly CSV or XLSX data file")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Filter data start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Filter data end date (YYYY-MM-DD)")

    # Target month(s)
    parser.add_argument("--target-month", type=str, default=None,
                        help="Single target month (YYYY-MM) for test set")
    parser.add_argument("--target-months", type=str, default=None,
                        help="Comma-separated target months for multi-month runs")

    # Model
    parser.add_argument("--model-profile", type=str, required=True,
                        choices=list(MODEL_PROFILES.keys()),
                        help="Model profile to train")

    # Output
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for artifacts")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience on val sMAPE (default: 7)")
    parser.add_argument("--val-days", type=int, default=30,
                        help="Validation window in calendar days (default: 30)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Device / AMP
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable automatic mixed precision")

    # Dev mode
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Smoke test: last 10 train days + 5 val days + 3 epochs")

    # ── Phase DeepFinal-2: feature pipeline ──────────────────────────
    parser.add_argument("--sgdfnet-predictions", type=str, default=None,
                        help="Path to CSV with real SGDFNet predictions (ds, sgdfnet_pred)")
    parser.add_argument("--allow-sgdfnet-fallback", action="store_true", default=False,
                        help="Allow sgdfnet_pred fallback to da_anchor (smoke/predict only)")
    parser.add_argument("--feature-mode", type=str, default="full",
                        choices=["minimal", "full"],
                        help="Feature pipeline mode: 'full' (default) uses feature builder, "
                             "'minimal' uses raw data columns only")
    parser.add_argument("--feature-audit-only", action="store_true", default=False,
                        help="Run feature audit and exit without training")
    parser.add_argument("--strict-feature-contract", action="store_true", default=False,
                        help="Fail if required features are missing")

    return parser.parse_args()


# ── Reproducibility ──────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Chinese → English column mapping ────────────────────────────────────

# Standard Chinese column names used in the Shandong spot market CSV
_CN_TIMESTAMP = "时刻"
_CN_DA_PRICE = "日前电价"
_CN_RT_PRICE = "实时电价"

# Chinese forecast columns → English feature names
_CN_FORECAST_MAP: dict[str, str] = {
    "地方电厂总加预测值": "local_plant_forecast",
    "联络线受电负荷预测值": "tie_line_forecast",
    "风电总加预测值": "wind_forecast",
    "光伏总加预测值": "solar_forecast",
    "核电总加预测值": "nuclear_forecast",
    "自备机组总加预测值": "self_supply_forecast",
    "试验机组总加预测值": "test_unit_forecast",
    "直调负荷预测值": "dispatched_load_forecast",
    "竞价空间预测值": "bidding_space_forecast",
    "新能源总加预测值": "renewable_forecast",
}

# Chinese actual columns → English feature names
_CN_ACTUAL_MAP: dict[str, str] = {
    "地方电厂总加实际值": "local_plant_actual",
    "联络线受电负荷实际值": "tie_line_actual",
    "风电总加实际值": "wind_actual",
    "光伏总加实际值": "solar_actual",
    "核电总加实际值": "nuclear_actual",
    "自备机组总加实际值": "self_supply_actual",
    "试验机组总加实际值": "test_unit_actual",
    "直调负荷实际值": "dispatched_load_actual",
    "竞价空间实际值": "bidding_space_actual",
    "新能源总加实际值": "renewable_actual",
}


def _rename_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Chinese column names to canonical English names.

    Maps the Shandong spot market CSV columns (时刻, 日前电价, 实时电价,
    plus forecast/actual columns) to the English names expected by the
    feature contract and dataset builder.
    """
    rename_map: dict[str, str] = {}

    # Core columns
    if _CN_TIMESTAMP in df.columns:
        rename_map[_CN_TIMESTAMP] = "ds"
    if _CN_DA_PRICE in df.columns:
        rename_map[_CN_DA_PRICE] = "da_anchor"
    if _CN_RT_PRICE in df.columns:
        rename_map[_CN_RT_PRICE] = "rt_actual"

    # Forecast feature columns
    for cn_name, en_name in _CN_FORECAST_MAP.items():
        if cn_name in df.columns:
            rename_map[cn_name] = en_name

    # Actual feature columns
    for cn_name, en_name in _CN_ACTUAL_MAP.items():
        if cn_name in df.columns:
            rename_map[cn_name] = en_name

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("Renamed %d Chinese columns to English names", len(rename_map))

    return df


# ── Data loading ─────────────────────────────────────────────────────────

def load_raw_data(
    data_path: str,
    sgdfnet_pred_path: str | None = None,
    allow_fallback: bool = False,
) -> pd.DataFrame:
    """Load the raw hourly dataset from CSV or XLSX.

    Tries utf-8-sig first, then gbk for CSV.  Handles both Chinese column
    names (Shandong spot market format) and English column names.  Renames
    columns to canonical names.  SGDFNet predictions are loaded separately
    (not auto-filled from ``da_anchor``).

    Args:
        data_path: Path to CSV or XLSX data file.
        sgdfnet_pred_path: Optional path to CSV with SGDFNet predictions.
        allow_fallback: If True, missing sgdfnet_pred is filled from
            da_anchor.  Only for smoke/predict runs.

    Returns:
        DataFrame with canonical column names, business-time columns,
        and ``sgdfnet_pred`` if available (or fallback if allowed).

    Raises:
        FileNotFoundError: If *data_path* does not exist.
        ValueError: If core columns are missing.
        ValueError: If ``sgdfnet_pred`` is missing and *allow_fallback*
            is False.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    logger.info("Loading data from %s", path)

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".csv":
        # Try utf-8-sig first, fall back to gbk
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError):
            logger.info("utf-8-sig failed, retrying with gbk encoding")
            df = pd.read_csv(path, encoding="gbk")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx")

    logger.info("Raw data loaded: %d rows, %d columns", len(df), len(df.columns))

    # ── Handle Chinese column names ──────────────────────────────────
    # If the timestamp column is in Chinese, rename all Chinese columns
    if _CN_TIMESTAMP in df.columns:
        df = _rename_chinese_columns(df)

    # Also handle common English aliases
    alias_map = {
        "timestamp": "ds",
        "datetime": "ds",
        "date_time": "ds",
        "time": "ds",
        "rt_price": "rt_actual",
        "forecast_price": "da_anchor",
        "day_ahead_price": "da_anchor",
    }
    rename_aliases = {k: v for k, v in alias_map.items() if k in df.columns and v not in df.columns}
    if rename_aliases:
        df = df.rename(columns=rename_aliases)

    # Parse timestamp
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError(
            "Data must contain a timestamp column. "
            f"Expected 'ds' or '时刻'. Found columns: {list(df.columns[:10])}..."
        )

    # Ensure da_anchor exists
    if "da_anchor" not in df.columns:
        raise ValueError(
            "Data must contain a day-ahead price column. "
            "Expected 'da_anchor', 'forecast_price', or '日前电价'."
        )

    # Ensure rt_actual exists
    if "rt_actual" not in df.columns:
        raise ValueError(
            "Data must contain a realtime price column. "
            "Expected 'rt_actual', 'rt_price', or '实时电价'."
        )

    # ── SGDFNet predictions ──────────────────────────────────────────
    # Load from file or create placeholder
    has_sgdfnet = "sgdfnet_pred" in df.columns

    if sgdfnet_pred_path and not has_sgdfnet:
        # Load from external file
        if Path(sgdfnet_pred_path).exists():
            sgd_df = pd.read_csv(sgdfnet_pred_path)
            # Try to find timestamp and prediction columns
            ts_col = None
            for col in ["ds", "timestamp", "时刻", "time"]:
                if col in sgd_df.columns:
                    ts_col = col
                    break
            if ts_col is None:
                # Assume first column is timestamp
                ts_col = sgd_df.columns[0]

            sgd_df[ts_col] = pd.to_datetime(sgd_df[ts_col])
            sgd_map = sgd_df.set_index(ts_col)["sgdfnet_pred"].to_dict()
            df["sgdfnet_pred"] = df["ds"].map(sgd_map)
            logger.info("Loaded SGDFNet predictions from %s", sgdfnet_pred_path)
            has_sgdfnet = True
        else:
            logger.warning("SGDFNet predictions file not found: %s", sgdfnet_pred_path)

    if not has_sgdfnet:
        if allow_fallback:
            logger.info("sgdfnet_pred not found — using da_anchor as placeholder")
            df["sgdfnet_pred"] = df["da_anchor"]
        else:
            raise ValueError(
                "Missing sgdfnet_pred for formal training. "
                "Provide SGDFNet predictions via --sgdfnet-predictions "
                "or use --allow-sgdfnet-fallback for smoke only."
            )

    # Handle NaN in sgdfnet_pred
    if "sgdfnet_pred" in df.columns:
        mask = df["sgdfnet_pred"].isna()
        if mask.any():
            if allow_fallback:
                df.loc[mask, "sgdfnet_pred"] = df.loc[mask, "da_anchor"]
                logger.warning(
                    "Filled %d NaN sgdfnet_pred values with da_anchor (fallback)",
                    int(mask.sum()),
                )
            else:
                raise ValueError(
                    f"sgdfnet_pred has {int(mask.sum())} NaN values. "
                    "Provide complete predictions or use --allow-sgdfnet-fallback."
                )

    # Also create forecast_price alias if missing (feature contract expects it)
    if "forecast_price" not in df.columns and "da_anchor" in df.columns:
        df["forecast_price"] = df["da_anchor"]

    return df


def filter_date_range(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Filter DataFrame to [start_date, end_date] inclusive."""
    if start_date:
        df = df[df["ds"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["ds"] <= pd.Timestamp(end_date)]
    logger.info("After date filter: %d rows", len(df))
    return df.reset_index(drop=True)


# ── Loss functions ───────────────────────────────────────────────────────

class ResidualMAELoss(nn.Module):
    """MAE on residual (rt_actual - sgdfnet_pred)."""

    def forward(self, residual_pred: torch.Tensor, residual_true: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(residual_pred, residual_true)


class Period916LossWeight(nn.Module):
    """Multiply loss by 1.5 for hours 9-16 (period segment_id == 1)."""

    def __init__(self, weight: float = 1.5):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        loss_per_hour: torch.Tensor,
        period_24: torch.Tensor,
    ) -> torch.Tensor:
        """loss_per_hour: [B, 24], period_24: [B, 24] (0/1/2)"""
        w = torch.where(period_24 == 1, self.weight, 1.0)
        return (loss_per_hour * w).mean()


class TrendKnightRTLoss(nn.Module):
    """Combined loss for TrendKnightRT training.

    Loss = 0.5 * smape_loss + 0.3 * delta_mae_loss
         + 0.1 * residual_mae_loss + 0.1 * smoothness_loss

    Period 9_16 weighted loss: multiply total by 1.5 for hours 9-16.
    """

    def __init__(self, period_916_weight: float = 1.5):
        super().__init__()
        self.w_smape = 0.5
        self.w_delta = 0.3
        self.w_residual = 0.1
        self.w_smooth = 0.1

        self.smape_loss = SMAPEFloor50Loss(floor=50.0)
        self.delta_mae = DeltaMAELoss()
        self.residual_mae = ResidualMAELoss()
        self.smoothness = SmoothnessLoss()
        self.period_weighter = Period916LossWeight(weight=period_916_weight)

    def forward(
        self,
        rt_pred: torch.Tensor,          # [B, 24]
        rt_true: torch.Tensor,          # [B, 24]
        delta_pred: torch.Tensor,       # [B, 24]
        delta_true: torch.Tensor,       # [B, 24]
        residual_pred: torch.Tensor,    # [B, 24]
        residual_true: torch.Tensor,    # [B, 24]
        period_24: torch.Tensor,        # [B, 24]
        mask_24: torch.Tensor,          # [B, 24]
    ) -> dict[str, torch.Tensor]:
        # Apply mask: zero out invalid hours
        rt_pred_m = rt_pred * mask_24
        rt_true_m = rt_true * mask_24
        delta_pred_m = delta_pred * mask_24
        delta_true_m = delta_true * mask_24
        residual_pred_m = residual_pred * mask_24
        residual_true_m = residual_true * mask_24

        losses: dict[str, torch.Tensor] = {}

        # sMAPE on full realtime prediction
        losses["smape"] = self.smape_loss(rt_pred_m, rt_true_m)

        # Delta MAE
        losses["delta_mae"] = self.delta_mae(delta_pred_m, delta_true_m)

        # Residual MAE
        losses["residual_mae"] = self.residual_mae(residual_pred_m, residual_true_m)

        # Smoothness on delta prediction sequence
        losses["smoothness"] = self.smoothness(delta_pred_m)

        # Weighted sum
        total = (
            self.w_smape * losses["smape"]
            + self.w_delta * losses["delta_mae"]
            + self.w_residual * losses["residual_mae"]
            + self.w_smooth * losses["smoothness"]
        )

        # Period 9-16 weighting: apply to the total loss
        # Build per-hour absolute error for period weighting
        per_hour_err = torch.abs(rt_pred_m - rt_true_m)
        period_factor = self.period_weighter(per_hour_err, period_24)
        # Normalize: period_factor / mean(per_hour_err) gives a multiplier
        mean_err = per_hour_err.mean().clamp(min=1e-8)
        period_multiplier = period_factor / mean_err
        # Clamp multiplier to [0.5, 3.0] for stability
        period_multiplier = period_multiplier.clamp(0.5, 3.0)
        total = total * period_multiplier

        losses["total"] = total
        return losses


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run evaluation and return aggregated metrics."""
    model.eval()

    all_rt_pred = []
    all_rt_true = []
    all_delta_pred = []
    all_delta_true = []
    all_residual_pred = []
    all_residual_true = []

    smape_accum = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features_24h"].to(device)
        segment_id = batch["segment_id"].to(device)
        da_anchor = batch["da_anchor_24"].to(device)
        sgdfnet_pred = batch["sgdfnet_pred_24"].to(device)
        hour_ids = batch["hour_ids"].to(device)
        mask = batch["mask_24"].to(device)

        delta_target = batch["delta_target_24"].to(device)
        residual_target = batch["residual_target_24"].to(device)

        out = model(features, segment_id, da_anchor, sgdfnet_pred, hour_ids)

        rt_pred = out["trend_rt_pred_24"]
        rt_true = da_anchor + delta_target  # reconstruct rt_actual from anchor + delta

        # sMAPE per batch
        rt_pred_m = rt_pred * mask
        rt_true_m = rt_true * mask
        smape_accum += smape_floor50(
            rt_true_m.cpu().numpy().ravel(),
            rt_pred_m.cpu().numpy().ravel(),
        )
        n_batches += 1

        all_rt_pred.append(rt_pred_m.cpu().numpy().ravel())
        all_rt_true.append(rt_true_m.cpu().numpy().ravel())
        all_delta_pred.append((out["delta_pred_24"] * mask).cpu().numpy().ravel())
        all_delta_true.append((delta_target * mask).cpu().numpy().ravel())
        all_residual_pred.append((out["residual_to_sgdfnet_24"] * mask).cpu().numpy().ravel())
        all_residual_true.append((residual_target * mask).cpu().numpy().ravel())

    rt_pred_all = np.concatenate(all_rt_pred)
    rt_true_all = np.concatenate(all_rt_true)
    delta_pred_all = np.concatenate(all_delta_pred)
    delta_true_all = np.concatenate(all_delta_true)
    residual_pred_all = np.concatenate(all_residual_pred)
    residual_true_all = np.concatenate(all_residual_true)

    # Filter out zero-masked entries
    valid = rt_true_all != 0.0
    if valid.sum() == 0:
        return {"val_smape_floor50": float("nan"), "val_delta_mae": float("nan")}

    metrics = {
        "val_smape_floor50": smape_floor50(rt_true_all[valid], rt_pred_all[valid]),
        "val_delta_mae": float(np.mean(np.abs(delta_pred_all[valid] - delta_true_all[valid]))),
        "val_residual_mae": float(np.mean(np.abs(residual_pred_all[valid] - residual_true_all[valid]))),
        "val_rt_mae": float(np.mean(np.abs(rt_pred_all[valid] - rt_true_all[valid]))),
    }
    return metrics


# ── Training loop ────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TrendKnightRTLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train for one epoch and return loss dictionary."""
    model.train()
    loss_accum: dict[str, float] = {"total": 0.0, "smape": 0.0, "delta_mae": 0.0,
                                     "residual_mae": 0.0, "smoothness": 0.0}
    n_batches = 0

    for batch in loader:
        features = batch["features_24h"].to(device)
        segment_id = batch["segment_id"].to(device)
        da_anchor = batch["da_anchor_24"].to(device)
        sgdfnet_pred = batch["sgdfnet_pred_24"].to(device)
        hour_ids = batch["hour_ids"].to(device)
        mask = batch["mask_24"].to(device)
        period_24 = batch["period_24"].to(device)

        delta_target = batch["delta_target_24"].to(device)
        residual_target = batch["residual_target_24"].to(device)

        # Reconstruct rt_true = da_anchor + delta_target
        rt_true = da_anchor + delta_target

        use_amp = scaler is not None

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(features, segment_id, da_anchor, sgdfnet_pred, hour_ids)

            losses = criterion(
                rt_pred=out["trend_rt_pred_24"],
                rt_true=rt_true,
                delta_pred=out["delta_pred_24"],
                delta_true=delta_target,
                residual_pred=out["residual_to_sgdfnet_24"],
                residual_true=residual_target,
                period_24=period_24,
                mask_24=mask,
            )

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k in loss_accum:
            loss_accum[k] += losses[k].item()
        n_batches += 1

    # Average
    for k in loss_accum:
        loss_accum[k] /= max(n_batches, 1)

    return loss_accum


# ── Cosine Annealing Scheduler ───────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """CosineAnnealingLR scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


# ── Main training function ───────────────────────────────────────────────

def run_training(
    raw_df: pd.DataFrame,
    args: argparse.Namespace,
    profile: dict,
    target_month: str,
    output_dir: Path,
    feature_info: dict | None = None,
) -> dict:
    """Run full training pipeline for a single target month.

    Returns a dict with training results and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Device ───────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # AMP: auto-detect GPU
    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        logger.warning("AMP requested but no CUDA device — disabling AMP")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # ── Fast-dev-run overrides ───────────────────────────────────────
    epochs = args.epochs
    val_days = args.val_days
    patience = args.patience

    if args.fast_dev_run:
        epochs = min(epochs, 3)
        val_days = 5
        patience = 2
        logger.info(
            "fast-dev-run: epochs=%d, val_days=%d, patience=%d",
            epochs, val_days, patience,
        )

    # ── Build datasets ───────────────────────────────────────────────
    logger.info("Building datasets for target_month=%s ...", target_month)

    train_min_days = 10 if args.fast_dev_run else 90
    train_ds, val_ds, test_ds, manifest = build_training_datasets_final(
        raw_df,
        target_month=target_month,
        val_days=val_days,
        train_min_days=train_min_days,
        allow_sgdfnet_fallback=(
            args.allow_sgdfnet_fallback or args.fast_dev_run
        ),
    )

    logger.info(
        "Datasets: train=%d days, val=%d days, test=%d days",
        train_ds.n_days, val_ds.n_days, test_ds.n_days,
    )

    if train_ds.n_days == 0 or val_ds.n_days == 0:
        raise ValueError(
            f"Empty train/val set: train={train_ds.n_days}, val={val_ds.n_days}. "
            f"Check date range and target_month={target_month}."
        )

    # ── DataLoaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_final, drop_last=False, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_final, drop_last=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_final, drop_last=False, num_workers=0,
    ) if test_ds.n_days > 0 else None

    # ── Determine input_dim from dataset ─────────────────────────────
    input_dim = train_ds.input_dim
    logger.info("Input feature dim: %d", input_dim)

    # ── Build model ──────────────────────────────────────────────────
    model_config = TrendKnightRTConfig(
        input_dim=input_dim,
        hidden_dim=profile.get("hidden_dim", 64),
        num_layers=profile.get("num_layers", 2),
        dropout=profile.get("dropout", 0.1),
        backbone=profile["backbone"],
        tcn_kernel_size=profile.get("tcn_kernel_size", 3),
        tcn_dilation_base=profile.get("tcn_dilation_base", 2),
        transformer_nhead=profile.get("transformer_nhead", 4),
        transformer_dim_ff=profile.get("transformer_dim_ff", 128),
        fusion_mode=profile.get("fusion_mode", "C"),
        multiscale=profile.get("multiscale", True),
        use_sgdfnet_residual_head=True,
        use_delta_head=True,
        use_confidence_head=True,
        use_period_bias=True,
    )

    model = build_trendknight_rt(model_config).to(device)
    n_params = count_parameters(model)
    logger.info("Model built: %s backbone, %s params", profile["backbone"], f"{n_params:,}")

    # ── Optimizer & scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, epochs)
    criterion = TrendKnightRTLoss(period_916_weight=1.5)

    # ── Training loop ────────────────────────────────────────────────
    best_val_smape = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict] = []

    print()
    print("=" * 80)
    print(f"  TrendKnightRT Training — {args.model_profile}")
    print(f"  Target month: {target_month}  |  Device: {device}  |  AMP: {use_amp}")
    print(f"  Train: {train_ds.n_days}d  |  Val: {val_ds.n_days}d  |  Test: {test_ds.n_days}d")
    print(f"  Params: {n_params:,}  |  Epochs: {epochs}  |  LR: {args.lr}")
    print("=" * 80)
    print()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
        )

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        val_smape = val_metrics["val_smape_floor50"]

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - t_epoch

        # Track best
        is_best = val_smape < best_val_smape
        if is_best:
            best_val_smape = val_smape
            best_epoch = epoch
            patience_counter = 0
            # Save best model checkpoint
            _save_best_model(model, model_config, output_dir)
        else:
            patience_counter += 1

        # Record history
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_losses["total"],
            "train_smape": train_losses["smape"],
            "train_delta_mae": train_losses["delta_mae"],
            "train_residual_mae": train_losses["residual_mae"],
            "train_smoothness": train_losses["smoothness"],
            "val_smape_floor50": val_smape,
            "val_delta_mae": val_metrics["val_delta_mae"],
            "val_rt_mae": val_metrics["val_rt_mae"],
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_record)

        # Print progress
        best_str = f"{best_val_smape:.4f}" if best_val_smape < float("inf") else "N/A"
        marker = " *" if is_best else ""
        print(
            f"  Epoch {epoch:3d}/{epochs}  |  "
            f"train_loss={train_losses['total']:.4f}  "
            f"val_smape={val_smape:.4f}  "
            f"best={best_str}{marker}  "
            f"lr={current_lr:.6f}  "
            f"{epoch_time:.1f}s"
        )

        # Early stopping
        if patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d, best_smape=%.4f at epoch %d)",
                epoch, patience, best_val_smape, best_epoch,
            )
            break

    print()
    print(f"  Training complete. Best val sMAPE: {best_val_smape:.4f} at epoch {best_epoch}")
    print()

    # ── Load best model for test evaluation ──────────────────────────
    test_metrics: dict = {}
    if test_loader is not None and best_epoch > 0:
        ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        test_metrics = evaluate_model(model, test_loader, device)
        logger.info("Test metrics: %s", test_metrics)

    # ── Save artifacts ───────────────────────────────────────────────
    elapsed = time.time() - t_start

    # config.yaml
    _save_config_yaml(output_dir, args, profile, model_config, target_month)

    # feature_manifest.json
    with open(output_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    # train_manifest.json
    is_fallback_run = args.allow_sgdfnet_fallback or args.fast_dev_run
    train_manifest = {
        "model_profile": args.model_profile,
        "target_month": target_month,
        "backbone": profile["backbone"],
        "input_dim": input_dim,
        "n_params": n_params,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_smape_floor50": best_val_smape,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "device": str(device),
        "amp": use_amp,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": patience,
        "seed": args.seed,
        "fast_dev_run": args.fast_dev_run,
        "elapsed_seconds": round(elapsed, 1),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train_days": train_ds.n_days,
        "n_val_days": val_ds.n_days,
        "n_test_days": test_ds.n_days,
        # Phase DeepFinal-2 feature pipeline info
        "feature_mode": args.feature_mode,
        "n_features": input_dim,
        "sgdfnet_fallback_used": is_fallback_run,
        "sgdfnet_coverage": manifest.get("sgdfnet_coverage", 0.0),
        "feature_verdict": (feature_info or {}).get("verdict", "unknown"),
        "required_present": (feature_info or {}).get("required_present", []),
        "required_missing": (feature_info or {}).get("required_missing", []),
        "calendar_feature_generated": (feature_info or {}).get("calendar_feature_generated", False),
        "lag_feature_coverage": (feature_info or {}).get("lag_feature_coverage", 0.0),
        # Metric status: SMOKE_ONLY when fallback used
        "metric_status": "SMOKE_ONLY" if is_fallback_run else "FORMAL",
        "formal_metric": not is_fallback_run,
    }
    with open(output_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2, ensure_ascii=False)

    # metrics_summary.json
    metrics_summary = {
        "target_month": target_month,
        "best_val_smape_floor50": best_val_smape,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics if test_metrics else None,
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    # training_curves.csv
    if history:
        with open(output_dir / "training_curves.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Artifacts saved to: {output_dir}")
    print(f"    best_model.pt")
    print(f"    config.yaml")
    print(f"    feature_manifest.json")
    print(f"    train_manifest.json")
    print(f"    metrics_summary.json")
    print(f"    training_curves.csv")
    print("=" * 60)

    return {
        "best_val_smape": best_val_smape,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "history": history,
        "test_metrics": test_metrics,
        "elapsed": elapsed,
    }


# ── Checkpoint saving ────────────────────────────────────────────────────

def _save_best_model(
    model: nn.Module,
    config: TrendKnightRTConfig,
    output_dir: Path,
) -> None:
    """Save best model checkpoint with state_dict and config."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "model_class": "TrendKnightRT",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(checkpoint, output_dir / "best_model.pt")


def _save_config_yaml(
    output_dir: Path,
    args: argparse.Namespace,
    profile: dict,
    model_config: TrendKnightRTConfig,
    target_month: str,
) -> None:
    """Save full training config as YAML."""
    cfg = {
        "model_profile": args.model_profile,
        "profile_description": profile.get("description", ""),
        "target_month": target_month,
        "data_path": args.data_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "val_days": args.val_days,
            "seed": args.seed,
            "amp": args.amp,
            "device": args.device,
            "fast_dev_run": args.fast_dev_run,
        },
        "model": asdict(model_config),
        "profile": profile,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Resolve profile
    profile = MODEL_PROFILES[args.model_profile]
    logger.info("Model profile: %s — %s", args.model_profile, profile["description"])

    # ── Feature mode setup ──────────────────────────────────────────
    is_minimal = args.feature_mode == "minimal"
    if is_minimal and not args.fast_dev_run and not args.feature_audit_only:
        logger.warning(
            "feature_mode=minimal is for smoke only — "
            "do NOT use for formal training evaluation."
        )

    # ── Safety check: formal full mode requires SGDFNet predictions ─
    if args.feature_mode == "full" and not args.fast_dev_run:
        has_real_sgdfnet = (
            args.sgdfnet_predictions is not None
            and Path(args.sgdfnet_predictions).exists()
        )
        if not has_real_sgdfnet and not args.allow_sgdfnet_fallback:
            raise ValueError(
                "Formal full-feature training requires real SGDFNet predictions.\n"
                "  Provide --sgdfnet-predictions <path> or use --allow-sgdfnet-fallback\n"
                "  for smoke/predict only (metrics will be marked SMOKE_ONLY)."
            )

    # ── Load data ───────────────────────────────────────────────────
    raw_df = load_raw_data(
        args.data_path,
        sgdfnet_pred_path=args.sgdfnet_predictions,
        allow_fallback=args.allow_sgdfnet_fallback or args.fast_dev_run,
    )

    # Filter date range
    if args.start_date or args.end_date:
        raw_df = filter_date_range(raw_df, args.start_date, args.end_date)

    # ── Feature building (full mode) ────────────────────────────────
    feature_info: dict = {}

    if args.feature_mode == "full":
        from models.deep_sgdf_delta.realtime_feature_builder import (
            build_realtime_features,
            audit_feature_coverage,
        )

        logger.info("Building full feature set via realtime_feature_builder...")

        raw_df = build_realtime_features(
            raw_df,
            sgdfnet_pred_df=None,  # already merged in load_raw_data
            mode="FULL_DAY",
            allow_sgdfnet_fallback=(
                args.allow_sgdfnet_fallback or args.fast_dev_run
            ),
        )

        # Audit features
        audit = audit_feature_coverage(raw_df)
        feature_info = audit
        feature_info["feature_mode"] = "full"

        logger.info(
            "Feature audit: n_features=%d, verdict=%s, sgdfnet_coverage=%.1f%%",
            audit["n_features"], audit["verdict"], audit["sgdfnet_coverage"],
        )

        if audit["required_missing"]:
            logger.warning(
                "Missing required features: %s", audit["required_missing"],
            )
            if args.strict_feature_contract:
                raise ValueError(
                    f"Strict feature contract: missing required features: "
                    f"{audit['required_missing']}"
                )

        if args.feature_audit_only:
            print()
            print("=" * 60)
            print("  Feature Audit Only — exiting")
            print("=" * 60)
            audit_report_dir = PROJECT_ROOT / "reports" / "local" / "deep_final" / "features"
            audit_report_dir.mkdir(parents=True, exist_ok=True)
            import json
            with open(audit_report_dir / "realtime_feature_audit.json", "w") as f:
                json.dump(audit, f, indent=2, default=str)
            print(f"  Audit saved to {audit_report_dir / 'realtime_feature_audit.json'}")
            print(f"  Verdict: {audit['verdict']}")
            print(f"  n_features: {audit['n_features']}")
            return
    else:
        # Minimal mode: use raw columns only
        feature_info = {
            "feature_mode": "minimal",
            "n_features": len([c for c in ALL_FEATURES if c in raw_df.columns]),
            "verdict": "MINIMAL_MODE",
        }
        logger.info("Minimal feature mode — using raw data columns only")

    # Determine target months
    if args.target_months:
        target_months = [m.strip() for m in args.target_months.split(",")]
    elif args.target_month:
        target_months = [args.target_month.strip()]
    else:
        raise ValueError(
            "Must specify --target-month or --target-months"
        )

    logger.info("Target months: %s", target_months)

    # Resolve base output directory
    if args.out_dir:
        base_out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out_dir = PROJECT_ROOT / "artifacts" / "trendknight_rt" / f"exp_{timestamp}"

    # Run training for each target month
    results: list[dict] = []
    for i, target_month in enumerate(target_months):
        logger.info("=" * 60)
        logger.info("Processing target month %d/%d: %s", i + 1, len(target_months), target_month)
        logger.info("=" * 60)

        if len(target_months) > 1:
            month_out_dir = base_out_dir / target_month
        else:
            month_out_dir = base_out_dir

        result = run_training(raw_df, args, profile, target_month, month_out_dir,
                              feature_info=feature_info)
        result["target_month"] = target_month
        result["output_dir"] = str(month_out_dir)
        results.append(result)

    # Print final summary for multi-month runs
    if len(results) > 1:
        print()
        print("=" * 60)
        print("  Multi-Month Summary")
        print("=" * 60)
        for r in results:
            print(
                f"  {r['target_month']}  |  "
                f"best_smape={r['best_val_smape']:.4f}  "
                f"best_epoch={r['best_epoch']}  "
                f"params={r['n_params']:,}"
            )
        avg_smape = np.mean([r["best_val_smape"] for r in results
                             if r["best_val_smape"] < float("inf")])
        print(f"  {'Average':>10s}  |  avg_smape={avg_smape:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
