#!/usr/bin/env python
"""Multi-month walk-forward backtest for model comparison and champion selection.

For each target month and each model profile, trains a fresh model, predicts
on the test month, and evaluates metrics.  Results are collected into a
leaderboard, and a champion is selected based on multi-criteria ranking.

Usage:
    python scripts/run_realtime_model_backtest.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-months 2026-01,2026-02,2026-03,2026-04,2026-05 \
        --profiles trendknight_rt_tcn,trendknight_rt_gru \
        --out-dir reports/local/deep_final/backtest \
        --epochs 30 --batch-size 128

    # Skip training (reuse existing artifacts):
    python scripts/run_realtime_model_backtest.py \
        --data-path data.csv --target-months 2026-01,2026-02 \
        --profiles trendknight_rt_tcn \
        --out-dir reports/local/deep_final/backtest \
        --skip-training

    # Fast dev run:
    python scripts/run_realtime_model_backtest.py \
        --data-path data.csv --target-months 2026-02 \
        --profiles trendknight_rt_tcn \
        --out-dir reports/local/deep_final/backtest \
        --fast-dev-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
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
from models.deep_sgdf_delta.metrics import (
    compute_full_metrics,
    compute_monthly_metrics,
    compute_period_mask,
    smape_floor50,
)
from models.deep_sgdf_delta.business_time import (
    add_business_time_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_realtime_model_backtest")


# -- Model profiles (same as train_realtime_deep_model.py) --------------------

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


# -- Verdict thresholds -------------------------------------------------------

PASS_THRESHOLD = 15.0
STRONG_THRESHOLD = 17.0
STRONG_916_THRESHOLD = 22.0
ACCEPTABLE_THRESHOLD = 20.0


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-month walk-forward backtest for TrendKnightRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to hourly CSV data file",
    )
    parser.add_argument(
        "--target-months", type=str, required=True,
        help="Comma-separated target months (YYYY-MM)",
    )
    parser.add_argument(
        "--profiles", type=str, required=True,
        help="Comma-separated model profiles to evaluate",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for backtest results",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Max training epochs per month (default: 30)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping patience (default: 7)",
    )
    parser.add_argument(
        "--val-days", type=int, default=30,
        help="Validation window in days (default: 30)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False,
        help="Enable automatic mixed precision",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training; reuse existing artifacts from out-dir",
    )
    parser.add_argument(
        "--fast-dev-run", action="store_true",
        help="Smoke test: minimal epochs and data",
    )
    parser.add_argument(
        "--spike-threshold", type=float, default=500.0,
        help="Price threshold for spike classification (default: 500)",
    )
    return parser.parse_args()


# -- Reproducibility ----------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -- Data loading -------------------------------------------------------------

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load the raw hourly dataset from CSV or XLSX."""
    path = Path(data_path)
    if not path.exists():
        alt = PROJECT_ROOT / data_path
        if alt.exists():
            path = alt
        else:
            alt2 = PROJECT_ROOT.parent / data_path
            if alt2.exists():
                path = alt2
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = path.suffix.lower()
    logger.info("Loading data from %s", path)

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

    logger.info("Raw data loaded: %d rows, %d columns", len(df), len(df.columns))

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
        raise ValueError("Data must contain a timestamp column")

    # Ensure required columns
    if "da_anchor" not in df.columns:
        raise ValueError("Data must contain da_anchor / forecast_price / 日前电价")
    if "rt_actual" not in df.columns:
        raise ValueError("Data must contain rt_actual / rt_price / 实时电价")

    # Ensure sgdfnet_pred exists
    if "sgdfnet_pred" not in df.columns:
        logger.info("sgdfnet_pred not found — using da_anchor as placeholder")
        df["sgdfnet_pred"] = df["da_anchor"]
    else:
        mask = df["sgdfnet_pred"].isna()
        if mask.any():
            df.loc[mask, "sgdfnet_pred"] = df.loc[mask, "da_anchor"]

    # forecast_price alias
    if "forecast_price" not in df.columns and "da_anchor" in df.columns:
        df["forecast_price"] = df["da_anchor"]

    return df


# -- Loss functions (same as train_realtime_deep_model.py) --------------------

class ResidualMAELoss(nn.Module):
    def forward(self, residual_pred: torch.Tensor, residual_true: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(residual_pred, residual_true)


class Period916LossWeight(nn.Module):
    def __init__(self, weight: float = 1.5):
        super().__init__()
        self.weight = weight

    def forward(self, loss_per_hour: torch.Tensor, period_24: torch.Tensor) -> torch.Tensor:
        w = torch.where(period_24 == 1, self.weight, 1.0)
        return (loss_per_hour * w).mean()


class TrendKnightRTLoss(nn.Module):
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
        rt_pred: torch.Tensor,
        rt_true: torch.Tensor,
        delta_pred: torch.Tensor,
        delta_true: torch.Tensor,
        residual_pred: torch.Tensor,
        residual_true: torch.Tensor,
        period_24: torch.Tensor,
        mask_24: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rt_pred_m = rt_pred * mask_24
        rt_true_m = rt_true * mask_24
        delta_pred_m = delta_pred * mask_24
        delta_true_m = delta_true * mask_24
        residual_pred_m = residual_pred * mask_24
        residual_true_m = residual_true * mask_24

        losses: dict[str, torch.Tensor] = {}
        losses["smape"] = self.smape_loss(rt_pred_m, rt_true_m)
        losses["delta_mae"] = self.delta_mae(delta_pred_m, delta_true_m)
        losses["residual_mae"] = self.residual_mae(residual_pred_m, residual_true_m)
        losses["smoothness"] = self.smoothness(delta_pred_m)

        total = (
            self.w_smape * losses["smape"]
            + self.w_delta * losses["delta_mae"]
            + self.w_residual * losses["residual_mae"]
            + self.w_smooth * losses["smoothness"]
        )

        per_hour_err = torch.abs(rt_pred_m - rt_true_m)
        period_factor = self.period_weighter(per_hour_err, period_24)
        mean_err = per_hour_err.mean().clamp(min=1e-8)
        period_multiplier = (period_factor / mean_err).clamp(0.5, 3.0)
        total = total * period_multiplier

        losses["total"] = total
        return losses


# -- Training helpers ---------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TrendKnightRTLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
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

    for k in loss_accum:
        loss_accum[k] /= max(n_batches, 1)
    return loss_accum


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_rt_pred = []
    all_rt_true = []

    for batch in loader:
        features = batch["features_24h"].to(device)
        segment_id = batch["segment_id"].to(device)
        da_anchor = batch["da_anchor_24"].to(device)
        sgdfnet_pred = batch["sgdfnet_pred_24"].to(device)
        hour_ids = batch["hour_ids"].to(device)
        mask = batch["mask_24"].to(device)
        delta_target = batch["delta_target_24"].to(device)

        out = model(features, segment_id, da_anchor, sgdfnet_pred, hour_ids)
        rt_pred = out["trend_rt_pred_24"]
        rt_true = da_anchor + delta_target

        all_rt_pred.append((rt_pred * mask).cpu().numpy().ravel())
        all_rt_true.append((rt_true * mask).cpu().numpy().ravel())

    rt_pred_all = np.concatenate(all_rt_pred)
    rt_true_all = np.concatenate(all_rt_true)

    valid = rt_true_all != 0.0
    if valid.sum() == 0:
        return {"val_smape_floor50": float("nan")}

    return {"val_smape_floor50": smape_floor50(rt_true_all[valid], rt_pred_all[valid])}


# -- Single month training + evaluation ---------------------------------------

def run_single_month(
    raw_df: pd.DataFrame,
    profile_name: str,
    profile: dict,
    target_month: str,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Train and evaluate a single model for a single target month.

    Returns a result dict with metrics and metadata.
    """
    month_dir = out_dir / profile_name / target_month
    month_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    result: dict = {
        "profile": profile_name,
        "target_month": target_month,
        "status": "OK",
        "error": None,
    }

    # Check if we can skip training
    if args.skip_training:
        model_path = month_dir / "best_model.pt"
        if model_path.exists():
            logger.info("Skipping training for %s/%s (artifact exists)", profile_name, target_month)
            # Load existing metrics
            metrics_path = month_dir / "test_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                result.update(existing)
                result["elapsed_s"] = 0.0
                return result
        else:
            logger.warning("skip-training but no artifact found for %s/%s — training anyway",
                           profile_name, target_month)

    # Device setup
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # Fast-dev-run overrides
    epochs = args.epochs
    val_days = args.val_days
    patience = args.patience
    if args.fast_dev_run:
        epochs = min(epochs, 3)
        val_days = 5
        patience = 2

    # Build datasets
    logger.info("Building datasets for %s/%s ...", profile_name, target_month)
    train_min_days = 10 if args.fast_dev_run else 90
    try:
        train_ds, val_ds, test_ds, manifest = build_training_datasets_final(
            raw_df, target_month=target_month,
            val_days=val_days, train_min_days=train_min_days,
        )
    except Exception as e:
        logger.error("Dataset build failed for %s/%s: %s", profile_name, target_month, e)
        result["status"] = "FAILED"
        result["error"] = f"Dataset build failed: {e}"
        result["elapsed_s"] = time.time() - t_start
        return result

    logger.info("Datasets: train=%d, val=%d, test=%d days",
                train_ds.n_days, val_ds.n_days, test_ds.n_days)

    if train_ds.n_days == 0 or val_ds.n_days == 0:
        result["status"] = "FAILED"
        result["error"] = f"Empty train/val: train={train_ds.n_days}, val={val_ds.n_days}"
        result["elapsed_s"] = time.time() - t_start
        return result

    # DataLoaders
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

    input_dim = train_ds.input_dim

    # Build model
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
    result["n_params"] = n_params

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = TrendKnightRTLoss(period_916_weight=1.5)

    # Training loop
    best_val_smape = float("inf")
    best_epoch = -1
    patience_counter = 0

    logger.info("Training %s/%s: epochs=%d, params=%s",
                profile_name, target_month, epochs, f"{n_params:,}")

    try:
        for epoch in range(1, epochs + 1):
            train_losses = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
            )
            val_metrics = evaluate_model(model, val_loader, device)
            val_smape = val_metrics["val_smape_floor50"]
            scheduler.step()

            is_best = val_smape < best_val_smape
            if is_best:
                best_val_smape = val_smape
                best_epoch = epoch
                patience_counter = 0
                # Save checkpoint
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "config": asdict(model_config),
                    "model_class": "TrendKnightRT",
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }
                torch.save(checkpoint, month_dir / "best_model.pt")
            else:
                patience_counter += 1

            if epoch % 5 == 0 or is_best or epoch == epochs:
                logger.info(
                    "  Epoch %d/%d  loss=%.4f  val_smape=%.4f  best=%.4f%s",
                    epoch, epochs, train_losses["total"], val_smape,
                    best_val_smape, " *" if is_best else "",
                )

            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    except Exception as e:
        logger.error("Training failed for %s/%s at epoch: %s", profile_name, target_month, e)
        result["status"] = "FAILED"
        result["error"] = f"Training failed: {e}"
        result["elapsed_s"] = time.time() - t_start
        return result

    result["best_val_smape"] = best_val_smape
    result["best_epoch"] = best_epoch

    # Test evaluation
    test_metrics: dict = {}
    if test_loader is not None and best_epoch > 0:
        try:
            ckpt = torch.load(month_dir / "best_model.pt", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            # Collect full predictions for detailed metrics
            all_rt_pred = []
            all_rt_true = []
            all_da = []
            all_delta_pred = []
            all_delta_true = []
            all_hours = []

            with torch.no_grad():
                for batch in test_loader:
                    features = batch["features_24h"].to(device)
                    segment_id = batch["segment_id"].to(device)
                    da_anchor = batch["da_anchor_24"].to(device)
                    sgdfnet_pred = batch["sgdfnet_pred_24"].to(device)
                    hour_ids = batch["hour_ids"].to(device)
                    mask = batch["mask_24"].to(device)
                    delta_target = batch["delta_target_24"].to(device)

                    out = model(features, segment_id, da_anchor, sgdfnet_pred, hour_ids)
                    rt_pred = (out["trend_rt_pred_24"] * mask).cpu().numpy().ravel()
                    rt_true = ((da_anchor + delta_target) * mask).cpu().numpy().ravel()
                    da = (da_anchor * mask).cpu().numpy().ravel()
                    dp = (out["delta_pred_24"] * mask).cpu().numpy().ravel()
                    dt = (delta_target * mask).cpu().numpy().ravel()

                    B = features.size(0)
                    hours_batch = []
                    for i in range(B):
                        for h in range(24):
                            hours_batch.append(h + 1)

                    all_rt_pred.append(rt_pred)
                    all_rt_true.append(rt_true)
                    all_da.append(da)
                    all_delta_pred.append(dp)
                    all_delta_true.append(dt)
                    all_hours.extend(hours_batch[:len(rt_pred)])

            rt_pred_all = np.concatenate(all_rt_pred)
            rt_true_all = np.concatenate(all_rt_true)
            da_all = np.concatenate(all_da)
            dp_all = np.concatenate(all_delta_pred)
            dt_all = np.concatenate(all_delta_true)
            hours_all = np.array(all_hours[:len(rt_pred_all)])

            # Filter valid
            valid = rt_true_all != 0.0
            if valid.sum() > 0:
                rp = rt_pred_all[valid]
                rt = rt_true_all[valid]
                da_v = da_all[valid]
                dp_v = dp_all[valid]
                dt_v = dt_all[valid]
                hrs = hours_all[valid]

                test_metrics["overall_sMAPE_floor50"] = smape_floor50(rt, rp)
                test_metrics["delta_mae"] = float(np.mean(np.abs(dp_v - dt_v)))
                test_metrics["rows_total"] = int(valid.sum())

                for period in ("1_8", "9_16", "17_24"):
                    pmask = compute_period_mask(hrs, period)
                    if pmask.sum() > 0:
                        test_metrics[f"{period}_sMAPE_floor50"] = smape_floor50(rt[pmask], rp[pmask])
                    else:
                        test_metrics[f"{period}_sMAPE_floor50"] = float("nan")

                # Bucket metrics
                spike_mask = np.abs(rt) > args.spike_threshold
                neg_mask = rt < 0.0
                normal_mask = ~spike_mask & ~neg_mask

                for bname, bmask in [("normal", normal_mask), ("negative", neg_mask), ("spike", spike_mask)]:
                    if bmask.sum() > 0:
                        test_metrics[f"{bname}_sMAPE_floor50"] = smape_floor50(rt[bmask], rp[bmask])
                        test_metrics[f"{bname}_count"] = int(bmask.sum())
                    else:
                        test_metrics[f"{bname}_sMAPE_floor50"] = float("nan")
                        test_metrics[f"{bname}_count"] = 0

                logger.info("Test metrics for %s/%s: overall_sMAPE=%.4f",
                            profile_name, target_month, test_metrics["overall_sMAPE_floor50"])
            else:
                test_metrics["overall_sMAPE_floor50"] = float("nan")
                test_metrics["rows_total"] = 0

            # Save test metrics
            test_metrics_json = {}
            for k, v in test_metrics.items():
                if isinstance(v, float) and np.isnan(v):
                    test_metrics_json[k] = None
                else:
                    test_metrics_json[k] = v
            with open(month_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics_json, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error("Test evaluation failed for %s/%s: %s", profile_name, target_month, e)
            test_metrics = {"overall_sMAPE_floor50": float("nan"), "error": str(e)}

    result.update(test_metrics)
    result["elapsed_s"] = time.time() - t_start

    # Model size
    model_pt = month_dir / "best_model.pt"
    if model_pt.exists():
        result["model_size_mb"] = round(model_pt.stat().st_size / (1024 * 1024), 2)

    return result


# -- DA anchor baseline for comparison ----------------------------------------

def compute_da_anchor_metrics(
    raw_df: pd.DataFrame,
    target_month: str,
    spike_threshold: float = 500.0,
) -> dict:
    """Compute DA anchor baseline metrics for a target month."""
    df = raw_df.copy()
    df = add_business_time_columns(df, timestamp_col="ds")

    target_period = pd.Period(target_month, freq="M")
    bd_month = pd.to_datetime(df["business_day"]).dt.to_period("M")
    mask = bd_month == target_period
    test_df = df[mask].copy()

    if test_df.empty:
        return {"overall_sMAPE_floor50": float("nan"), "rows_total": 0}

    yt = test_df["rt_actual"].to_numpy(dtype=float)
    yp = test_df["da_anchor"].to_numpy(dtype=float)
    hours = test_df["hour_business"].to_numpy(dtype=int)

    metrics: dict = {}
    metrics["overall_sMAPE_floor50"] = smape_floor50(yt, yp)
    metrics["rows_total"] = len(test_df)

    for period in ("1_8", "9_16", "17_24"):
        pmask = compute_period_mask(hours, period)
        if pmask.sum() > 0:
            metrics[f"{period}_sMAPE_floor50"] = smape_floor50(yt[pmask], yp[pmask])
        else:
            metrics[f"{period}_sMAPE_floor50"] = float("nan")

    spike_mask = np.abs(yt) > spike_threshold
    neg_mask = yt < 0.0
    normal_mask = ~spike_mask & ~neg_mask

    for bname, bmask in [("normal", normal_mask), ("negative", neg_mask), ("spike", spike_mask)]:
        if bmask.sum() > 0:
            metrics[f"{bname}_sMAPE_floor50"] = smape_floor50(yt[bmask], yp[bmask])
            metrics[f"{bname}_count"] = int(bmask.sum())
        else:
            metrics[f"{bname}_sMAPE_floor50"] = float("nan")
            metrics[f"{bname}_count"] = 0

    return metrics


# -- Champion selection -------------------------------------------------------

def select_champion(
    results: list[dict],
    da_anchor_metrics: dict[str, dict],
) -> dict:
    """Select champion profile based on multi-criteria ranking.

    Criteria (in priority order):
      1. Mean monthly overall sMAPE (lower is better)
      2. Mean monthly 9_16 sMAPE (lower is better)
      3. Negative bucket not worse than DA anchor
      4. Runtime / model size (lower is better)
    """
    # Group results by profile
    profile_results: dict[str, list[dict]] = {}
    for r in results:
        if r.get("status") != "OK":
            continue
        pname = r["profile"]
        if pname not in profile_results:
            profile_results[pname] = []
        profile_results[pname].append(r)

    if not profile_results:
        return {
            "champion": None,
            "reason": "No successful training runs",
            "verdict": "NO-GO",
        }

    # Compute per-profile aggregate metrics
    profile_scores: list[dict] = []
    for pname, runs in profile_results.items():
        smape_values = [r.get("overall_sMAPE_floor50", float("nan")) for r in runs]
        p916_values = [r.get("9_16_sMAPE_floor50", float("nan")) for r in runs]
        neg_values = [r.get("negative_sMAPE_floor50", float("nan")) for r in runs]
        model_sizes = [r.get("model_size_mb", 0) for r in runs]
        elapsed_values = [r.get("elapsed_s", 0) for r in runs]

        mean_smape = float(np.nanmean(smape_values)) if smape_values else float("nan")
        mean_916 = float(np.nanmean(p916_values)) if p916_values else float("nan")
        mean_neg = float(np.nanmean(neg_values)) if neg_values else float("nan")
        total_size = sum(model_sizes) if model_sizes else 0
        total_time = sum(elapsed_values) if elapsed_values else 0

        # Check negative bucket vs DA anchor
        da_neg_values = []
        for r in runs:
            month = r["target_month"]
            if month in da_anchor_metrics:
                da_neg = da_anchor_metrics[month].get("negative_sMAPE_floor50", float("nan"))
                if not np.isnan(da_neg):
                    da_neg_values.append(da_neg)

        neg_worse_than_da = False
        if da_neg_values and not np.isnan(mean_neg):
            mean_da_neg = np.mean(da_neg_values)
            neg_worse_than_da = mean_neg > mean_da_neg * 1.2  # 20% tolerance

        profile_scores.append({
            "profile": pname,
            "mean_overall_smape": mean_smape,
            "mean_916_smape": mean_916,
            "mean_neg_smape": mean_neg,
            "neg_worse_than_da": neg_worse_than_da,
            "total_model_size_mb": total_size,
            "total_time_s": total_time,
            "n_months": len(runs),
            "months": [r["target_month"] for r in runs],
        })

    # Sort by criteria
    profile_scores.sort(key=lambda x: (
        x["mean_overall_smape"] if not np.isnan(x["mean_overall_smape"]) else 999,
        x["mean_916_smape"] if not np.isnan(x["mean_916_smape"]) else 999,
        x["neg_worse_than_da"],
        x["total_model_size_mb"],
    ))

    champion = profile_scores[0]

    # Determine verdict
    mean_smape = champion["mean_overall_smape"]
    mean_916 = champion["mean_916_smape"]

    if np.isnan(mean_smape):
        verdict = "NO-GO"
        verdict_reason = "Mean overall sMAPE is NaN"
    elif mean_smape >= ACCEPTABLE_THRESHOLD:
        verdict = "NO-GO"
        verdict_reason = f"Mean overall sMAPE={mean_smape:.2f} >= {ACCEPTABLE_THRESHOLD}"
    elif mean_smape < PASS_THRESHOLD:
        verdict = "PASS"
        verdict_reason = f"Mean overall sMAPE={mean_smape:.2f} < {PASS_THRESHOLD}"
    elif mean_smape < STRONG_THRESHOLD and not np.isnan(mean_916) and mean_916 < STRONG_916_THRESHOLD:
        verdict = "STRONG"
        verdict_reason = (
            f"Mean overall sMAPE={mean_smape:.2f} < {STRONG_THRESHOLD} "
            f"AND 9_16={mean_916:.2f} < {STRONG_916_THRESHOLD}"
        )
    elif mean_smape < ACCEPTABLE_THRESHOLD:
        # Check if it beats DA anchor
        da_smape_values = []
        for month in champion["months"]:
            if month in da_anchor_metrics:
                da_val = da_anchor_metrics[month].get("overall_sMAPE_floor50", float("nan"))
                if not np.isnan(da_val):
                    da_smape_values.append(da_val)

        if da_smape_values and mean_smape >= np.mean(da_smape_values):
            verdict = "LOW_VALUE"
            verdict_reason = (
                f"Mean overall sMAPE={mean_smape:.2f} does not beat DA anchor "
                f"(mean={np.mean(da_smape_values):.2f}) but is stable"
            )
        else:
            verdict = "ACCEPTABLE"
            verdict_reason = f"Mean overall sMAPE={mean_smape:.2f} < {ACCEPTABLE_THRESHOLD}"
    else:
        verdict = "NO-GO"
        verdict_reason = f"Mean overall sMAPE={mean_smape:.2f} >= {ACCEPTABLE_THRESHOLD}"

    champion["verdict"] = verdict
    champion["verdict_reason"] = verdict_reason
    champion["all_profiles"] = profile_scores

    return champion


# -- Report generation --------------------------------------------------------

def write_champion_report(
    out_dir: Path,
    champion: dict,
    results: list[dict],
    da_anchor_metrics: dict[str, dict],
    target_months: list[str],
    profiles: list[str],
) -> None:
    """Write champion_report.md."""
    lines = [
        "# Backtest Champion Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Target months:** {', '.join(target_months)}",
        f"**Profiles evaluated:** {', '.join(profiles)}",
        "",
        "## Verdict",
        "",
        f"**Verdict:** {champion.get('verdict', 'N/A')}",
        "",
        f"**Reason:** {champion.get('verdict_reason', 'N/A')}",
        "",
        "## Verdict Thresholds",
        "",
        "| Level | Condition |",
        "|-------|-----------|",
        f"| PASS | mean overall < {PASS_THRESHOLD} |",
        f"| STRONG | mean overall < {STRONG_THRESHOLD} AND 9_16 < {STRONG_916_THRESHOLD} |",
        f"| ACCEPTABLE | mean overall < {ACCEPTABLE_THRESHOLD} |",
        f"| LOW_VALUE | does not beat DA anchor but stable |",
        f"| NO-GO | mean overall >= {ACCEPTABLE_THRESHOLD} or unstable |",
        "",
    ]

    # Champion details
    champ_profile = champion.get("profile", "N/A")
    if champ_profile:
        lines.extend([
            "## Champion Profile",
            "",
            f"**Profile:** {champ_profile}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean overall sMAPE | {champion.get('mean_overall_smape', float('nan')):.4f} |",
            f"| Mean 9_16 sMAPE | {champion.get('mean_916_smape', float('nan')):.4f} |",
            f"| Mean negative sMAPE | {champion.get('mean_neg_smape', float('nan')):.4f} |",
            f"| Negative worse than DA | {champion.get('neg_worse_than_da', False)} |",
            f"| Total model size | {champion.get('total_model_size_mb', 0):.1f} MB |",
            f"| Total training time | {champion.get('total_time_s', 0):.1f} s |",
            f"| Months trained | {champion.get('n_months', 0)} |",
            "",
        ])

    # Full leaderboard
    lines.extend([
        "## Full Leaderboard (all profiles)",
        "",
        "| Profile | Mean sMAPE | Mean 9_16 | Mean Neg | Neg worse? | Size (MB) | Time (s) |",
        "|---------|-----------|-----------|----------|------------|-----------|----------|",
    ])

    for ps in champion.get("all_profiles", []):
        lines.append(
            f"| {ps['profile']} "
            f"| {ps['mean_overall_smape']:.2f} "
            f"| {ps['mean_916_smape']:.2f} "
            f"| {ps['mean_neg_smape']:.2f} "
            f"| {ps['neg_worse_than_da']} "
            f"| {ps['total_model_size_mb']:.1f} "
            f"| {ps['total_time_s']:.1f} "
            f"|"
        )

    lines.append("")

    # Per-month detail
    lines.extend([
        "## Per-Month Detail",
        "",
        "| Profile | Month | Status | Overall sMAPE | 1_8 | 9_16 | 17_24 | Normal | Negative |",
        "|---------|-------|--------|---------------|-----|------|-------|--------|----------|",
    ])

    for r in results:
        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v:.2f}"

        lines.append(
            f"| {r['profile']} "
            f"| {r['target_month']} "
            f"| {r['status']} "
            f"| {fmt(r.get('overall_sMAPE_floor50'))} "
            f"| {fmt(r.get('1_8_sMAPE_floor50'))} "
            f"| {fmt(r.get('9_16_sMAPE_floor50'))} "
            f"| {fmt(r.get('17_24_sMAPE_floor50'))} "
            f"| {fmt(r.get('normal_sMAPE_floor50'))} "
            f"| {fmt(r.get('negative_sMAPE_floor50'))} "
            f"|"
        )

    lines.append("")

    # DA anchor reference
    lines.extend([
        "## DA Anchor Reference",
        "",
        "| Month | Overall sMAPE | 1_8 | 9_16 | 17_24 | Normal | Negative |",
        "|-------|---------------|-----|------|-------|--------|----------|",
    ])

    for month in target_months:
        if month in da_anchor_metrics:
            m = da_anchor_metrics[month]
            lines.append(
                f"| {month} "
                f"| {m.get('overall_sMAPE_floor50', float('nan')):.2f} "
                f"| {m.get('1_8_sMAPE_floor50', float('nan')):.2f} "
                f"| {m.get('9_16_sMAPE_floor50', float('nan')):.2f} "
                f"| {m.get('17_24_sMAPE_floor50', float('nan')):.2f} "
                f"| {m.get('normal_sMAPE_floor50', float('nan')):.2f} "
                f"| {m.get('negative_sMAPE_floor50', float('nan')):.2f} "
                f"|"
            )

    lines.append("")

    # Selection rationale
    lines.extend([
        "## Selection Rationale",
        "",
        "Champion selection criteria (in priority order):",
        "",
        "1. **Primary:** Mean monthly overall sMAPE (lower is better)",
        "2. **Secondary:** Mean monthly 9_16 sMAPE (lower is better)",
        "3. **Third:** Negative bucket not worse than DA anchor (>20% worse = penalized)",
        "4. **Fourth:** Total runtime / model size (lower is better)",
        "",
    ])

    if champion.get("error"):
        lines.extend([
            "## Errors",
            "",
            f"- {champion['error']}",
            "",
        ])

    report_path = out_dir / "champion_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Champion report written to %s", report_path)


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Parse inputs
    target_months = [m.strip() for m in args.target_months.split(",") if m.strip()]
    profile_names = [p.strip() for p in args.profiles.split(",") if p.strip()]

    # Validate profiles
    for pname in profile_names:
        if pname not in MODEL_PROFILES:
            logger.error("Unknown profile: %s. Available: %s", pname, list(MODEL_PROFILES.keys()))
            sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  TrendKnightRT Multi-Month Backtest")
    logger.info("=" * 60)
    logger.info("  Data:     %s", args.data_path)
    logger.info("  Months:   %s", target_months)
    logger.info("  Profiles: %s", profile_names)
    logger.info("  Output:   %s", out_dir)
    logger.info("  Epochs:   %d", args.epochs)
    logger.info("  Skip train: %s", args.skip_training)
    logger.info("=" * 60)

    # Set seed
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load data
    raw_df = load_raw_data(args.data_path)

    # Compute DA anchor baseline for each month
    logger.info("")
    logger.info("Computing DA anchor baseline metrics ...")
    da_anchor_metrics: dict[str, dict] = {}
    for month in target_months:
        da_m = compute_da_anchor_metrics(raw_df, month, args.spike_threshold)
        da_anchor_metrics[month] = da_m
        logger.info("  %s: DA anchor overall_sMAPE=%.4f", month, da_m.get("overall_sMAPE_floor50", float("nan")))

    # Run backtest for each profile × month
    all_results: list[dict] = []
    total_combos = len(profile_names) * len(target_months)
    combo_idx = 0

    for pname in profile_names:
        profile = MODEL_PROFILES[pname]
        logger.info("")
        logger.info("=" * 60)
        logger.info("  Profile: %s — %s", pname, profile["description"])
        logger.info("=" * 60)

        for month in target_months:
            combo_idx += 1
            logger.info("")
            logger.info("--- [%d/%d] %s / %s ---", combo_idx, total_combos, pname, month)

            try:
                result = run_single_month(
                    raw_df, pname, profile, month, args, device, out_dir,
                )
                all_results.append(result)

                status = result.get("status", "UNKNOWN")
                smape = result.get("overall_sMAPE_floor50", float("nan"))
                elapsed = result.get("elapsed_s", 0)

                if status == "OK":
                    logger.info("  Result: OK  sMAPE=%.4f  %.1fs", smape, elapsed)
                else:
                    logger.warning("  Result: %s  %s  %.1fs", status, result.get("error", ""), elapsed)

            except Exception as e:
                logger.error("Unhandled exception for %s/%s: %s", pname, month, e)
                traceback.print_exc()
                all_results.append({
                    "profile": pname,
                    "target_month": month,
                    "status": "FAILED",
                    "error": str(e),
                    "elapsed_s": 0,
                })

    # Build leaderboard CSV
    logger.info("")
    logger.info("Building leaderboard ...")

    leaderboard_rows = []
    for r in all_results:
        row = {
            "profile": r.get("profile"),
            "target_month": r.get("target_month"),
            "status": r.get("status"),
            "overall_sMAPE_floor50": r.get("overall_sMAPE_floor50"),
            "1_8_sMAPE_floor50": r.get("1_8_sMAPE_floor50"),
            "9_16_sMAPE_floor50": r.get("9_16_sMAPE_floor50"),
            "17_24_sMAPE_floor50": r.get("17_24_sMAPE_floor50"),
            "normal_sMAPE_floor50": r.get("normal_sMAPE_floor50"),
            "negative_sMAPE_floor50": r.get("negative_sMAPE_floor50"),
            "spike_sMAPE_floor50": r.get("spike_sMAPE_floor50"),
            "delta_mae": r.get("delta_mae"),
            "best_val_smape": r.get("best_val_smape"),
            "best_epoch": r.get("best_epoch"),
            "n_params": r.get("n_params"),
            "model_size_mb": r.get("model_size_mb"),
            "elapsed_s": r.get("elapsed_s"),
            "error": r.get("error"),
        }

        # Add DA anchor reference for comparison
        month = r.get("target_month", "")
        if month in da_anchor_metrics:
            row["da_anchor_sMAPE"] = da_anchor_metrics[month].get("overall_sMAPE_floor50")
        else:
            row["da_anchor_sMAPE"] = float("nan")

        leaderboard_rows.append(row)

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    leaderboard_df.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    logger.info("Leaderboard saved to %s", out_dir / "leaderboard.csv")

    # Champion selection
    logger.info("")
    logger.info("Selecting champion ...")
    champion = select_champion(all_results, da_anchor_metrics)

    # Save champion summary
    champion_summary = dict(champion)
    # Sanitize for JSON
    for k, v in champion_summary.items():
        if isinstance(v, float) and np.isnan(v):
            champion_summary[k] = None
    if "all_profiles" in champion_summary:
        for ps in champion_summary["all_profiles"]:
            for k, v in ps.items():
                if isinstance(v, float) and np.isnan(v):
                    ps[k] = None

    champion_summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    champion_summary["target_months"] = target_months
    champion_summary["profiles_evaluated"] = profile_names

    with open(out_dir / "champion_summary.json", "w", encoding="utf-8") as f:
        json.dump(champion_summary, f, ensure_ascii=False, indent=2)
    logger.info("Champion summary saved to %s", out_dir / "champion_summary.json")

    # Write champion report
    write_champion_report(out_dir, champion, all_results, da_anchor_metrics, target_months, profile_names)

    # Summary
    print()
    print("=" * 70)
    print("  Backtest Complete")
    print("=" * 70)
    print()
    print(f"  Verdict:  {champion.get('verdict', 'N/A')}")
    print(f"  Reason:   {champion.get('verdict_reason', 'N/A')}")
    print(f"  Champion: {champion.get('profile', 'N/A')}")
    print()

    if champion.get("profile"):
        print(f"  {'Profile':<30s} {'Mean sMAPE':>12s} {'Mean 9_16':>12s} {'Verdict':>10s}")
        print("  " + "-" * 66)
        for ps in champion.get("all_profiles", []):
            marker = " <--" if ps["profile"] == champion["profile"] else ""
            print(
                f"  {ps['profile']:<30s} "
                f"{ps['mean_overall_smape']:>12.2f} "
                f"{ps['mean_916_smape']:>12.2f} "
                f"{marker}"
            )
    else:
        print("  No champion selected.")

    print()

    # DA anchor reference
    print("  DA Anchor Reference:")
    for month in target_months:
        if month in da_anchor_metrics:
            da_val = da_anchor_metrics[month].get("overall_sMAPE_floor50", float("nan"))
            print(f"    {month}: {da_val:.2f}")

    print()
    print(f"  Output: {out_dir}")
    print("  Files:")
    for fpath in sorted(out_dir.rglob("*")):
        if fpath.is_file():
            rel = fpath.relative_to(out_dir)
            size_kb = fpath.stat().st_size / 1024
            print(f"    {str(rel):<50s}  {size_kb:>8.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
