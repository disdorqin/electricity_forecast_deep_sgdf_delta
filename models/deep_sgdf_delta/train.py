"""Training logic for DeepSGDFDelta.

Supports:
  - Configurable backbone (TCN / GRU)
  - Combined loss (sMAPE_floor50 + delta_mae + period_916 + smoothness)
  - Early stopping with patience
  - AMP (optional, for RTX 4060 Laptop)
  - Walk-forward training per decision day
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import DeltaSequenceDataset, build_training_datasets, _collate_fn
from .losses import CombinedLoss
from .metrics import smape_floor50
from .model import DeepSGDFDeltaConfig, build_model

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Model
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    backbone: str = "tcn"  # "tcn" or "gru"
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    segment_embed_dim: int = 8
    use_global_residual: bool = True
    global_residual_weight: float = 0.3

    # Training
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    amp_enabled: bool = False
    device: str = "auto"  # "auto", "cpu", "cuda"

    # Sequence
    window_days: int = 7

    # Loss weights
    w_smape: float = 0.55
    w_delta_mae: float = 0.25
    w_period: float = 0.10
    w_smooth: float = 0.10
    period_916_weight: float = 2.0

    # Data
    val_days: int = 30
    train_min_rows: int = 2160

    # Reproducibility
    seed: int = 42


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        features = batch["features"].to(device)
        segment_ids = batch["segment_id"].to(device)
        da_anchor = batch["da_anchor"].to(device)
        delta_target = batch["delta_target"].to(device)
        rt_actual = batch["rt_actual"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                out = model(features, segment_ids, da_anchor)
                losses = loss_fn(
                    rt_pred=out["rt_pred"],
                    rt_true=rt_actual,
                    delta_pred=out["delta_pred"],
                    delta_true=delta_target,
                    segment_ids=segment_ids,
                    delta_pred_sequence=None,
                )
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(features, segment_ids, da_anchor)
            losses = loss_fn(
                rt_pred=out["rt_pred"],
                rt_true=rt_actual,
                delta_pred=out["delta_pred"],
                delta_true=delta_target,
                segment_ids=segment_ids,
                delta_pred_sequence=None,
            )
            losses["total"].backward()
            optimizer.step()

        total_loss += losses["total"].item() * len(batch["features"])
        count += len(batch["features"])

    return {"train_loss": total_loss / max(count, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_rt_true = []
    all_rt_pred = []
    all_delta_true = []
    all_delta_pred = []

    for batch in loader:
        features = batch["features"].to(device)
        segment_ids = batch["segment_id"].to(device)
        da_anchor = batch["da_anchor"].to(device)

        out = model(features, segment_ids, da_anchor)

        all_rt_true.append(batch["rt_actual"].numpy())
        all_rt_pred.append(out["rt_pred"].cpu().numpy())
        all_delta_true.append(batch["delta_target"].numpy())
        all_delta_pred.append(out["delta_pred"].cpu().numpy())

    rt_true = np.concatenate(all_rt_true)
    rt_pred = np.concatenate(all_rt_pred)
    delta_true = np.concatenate(all_delta_true)
    delta_pred = np.concatenate(all_delta_pred)

    return {
        "val_smape_floor50": smape_floor50(rt_true, rt_pred),
        "val_delta_mae": float(np.mean(np.abs(delta_pred - delta_true))),
        "val_samples": len(rt_true),
    }


def train_model(
    raw_df: pd.DataFrame,
    feature_config,
    train_config: TrainConfig,
    *,
    decision_day: pd.Timestamp,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Train DeepSGDFDelta for a single decision day (walk-forward).

    Returns a dict with model, metrics, and training history.
    """
    _set_seed(train_config.seed)
    device = _resolve_device(train_config.device)

    # Build datasets
    train_ds, val_ds, feature_cols = build_training_datasets(
        raw_df, feature_config,
        decision_day=decision_day,
        val_days=train_config.val_days,
        window_days=train_config.window_days,
        train_min_rows=train_config.train_min_rows,
    )

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Features: {len(feature_cols)}")

    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=False, collate_fn=_collate_fn)

    # Build model
    model_config = DeepSGDFDeltaConfig(
        input_dim=len(feature_cols),
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
        backbone=train_config.backbone,
        tcn_kernel_size=train_config.tcn_kernel_size,
        tcn_dilation_base=train_config.tcn_dilation_base,
        segment_embed_dim=train_config.segment_embed_dim,
        use_global_residual=train_config.use_global_residual,
        global_residual_weight=train_config.global_residual_weight,
        amp_enabled=train_config.amp_enabled,
    )
    model = build_model(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    loss_fn = CombinedLoss(
        w_smape=train_config.w_smape,
        w_delta_mae=train_config.w_delta_mae,
        w_period=train_config.w_period,
        w_smooth=train_config.w_smooth,
        period_916_weight=train_config.period_916_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs)

    scaler = torch.amp.GradScaler("cuda") if train_config.amp_enabled and device.type == "cuda" else None

    # Training loop
    best_val_smape = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"val_smape={val_metrics['val_smape_floor50']:.4f} | "
            f"val_delta_mae={val_metrics['val_delta_mae']:.4f}"
        )

        # Early stopping
        if val_metrics["val_smape_floor50"] < best_val_smape:
            best_val_smape = val_metrics["val_smape_floor50"]
            best_epoch = epoch
            patience_counter = 0
            if output_dir:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= train_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    # Load best model
    if output_dir and (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    return {
        "model": model,
        "model_config": model_config,
        "feature_cols": feature_cols,
        "best_val_smape": best_val_smape,
        "best_epoch": best_epoch,
        "history": history,
        "total_params": total_params,
        "device": str(device),
    }
