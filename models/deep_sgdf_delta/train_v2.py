"""Training logic for DeepSGDFDeltaV2 — day-level 24-hour decoder.

Supports:
  - CombinedLoss adapted for 24h sequences (smoothness on full 24h)
  - Early stopping with patience
  - AMP (mixed precision) support
  - Cosine annealing LR scheduler
  - Walk-forward training per decision_day
  - --fast-dev-run mode (tiny data subset, few epochs)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .dataset_v2 import DaySequenceDataset, collate_fn_v2, build_training_datasets_v2
from .losses import CombinedLoss, SmoothnessLoss
from .metrics import smape_floor50
from .model_v2 import DeepSGDFDeltaV2Config, DeepSGDFDeltaV2, build_model_v2, count_parameters

logger = logging.getLogger(__name__)


# ── Train Config ─────────────────────────────────────────────────────

@dataclass
class TrainV2Config:
    """Training hyperparameters for V2."""
    # Model
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    backbone: str = "tcn"  # "tcn" | "gru" | "transformer_tiny"
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    transformer_nhead: int = 4
    transformer_dim_ff: int = 128
    hour_embed_dim: int = 8
    segment_embed_dim: int = 8
    use_residual_head: bool = True
    residual_weight: float = 0.3

    # Training
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    amp_enabled: bool = False
    device: str = "auto"

    # Loss weights
    w_smape: float = 0.55
    w_delta_mae: float = 0.25
    w_period: float = 0.10
    w_smooth: float = 0.10
    period_916_weight: float = 2.0

    # Data
    val_days: int = 30
    train_min_days: int = 90

    # Reproducibility
    seed: int = 42


# ── V2 Loss Wrapper ──────────────────────────────────────────────────

class CombinedLossV2(nn.Module):
    """Adapts CombinedLoss for 24-hour sequence predictions.

    Key difference from V1:
      - smooth_loss operates on the full [B, 24] sequence directly
      - Other losses flatten the 24h predictions to [B*24]
      - Supports valid_mask to skip padded hours
    """

    def __init__(
        self,
        w_smape: float = 0.55,
        w_delta_mae: float = 0.25,
        w_period: float = 0.10,
        w_smooth: float = 0.10,
        period_916_weight: float = 2.0,
        floor: float = 50.0,
    ):
        super().__init__()
        self.w_smape = w_smape
        self.w_delta_mae = w_delta_mae
        self.w_period = w_period
        self.w_smooth = w_smooth

        self.combined = CombinedLoss(
            w_smape=w_smape,
            w_delta_mae=w_delta_mae,
            w_period=w_period,
            w_smooth=w_smooth,
            period_916_weight=period_916_weight,
            floor=floor,
        )

    def forward(
        self,
        rt_pred_24: torch.Tensor,       # [B, 24]
        rt_true_24: torch.Tensor,        # [B, 24]
        delta_pred_24: torch.Tensor,     # [B, 24]
        delta_true_24: torch.Tensor,     # [B, 24]
        segment_ids_24: torch.Tensor,    # [B, 24]
        valid_mask: torch.Tensor | None = None,  # [B, 24]
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss on 24-hour predictions.

        For smoothness loss, delta_pred_24 [B, 24] is passed directly
        as the sequence — the SmoothnessLoss computes hour-to-hour diffs.

        For sMAPE/delta_MAE/period losses, valid hours are flattened.
        """
        device = rt_pred_24.device

        if valid_mask is not None:
            mask_bool = valid_mask.bool().view(-1)
            rt_pred_flat = rt_pred_24.reshape(-1)[mask_bool]
            rt_true_flat = rt_true_24.reshape(-1)[mask_bool]
            delta_pred_flat = delta_pred_24.reshape(-1)[mask_bool]
            delta_true_flat = delta_true_24.reshape(-1)[mask_bool]
            seg_flat = segment_ids_24.reshape(-1)[mask_bool]
        else:
            rt_pred_flat = rt_pred_24.reshape(-1)
            rt_true_flat = rt_true_24.reshape(-1)
            delta_pred_flat = delta_pred_24.reshape(-1)
            delta_true_flat = delta_true_24.reshape(-1)
            seg_flat = segment_ids_24.reshape(-1)

        if len(rt_pred_flat) == 0:
            zero = torch.tensor(0.0, device=device)
            return {"total": zero, "smape": zero, "delta_mae": zero, "period": zero, "smooth": zero}

        # Use the CombinedLoss for flat losses
        losses = self.combined(
            rt_pred=rt_pred_flat,
            rt_true=rt_true_flat,
            delta_pred=delta_pred_flat,
            delta_true=delta_true_flat,
            segment_ids=seg_flat,
            # Override smooth with None — we compute it ourselves below
            delta_pred_sequence=None,
        )

        # Smoothness loss on the full 24h sequence
        # Only use rows where at least 2 consecutive hours are valid
        if valid_mask is not None:
            # Mask out invalid hours by setting their delta to 0
            # (they won't contribute meaningful gradients to smoothness)
            masked_delta = delta_pred_24 * valid_mask
            losses["smooth"] = self.combined.smooth_loss(masked_delta)
        else:
            losses["smooth"] = self.combined.smooth_loss(delta_pred_24)

        # Recompute total with the real smooth loss
        losses["total"] = (
            self.w_smape * losses["smape"]
            + self.w_delta_mae * losses["delta_mae"]
            + self.w_period * losses["period"]
            + self.w_smooth * losses["smooth"]
        )

        return losses


# ── Helper functions ─────────────────────────────────────────────────

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


# ── Training loop ────────────────────────────────────────────────────

def train_one_epoch_v2(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedLossV2,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train one epoch on day-level 24h samples."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        features_24h = batch["features_24h"].to(device)        # [B, 24, F]
        segment_id = batch["segment_id"].to(device)             # [B]
        da_anchor_24 = batch["da_anchor_24"].to(device)         # [B, 24]
        delta_target_24 = batch["delta_target_24"].to(device)   # [B, 24]
        rt_actual_24 = batch["rt_actual_24"].to(device)         # [B, 24]
        segment_ids_24 = batch["segment_ids_24"].to(device)     # [B, 24]
        valid_mask = batch["valid_mask"].to(device)             # [B, 24]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                out = model(features_24h, segment_id, da_anchor_24)
                losses = loss_fn(
                    rt_pred_24=out["rt_pred_24"],
                    rt_true_24=rt_actual_24,
                    delta_pred_24=out["delta_pred_24"],
                    delta_true_24=delta_target_24,
                    segment_ids_24=segment_ids_24,
                    valid_mask=valid_mask,
                )
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(features_24h, segment_id, da_anchor_24)
            losses = loss_fn(
                rt_pred_24=out["rt_pred_24"],
                rt_true_24=rt_actual_24,
                delta_pred_24=out["delta_pred_24"],
                delta_true_24=delta_target_24,
                segment_ids_24=segment_ids_24,
                valid_mask=valid_mask,
            )
            losses["total"].backward()
            optimizer.step()

        batch_size = features_24h.size(0)
        total_loss += losses["total"].item() * batch_size
        count += batch_size

    return {"train_loss": total_loss / max(count, 1)}


# ── Validation ───────────────────────────────────────────────────────

@torch.no_grad()
def validate_v2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate on day-level 24h samples, computing per-hour metrics."""
    model.eval()
    all_rt_true = []
    all_rt_pred = []
    all_delta_true = []
    all_delta_pred = []

    for batch in loader:
        features_24h = batch["features_24h"].to(device)
        segment_id = batch["segment_id"].to(device)
        da_anchor_24 = batch["da_anchor_24"].to(device)

        out = model(features_24h, segment_id, da_anchor_24)

        valid_mask = batch["valid_mask"]  # [B, 24]
        rt_true = batch["rt_actual_24"]   # [B, 24]
        rt_pred = out["rt_pred_24"].cpu() # [B, 24]
        delta_true = batch["delta_target_24"]  # [B, 24]
        delta_pred = out["delta_pred_24"].cpu()  # [B, 24]

        # Only keep valid hours
        mask = valid_mask.bool()
        all_rt_true.append(rt_true[mask].numpy())
        all_rt_pred.append(rt_pred[mask].numpy())
        all_delta_true.append(delta_true[mask].numpy())
        all_delta_pred.append(delta_pred[mask].numpy())

    if not all_rt_true or all(len(a) == 0 for a in all_rt_true):
        return {"val_smape_floor50": float("inf"), "val_delta_mae": float("inf"), "val_samples": 0}

    rt_true_all = np.concatenate(all_rt_true)
    rt_pred_all = np.concatenate(all_rt_pred)
    delta_true_all = np.concatenate(all_delta_true)
    delta_pred_all = np.concatenate(all_delta_pred)

    return {
        "val_smape_floor50": smape_floor50(rt_true_all, rt_pred_all),
        "val_delta_mae": float(np.mean(np.abs(delta_pred_all - delta_true_all))),
        "val_samples": len(rt_true_all),
    }


# ── Main training entry point ────────────────────────────────────────

def train_model_v2(
    raw_df: pd.DataFrame,
    feature_config,
    train_config: TrainV2Config,
    *,
    decision_day: pd.Timestamp,
    output_dir: Path | None = None,
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    """Train DeepSGDFDeltaV2 for a single decision day (walk-forward).

    Args:
        raw_df: Raw data DataFrame
        feature_config: SGDFNet FeatureConfig
        train_config: V2 training config
        decision_day: Walk-forward decision day
        output_dir: Directory for saving best model checkpoint
        fast_dev_run: If True, use tiny subset and few epochs for quick test

    Returns:
        Dict with model, metrics, history, etc.
    """
    _set_seed(train_config.seed)
    device = _resolve_device(train_config.device)

    # Fast-dev-run overrides
    if fast_dev_run:
        train_config = _apply_fast_dev_overrides(train_config)

    # Build datasets
    train_min = 10 if fast_dev_run else train_config.train_min_days
    train_ds, val_ds, feature_cols = build_training_datasets_v2(
        raw_df, feature_config,
        decision_day=decision_day,
        val_days=train_config.val_days,
        train_min_days=train_min,
    )

    # Fast-dev-run: limit dataset size
    if fast_dev_run:
        max_days = min(20, len(train_ds))
        train_ds = _SubsetDayDataset(train_ds, max_days)
        max_val = min(5, len(val_ds))
        val_ds = _SubsetDayDataset(val_ds, max_val)

    logger.info(
        f"V2 Train days: {len(train_ds)}, Val days: {len(val_ds)}, "
        f"Features: {len(feature_cols)}, Device: {device}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size,
        shuffle=True, collate_fn=collate_fn_v2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size,
        shuffle=False, collate_fn=collate_fn_v2,
    )

    # Build model
    model_config = DeepSGDFDeltaV2Config(
        input_dim=len(feature_cols),
        hidden_dim=train_config.hidden_dim,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
        backbone=train_config.backbone,
        tcn_kernel_size=train_config.tcn_kernel_size,
        tcn_dilation_base=train_config.tcn_dilation_base,
        transformer_nhead=train_config.transformer_nhead,
        transformer_dim_ff=train_config.transformer_dim_ff,
        hour_embed_dim=train_config.hour_embed_dim,
        segment_embed_dim=train_config.segment_embed_dim,
        use_residual_head=train_config.use_residual_head,
        residual_weight=train_config.residual_weight,
        amp_enabled=train_config.amp_enabled,
    )
    model = build_model_v2(model_config).to(device)

    total_params = count_parameters(model)
    logger.info(f"V2 Model parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    loss_fn = CombinedLossV2(
        w_smape=train_config.w_smape,
        w_delta_mae=train_config.w_delta_mae,
        w_period=train_config.w_period,
        w_smooth=train_config.w_smooth,
        period_916_weight=train_config.period_916_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config.epochs,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if train_config.amp_enabled and device.type == "cuda"
        else None
    )

    # Training loop
    best_val_smape = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = train_one_epoch_v2(model, train_loader, loss_fn, optimizer, device, scaler)
        val_metrics = validate_v2(model, val_loader, device)
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
            f"val_delta_mae={val_metrics['val_delta_mae']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping
        if val_metrics["val_smape_floor50"] < best_val_smape:
            best_val_smape = val_metrics["val_smape_floor50"]
            best_epoch = epoch
            patience_counter = 0
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / "best_model_v2.pt")
        else:
            patience_counter += 1
            if patience_counter >= train_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    elapsed = time.time() - t_start

    # Load best model
    if output_dir and (output_dir / "best_model_v2.pt").exists():
        model.load_state_dict(
            torch.load(output_dir / "best_model_v2.pt", weights_only=True)
        )

    # Save training summary
    summary = {
        "best_val_smape": best_val_smape,
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "total_params": total_params,
        "elapsed_seconds": round(elapsed, 1),
        "backbone": train_config.backbone,
        "hidden_dim": train_config.hidden_dim,
        "fast_dev_run": fast_dev_run,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train_summary_v2.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"V2 Training complete: best_smape={best_val_smape:.4f} "
        f"at epoch {best_epoch}, {elapsed:.1f}s elapsed"
    )

    return {
        "model": model,
        "model_config": model_config,
        "feature_cols": feature_cols,
        "best_val_smape": best_val_smape,
        "best_epoch": best_epoch,
        "history": history,
        "total_params": total_params,
        "device": str(device),
        "summary": summary,
    }


# ── Fast-dev-run helpers ─────────────────────────────────────────────

def _apply_fast_dev_overrides(config: TrainV2Config) -> TrainV2Config:
    """Return a copy of config with fast-dev-run overrides."""
    from copy import deepcopy
    cfg = deepcopy(config)
    cfg.epochs = min(cfg.epochs, 3)
    cfg.batch_size = min(cfg.batch_size, 8)
    cfg.val_days = min(cfg.val_days, 7)
    cfg.early_stopping_patience = 2
    cfg.amp_enabled = False
    logger.info("Fast-dev-run mode: epochs<=3, batch<=8, val_days<=7")
    return cfg


class _SubsetDayDataset(Dataset):
    """Wrapper to take a subset of a DaySequenceDataset for fast-dev-run."""

    def __init__(self, dataset: DaySequenceDataset, max_samples: int):
        self._ds = dataset
        self._indices = list(range(min(max_samples, len(dataset))))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._ds[self._indices[idx]]


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    """CLI entry point for V2 training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train DeepSGDFDeltaV2")
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw data file")
    parser.add_argument("--decision-day", type=str, required=True, help="Decision day (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="reports/local/deep_sgdf_delta_v2")
    parser.add_argument("--backbone", type=str, default="tcn", choices=["tcn", "gru", "transformer_tiny"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick test with tiny data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from sgdfnet.data_contract import load_dataset

    raw_df = load_dataset(args.data_path)
    decision_day = pd.Timestamp(args.decision_day)

    config = TrainV2Config(
        hidden_dim=args.hidden_dim,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_days=args.val_days,
        device=args.device,
        amp_enabled=args.amp,
        seed=args.seed,
    )

    from .dataset_v2 import DEFAULT_FEATURE_CONFIG

    result = train_model_v2(
        raw_df, DEFAULT_FEATURE_CONFIG, config,
        decision_day=decision_day,
        output_dir=Path(args.output_dir),
        fast_dev_run=args.fast_dev_run,
    )

    print(f"\nBest val sMAPE_floor50: {result['best_val_smape']:.4f}")
    print(f"Total params: {result['total_params']:,}")


if __name__ == "__main__":
    main()
