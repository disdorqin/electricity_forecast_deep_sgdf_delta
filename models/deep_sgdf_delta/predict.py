"""Prediction logic for DeepSGDFDelta.

Supports three blend modes:
  - deep_only: use only the deep model prediction
  - sgdfnet_blend: weighted average of SGDFNet and deep model
  - sgdfnet_residual: deep model predicts residual on top of SGDFNet

All modes use only D-30 to D-1 validation window for weight learning.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import DeltaSequenceDataset, build_predict_dataset, _collate_fn
from .model import DeepSGDFDeltaConfig, DeepSGDFDeltaModel
from .train import TrainConfig

logger = logging.getLogger(__name__)

BlendMode = Literal["deep_only", "sgdfnet_blend", "sgdfnet_residual"]


@torch.no_grad()
def predict_delta(
    model: DeepSGDFDeltaModel,
    dataset: DeltaSequenceDataset,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Run prediction on a dataset and return a DataFrame with delta_pred and rt_pred."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    model.eval()

    rows = []
    for batch in loader:
        features = batch["features"].to(device)
        segment_ids = batch["segment_id"].to(device)
        da_anchor = batch["da_anchor"].to(device)

        out = model(features, segment_ids, da_anchor)

        for i in range(len(batch["da_anchor"])):
            rows.append({
                "delta_pred": out["delta_pred"][i].cpu().item(),
                "rt_pred": out["rt_pred"][i].cpu().item(),
                "da_anchor": batch["da_anchor"][i].item(),
                "segment_id": batch["segment_id"][i].item(),
                "hour": batch["hour"][i].item(),
            })

    return pd.DataFrame(rows)


def predict_with_blend(
    deep_pred_df: pd.DataFrame,
    sgdfnet_pred_df: pd.DataFrame | None,
    *,
    mode: BlendMode = "deep_only",
    blend_weight: float = 0.5,
) -> pd.DataFrame:
    """Apply blend mode to combine deep and SGDFNet predictions.

    Args:
        deep_pred_df: deep model predictions (must have delta_pred, rt_pred, da_anchor, hour)
        sgdfnet_pred_df: SGDFNet predictions (must have delta_hat, rt_hat, hour)
        mode: blend mode
        blend_weight: weight for SGDFNet in sgdfnet_blend mode

    Returns:
        DataFrame with final delta_pred and rt_pred columns
    """
    if mode == "deep_only" or sgdfnet_pred_df is None:
        return deep_pred_df.copy()

    # Merge on hour for alignment
    merged = deep_pred_df.merge(
        sgdfnet_pred_df[["hour", "delta_hat", "rt_hat"]].rename(
            columns={"delta_hat": "sgdf_delta", "rt_hat": "sgdf_rt"}
        ),
        on="hour",
        how="left",
        suffixes=("", "_sgdf"),
    )

    if mode == "sgdfnet_blend":
        w = blend_weight
        merged["delta_pred"] = w * merged["sgdf_delta"] + (1 - w) * merged["delta_pred"]
        merged["rt_pred"] = merged["da_anchor"] + merged["delta_pred"]
    elif mode == "sgdfnet_residual":
        # Deep model predicts residual on top of SGDFNet
        merged["delta_pred"] = merged["sgdf_delta"] + merged["delta_pred"]
        merged["rt_pred"] = merged["da_anchor"] + merged["delta_pred"]
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    return merged.drop(columns=["sgdf_delta", "sgdf_rt"], errors="ignore")


def find_optimal_blend_weight(
    deep_val_pred: pd.DataFrame,
    sgdfnet_val_pred: pd.DataFrame,
    val_y_true: np.ndarray,
    *,
    candidates: list[float] | None = None,
) -> float:
    """Find optimal blend weight using validation data (D-30 to D-1 only).

    Searches over candidate weights and picks the one with lowest sMAPE_floor50.
    """
    from .metrics import smape_floor50

    if candidates is None:
        candidates = [0.2, 0.4, 0.6, 0.8]

    best_w = 0.5
    best_score = float("inf")

    for w in candidates:
        blended_rt = w * sgdfnet_val_pred["rt_hat"].values + (1 - w) * deep_val_pred["rt_pred"].values
        score = smape_floor50(val_y_true, blended_rt)
        if score < best_score:
            best_score = score
            best_w = w

    logger.info(f"Optimal blend weight: {best_w} (val sMAPE: {best_score:.4f})")
    return best_w
