"""Prediction logic for DeepSGDFDeltaV2 — day-level 24-hour decoder.

Workflow:
  1. Run model on 24h input -> delta_pred_24 [B, 24], rt_pred_24 [B, 24]
  2. Expand back to per-row format: one row per (business_day, hour)
  3. Output DataFrame with business_day, hour, delta_pred, rt_pred, da_anchor, segment_id

Supports:
  - Batch prediction via DataLoader
  - Checkpoint loading
  - Blend modes (deep_only, sgdfnet_blend, sgdfnet_residual)
  - Metrics computation when ground truth is available
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset_v2 import DaySequenceDataset, collate_fn_v2
from .model_v2 import DeepSGDFDeltaV2, DeepSGDFDeltaV2Config, build_model_v2

logger = logging.getLogger(__name__)

BlendMode = Literal["deep_only", "sgdfnet_blend", "sgdfnet_residual"]


# ── Core prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_delta_v2(
    model: DeepSGDFDeltaV2,
    dataset: DaySequenceDataset,
    device: torch.device,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run V2 prediction on a DaySequenceDataset.

    Returns a DataFrame with one row per (business_day, hour):
      - business_day: Timestamp
      - hour: int (1-24)
      - delta_pred: float
      - rt_pred: float
      - da_anchor: float
      - segment_id: int (0/1/2)
      - valid: bool (whether the hour was present in the input)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn_v2,
    )
    model.eval()

    all_rows: list[dict] = []
    business_days = dataset.business_days

    batch_day_offset = 0
    for batch in loader:
        features_24h = batch["features_24h"].to(device)       # [B, 24, F]
        segment_id = batch["segment_id"].to(device)            # [B]
        da_anchor_24 = batch["da_anchor_24"].to(device)        # [B, 24]
        valid_mask = batch["valid_mask"]                       # [B, 24]
        segment_ids_24 = batch["segment_ids_24"]               # [B, 24]

        out = model(features_24h, segment_id, da_anchor_24)
        delta_pred_24 = out["delta_pred_24"].cpu()             # [B, 24]
        rt_pred_24 = out["rt_pred_24"].cpu()                   # [B, 24]
        da_anchor_24_cpu = batch["da_anchor_24"]               # [B, 24]

        B = features_24h.size(0)
        for i in range(B):
            day_idx = batch_day_offset + i
            bd = business_days[day_idx] if day_idx < len(business_days) else pd.NaT

            for h in range(24):
                hour = h + 1  # 1-24
                all_rows.append({
                    "business_day": bd,
                    "hour": hour,
                    "delta_pred": float(delta_pred_24[i, h].item()),
                    "rt_pred": float(rt_pred_24[i, h].item()),
                    "da_anchor": float(da_anchor_24_cpu[i, h].item()),
                    "segment_id": int(segment_ids_24[i, h].item()),
                    "valid": bool(valid_mask[i, h].item()),
                })

        batch_day_offset += B

    return pd.DataFrame(all_rows)


# ── Blend modes ──────────────────────────────────────────────────────

def predict_with_blend_v2(
    deep_pred_df: pd.DataFrame,
    sgdfnet_pred_df: pd.DataFrame | None,
    *,
    mode: BlendMode = "deep_only",
    blend_weight: float = 0.5,
) -> pd.DataFrame:
    """Apply blend mode to combine deep V2 and SGDFNet predictions.

    Args:
        deep_pred_df: V2 predictions (business_day, hour, delta_pred, rt_pred, da_anchor)
        sgdfnet_pred_df: SGDFNet predictions (hour, delta_hat, rt_hat) or (business_day, hour, ...)
        mode: blend mode
        blend_weight: weight for SGDFNet in sgdfnet_blend mode

    Returns:
        DataFrame with final delta_pred and rt_pred columns
    """
    if mode == "deep_only" or sgdfnet_pred_df is None:
        return deep_pred_df.copy()

    # Determine merge keys
    merge_keys = ["hour"]
    if "business_day" in sgdfnet_pred_df.columns and "business_day" in deep_pred_df.columns:
        merge_keys = ["business_day", "hour"]

    sgdf_cols = sgdfnet_pred_df.copy()
    if "delta_hat" in sgdf_cols.columns:
        sgdf_cols = sgdf_cols.rename(columns={"delta_hat": "sgdf_delta", "rt_hat": "sgdf_rt"})

    merge_cols = merge_keys + [c for c in ["sgdf_delta", "sgdf_rt"] if c in sgdf_cols.columns]
    merged = deep_pred_df.merge(sgdf_cols[merge_cols], on=merge_keys, how="left")

    if mode == "sgdfnet_blend":
        w = blend_weight
        merged["delta_pred"] = w * merged["sgdf_delta"] + (1 - w) * merged["delta_pred"]
        merged["rt_pred"] = merged["da_anchor"] + merged["delta_pred"]
    elif mode == "sgdfnet_residual":
        merged["delta_pred"] = merged["sgdf_delta"] + merged["delta_pred"]
        merged["rt_pred"] = merged["da_anchor"] + merged["delta_pred"]
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    return merged.drop(columns=["sgdf_delta", "sgdf_rt"], errors="ignore")


# ── Checkpoint loading ───────────────────────────────────────────────

def load_model_v2(
    checkpoint_path: Path | str,
    config: DeepSGDFDeltaV2Config,
    device: torch.device | str = "cpu",
) -> DeepSGDFDeltaV2:
    """Load a trained V2 model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt state dict file
        config: Model config used during training
        device: Device to load the model onto

    Returns:
        Loaded DeepSGDFDeltaV2 model in eval mode
    """
    if isinstance(device, str):
        device = torch.device(device)
    model = build_model_v2(config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded V2 model from {checkpoint_path}")
    return model


# ── Metrics on predictions ──────────────────────────────────────────

def compute_v2_metrics(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame | None = None,
) -> dict:
    """Compute metrics from V2 prediction DataFrame.

    Args:
        pred_df: V2 predictions with columns (business_day, hour, delta_pred, rt_pred, da_anchor)
        true_df: Ground truth DataFrame with columns (business_day, hour, rt_actual, delta_target).
                 If None, only basic stats are returned.

    Returns:
        Dict with metrics
    """
    from .metrics import smape_floor50, compute_full_metrics

    result = {
        "num_days": pred_df["business_day"].nunique(),
        "num_hours": len(pred_df),
        "valid_hours": int(pred_df["valid"].sum()) if "valid" in pred_df.columns else len(pred_df),
    }

    if true_df is not None:
        # Merge predictions with ground truth
        merge_keys = ["business_day", "hour"]
        available_keys = [k for k in merge_keys if k in true_df.columns and k in pred_df.columns]
        if not available_keys:
            available_keys = ["hour"]

        merged = pred_df.merge(
            true_df[available_keys + ["rt_actual", "delta_target"]].rename(
                columns={"rt_actual": "rt_true", "delta_target": "delta_true"}
            ),
            on=available_keys,
            how="inner",
        )

        if merged.empty:
            result["smape_floor50"] = float("nan")
            return result

        # Compute full metrics
        metrics_df = merged.rename(columns={
            "rt_true": "rt_actual",
            "delta_true": "delta_target",
        })
        full = compute_full_metrics(metrics_df)
        result.update(full)

    return result


# ── High-level prediction pipeline ──────────────────────────────────

def run_prediction_v2(
    raw_df: pd.DataFrame,
    feature_config,
    model_config: DeepSGDFDeltaV2Config,
    checkpoint_path: Path | str,
    *,
    target_day: pd.Timestamp,
    device: torch.device | str = "auto",
    batch_size: int = 64,
    visible_frame: pd.DataFrame | None = None,
    sgdfnet_pred_df: pd.DataFrame | None = None,
    blend_mode: BlendMode = "deep_only",
    blend_weight: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Full V2 prediction pipeline.

    1. Build prediction dataset
    2. Load model from checkpoint
    3. Run prediction
    4. Optionally blend with SGDFNet
    5. Compute metrics if ground truth available

    Args:
        raw_df: Raw data DataFrame
        feature_config: SGDFNet FeatureConfig
        model_config: V2 model config
        checkpoint_path: Path to trained model checkpoint
        target_day: Day to predict
        device: Compute device
        batch_size: Batch size for prediction
        visible_frame: Cutoff-safe frame (optional)
        sgdfnet_pred_df: SGDFNet predictions for blending (optional)
        blend_mode: Blend mode
        blend_weight: Blend weight

    Returns:
        (prediction_df, metrics_dict)
    """
    from .dataset_v2 import build_predict_dataset_v2

    # Resolve device
    if isinstance(device, str):
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

    # Build dataset
    pred_ds, feature_cols = build_predict_dataset_v2(
        raw_df, feature_config,
        target_day=target_day,
        visible_frame=visible_frame,
    )
    logger.info(f"V2 Predict dataset: {len(pred_ds)} days, {len(feature_cols)} features")

    # Load model
    model = load_model_v2(checkpoint_path, model_config, device)

    # Predict
    pred_df = predict_delta_v2(model, pred_ds, device, batch_size=batch_size)

    # Filter to valid hours only for final output
    pred_df_valid = pred_df[pred_df["valid"] == True].copy()
    pred_df_valid = pred_df_valid.drop(columns=["valid"])

    # Blend if needed
    if blend_mode != "deep_only" and sgdfnet_pred_df is not None:
        pred_df_valid = predict_with_blend_v2(
            pred_df_valid, sgdfnet_pred_df,
            mode=blend_mode, blend_weight=blend_weight,
        )

    # Compute metrics if ground truth is available
    true_frame = raw_df.copy()
    if "rt_actual" not in true_frame.columns:
        from sgdfnet.data_contract import RT_COL, DA_COL
        true_frame["rt_actual"] = pd.to_numeric(true_frame[RT_COL], errors="coerce")
        true_frame["da_anchor"] = pd.to_numeric(true_frame[DA_COL], errors="coerce")
        true_frame["delta_target"] = true_frame["rt_actual"] - true_frame["da_anchor"]
    if "business_day" not in true_frame.columns:
        from sgdfnet.data_contract import add_business_time_columns, TIMESTAMP_COL
        true_frame = add_business_time_columns(true_frame)

    true_day = true_frame[true_frame["business_day"] == target_day].copy()
    if not true_day.empty and "rt_actual" in true_day.columns:
        true_day = true_day[["business_day", "target_hour", "rt_actual", "delta_target"]].rename(
            columns={"target_hour": "hour"}
        )
        true_day["hour"] = true_day["hour"].astype(int)
        metrics = compute_v2_metrics(pred_df_valid, true_day)
    else:
        metrics = compute_v2_metrics(pred_df_valid)

    return pred_df_valid, metrics


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    """CLI entry point for V2 prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict with DeepSGDFDeltaV2")
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw data file")
    parser.add_argument("--target-day", type=str, required=True, help="Target day (YYYY-MM-DD)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save predictions CSV")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from sgdfnet.data_contract import load_dataset
    from .dataset_v2 import DEFAULT_FEATURE_CONFIG
    from .model_v2 import DeepSGDFDeltaV2Config

    raw_df = load_dataset(args.data_path)
    target_day = pd.Timestamp(args.target_day)

    # Use default config (should match training config)
    model_config = DeepSGDFDeltaV2Config()

    pred_df, metrics = run_prediction_v2(
        raw_df, DEFAULT_FEATURE_CONFIG, model_config,
        checkpoint_path=args.checkpoint,
        target_day=target_day,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Save predictions
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Predictions saved to {out_path}")

    # Print metrics
    print("\n=== V2 Prediction Results ===")
    print(f"Days predicted: {metrics.get('num_days', 'N/A')}")
    print(f"Hours predicted: {metrics.get('num_hours', 'N/A')}")
    if "smape_floor50" in metrics:
        print(f"sMAPE_floor50: {metrics['smape_floor50']:.4f}")
    if "delta_mae" in metrics:
        print(f"Delta MAE: {metrics['delta_mae']:.4f}")

    return pred_df, metrics


if __name__ == "__main__":
    main()
