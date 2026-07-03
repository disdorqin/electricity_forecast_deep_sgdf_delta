"""Prediction logic for TrendKnight-X v3 — multiscale + teacher fusion.

Workflow:
  1. Run model on 24h input -> delta_pred_24, rt_pred_24, confidence, shock
  2. Expand back to per-row format: one row per (business_day, hour)
  3. Output DataFrame with delta_pred, rt_pred, confidence, shock_sensitivity
  4. Optionally blend with teacher predictions

Supports:
  - Batch prediction via DataLoader
  - Checkpoint loading
  - Teacher blend modes (deep_only, teacher_weighted, teacher_residual)
  - Confidence-based filtering
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset_v3 import DaySequenceDatasetV3, collate_fn_v3
from .model_v3 import TrendKnightV3, TrendKnightV3Config, build_model_v3

logger = logging.getLogger(__name__)

BlendMode = Literal["deep_only", "teacher_weighted", "teacher_residual"]


# ── Core prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_delta_v3(
    model: TrendKnightV3,
    dataset: DaySequenceDatasetV3,
    device: torch.device,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run V3 prediction on a DaySequenceDatasetV3.

    Returns a DataFrame with one row per (business_day, hour):
      - business_day: Timestamp
      - hour: int (1-24)
      - delta_pred: float
      - rt_pred: float
      - da_anchor: float
      - segment_id: int (0/1/2)
      - confidence: float [0, 1]
      - shock_sensitivity: float [0, 1]
      - multiscale_trend: float
      - multiscale_seasonal: float
      - multiscale_shock: float
      - valid: bool
    """
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn_v3,
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
        teacher_pred = batch["teacher_pred_24"].to(device)     # [B, T, 24]
        teacher_mask = batch["teacher_mask_24"].to(device)     # [B, T]

        out = model(
            features_24h, segment_id, da_anchor_24,
            teacher_features=teacher_pred,
            teacher_mask=teacher_mask,
        )

        delta_pred_24 = out["delta_pred_24"].cpu()            # [B, 24]
        rt_pred_24 = out["rt_pred_24"].cpu()                  # [B, 24]
        confidence_24 = out["confidence_24"].cpu()             # [B, 24]
        shock_24 = out["shock_sensitivity_24"].cpu()           # [B, 24]
        ms_trend = out["multiscale_trend"].cpu()               # [B, 24]
        ms_seasonal = out["multiscale_seasonal"].cpu()         # [B, 24]
        ms_shock = out["multiscale_shock"].cpu()               # [B, 24]
        da_anchor_cpu = batch["da_anchor_24"]                  # [B, 24]

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
                    "da_anchor": float(da_anchor_cpu[i, h].item()),
                    "segment_id": int(segment_ids_24[i, h].item()),
                    "confidence": float(confidence_24[i, h].item()),
                    "shock_sensitivity": float(shock_24[i, h].item()),
                    "multiscale_trend": float(ms_trend[i, h].item()),
                    "multiscale_seasonal": float(ms_seasonal[i, h].item()),
                    "multiscale_shock": float(ms_shock[i, h].item()),
                    "valid": bool(valid_mask[i, h].item()),
                })

        batch_day_offset += B

    return pd.DataFrame(all_rows)


# ── Blend modes ──────────────────────────────────────────────────────

def predict_with_blend_v3(
    deep_pred_df: pd.DataFrame,
    teacher_pred_df: pd.DataFrame | None = None,
    *,
    mode: BlendMode = "deep_only",
    blend_weight: float = 0.3,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    """Blend V3 deep predictions with teacher predictions.

    Args:
        deep_pred_df: V3 predictions (business_day, hour, delta_pred, rt_pred,
                      confidence, shock_sensitivity, ...)
        teacher_pred_df: Teacher predictions with columns
                         (business_day, hour, teacher_delta_pred) or
                         (hour, teacher_delta_pred)
        mode: Blend mode
            - deep_only: use deep predictions only
            - teacher_weighted: weighted average of deep + teacher,
              weighted by deep confidence
            - teacher_residual: deep prediction = teacher + student residual
        blend_weight: Weight for teacher in teacher_weighted mode
        confidence_threshold: Only blend when confidence > threshold

    Returns:
        DataFrame with final predictions including blend_delta and blend_rt
    """
    result = deep_pred_df.copy()

    if mode == "deep_only" or teacher_pred_df is None:
        result["final_delta_pred"] = result["delta_pred"]
        result["final_rt_pred"] = result["rt_pred"]
        return result

    # Determine merge keys
    merge_keys = ["hour"]
    if "business_day" in teacher_pred_df.columns and "business_day" in result.columns:
        merge_keys = ["business_day", "hour"]

    teacher_cols = teacher_pred_df.copy()

    # Find teacher delta column
    teacher_delta_col = None
    for candidate in ["teacher_delta_pred", "delta_hat", "teacher_pred",
                       "delta_pred_teacher"]:
        if candidate in teacher_cols.columns:
            teacher_delta_col = candidate
            break
    if teacher_delta_col is None and "delta_pred" in teacher_cols.columns:
        teacher_delta_col = "delta_pred"

    if teacher_delta_col is None:
        logger.warning("No teacher delta column found, using deep_only")
        result["final_delta_pred"] = result["delta_pred"]
        result["final_rt_pred"] = result["rt_pred"]
        return result

    merge_cols = merge_keys + [teacher_delta_col]
    merged = result.merge(
        teacher_cols[merge_cols], on=merge_keys, how="left",
        suffixes=("", "_teacher"),
    )

    teacher_delta = merged[teacher_delta_col].fillna(merged["delta_pred"])

    if mode == "teacher_weighted":
        # Confidence-weighted blend: higher confidence -> more weight on deep
        conf = merged["confidence"].clip(0, 1)
        deep_weight = conf * (1 - blend_weight) + (1 - conf) * blend_weight
        merged["final_delta_pred"] = (
            deep_weight * merged["delta_pred"]
            + (1 - deep_weight) * teacher_delta
        )
    elif mode == "teacher_residual":
        # Teacher prediction + student's learned residual
        student_residual = merged["delta_pred"]
        merged["final_delta_pred"] = teacher_delta + student_residual
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    # Apply confidence threshold: below threshold, trust teacher more
    if confidence_threshold > 0:
        low_conf = merged["confidence"] < confidence_threshold
        merged.loc[low_conf, "final_delta_pred"] = teacher_delta[low_conf]

    merged["final_rt_pred"] = merged["da_anchor"] + merged["final_delta_pred"]

    # Clean up
    drop_cols = [c for c in [teacher_delta_col] if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    return merged


# ── Checkpoint loading ───────────────────────────────────────────────

def load_model_v3(
    checkpoint_path: Path | str,
    config: TrendKnightV3Config | None = None,
    device: torch.device | str = "cpu",
) -> tuple[TrendKnightV3, list[str] | None]:
    """Load a trained V3 model from checkpoint.

    Supports two checkpoint formats:
      1. Dict with 'model_state_dict', 'model_config', 'feature_cols'
         (saved by train_v3)
      2. Plain state_dict (config must be provided)

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        config: Model config (required for plain state_dict checkpoints)
        device: Device to load the model onto

    Returns:
        (model, feature_cols) — model in eval mode, feature columns if available
    """
    if isinstance(device, str):
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Check if it's a rich checkpoint (from train_v3)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if config is None:
            config = ckpt.get("model_config")
        feature_cols = ckpt.get("feature_cols")

        if config is None:
            raise ValueError(
                "Checkpoint does not contain model_config and no config was provided"
            )
    else:
        # Plain state_dict
        state_dict = ckpt
        feature_cols = None
        if config is None:
            raise ValueError(
                "Plain state_dict checkpoint requires config to be provided"
            )

    model = build_model_v3(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded V3 model from {checkpoint_path}")

    return model, feature_cols


# ── Metrics on predictions ──────────────────────────────────────────

def compute_v3_metrics(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame | None = None,
) -> dict:
    """Compute metrics from V3 prediction DataFrame.

    Args:
        pred_df: V3 predictions with columns
                 (business_day, hour, delta_pred, rt_pred, confidence, ...)
        true_df: Ground truth with columns
                 (business_day, hour, rt_actual, delta_target)

    Returns:
        Dict with metrics
    """
    from .metrics import smape_floor50, compute_full_metrics

    result = {
        "num_days": pred_df["business_day"].nunique(),
        "num_hours": len(pred_df),
        "valid_hours": int(pred_df["valid"].sum()) if "valid" in pred_df.columns else len(pred_df),
    }

    # Confidence stats
    if "confidence" in pred_df.columns:
        valid_conf = pred_df.loc[pred_df.get("valid", pd.Series(True)).astype(bool), "confidence"]
        if len(valid_conf) > 0:
            result["mean_confidence"] = float(valid_conf.mean())
            result["std_confidence"] = float(valid_conf.std())

    # Shock sensitivity stats
    if "shock_sensitivity" in pred_df.columns:
        valid_shock = pred_df.loc[
            pred_df.get("valid", pd.Series(True)).astype(bool), "shock_sensitivity"
        ]
        if len(valid_shock) > 0:
            result["mean_shock_sensitivity"] = float(valid_shock.mean())

    if true_df is not None:
        merge_keys = ["business_day", "hour"]
        available_keys = [k for k in merge_keys
                          if k in true_df.columns and k in pred_df.columns]
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

        metrics_df = merged.rename(columns={
            "rt_true": "rt_actual",
            "delta_true": "delta_target",
        })
        full = compute_full_metrics(metrics_df)
        result.update(full)

    return result


# ── High-level prediction pipeline ──────────────────────────────────

def run_prediction_v3(
    raw_df: pd.DataFrame,
    feature_config,
    model_config: TrendKnightV3Config,
    checkpoint_path: Path | str,
    *,
    target_day: pd.Timestamp,
    device: torch.device | str = "auto",
    batch_size: int = 64,
    visible_frame: pd.DataFrame | None = None,
    teacher_pred_df: pd.DataFrame | None = None,
    blend_mode: BlendMode = "deep_only",
    blend_weight: float = 0.3,
) -> tuple[pd.DataFrame, dict]:
    """Full V3 prediction pipeline.

    1. Build prediction dataset
    2. Load model from checkpoint
    3. Run prediction (with confidence + shock sensitivity)
    4. Optionally blend with teachers
    5. Compute metrics if ground truth available

    Args:
        raw_df: Raw data DataFrame
        feature_config: SGDFNet FeatureConfig
        model_config: V3 model config
        checkpoint_path: Path to trained model checkpoint
        target_day: Day to predict
        device: Compute device
        batch_size: Batch size for prediction
        visible_frame: Cutoff-safe frame (optional)
        teacher_pred_df: Teacher predictions for blending (optional)
        blend_mode: Blend mode
        blend_weight: Blend weight for teacher

    Returns:
        (prediction_df, metrics_dict)
    """
    from .dataset_v3 import build_predict_dataset_v3

    # Resolve device
    if isinstance(device, str):
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

    # Build dataset
    pred_ds, feature_cols = build_predict_dataset_v3(
        raw_df, feature_config,
        target_day=target_day,
        visible_frame=visible_frame,
        teacher_pred_df=teacher_pred_df,
        num_teachers=model_config.teacher_input_dim,
    )
    logger.info(f"V3 Predict dataset: {len(pred_ds)} days, {len(feature_cols)} features")

    # Load model
    model, _ = load_model_v3(checkpoint_path, model_config, device)

    # Predict
    pred_df = predict_delta_v3(model, pred_ds, device, batch_size=batch_size)

    # Filter to valid hours only for final output
    pred_df_valid = pred_df[pred_df["valid"] == True].copy()
    pred_df_valid = pred_df_valid.drop(columns=["valid"])

    # Blend if needed
    if blend_mode != "deep_only" and teacher_pred_df is not None:
        pred_df_valid = predict_with_blend_v3(
            pred_df_valid, teacher_pred_df,
            mode=blend_mode, blend_weight=blend_weight,
        )
    else:
        pred_df_valid["final_delta_pred"] = pred_df_valid["delta_pred"]
        pred_df_valid["final_rt_pred"] = pred_df_valid["rt_pred"]

    # Compute metrics if ground truth is available
    true_frame = raw_df.copy()
    if "rt_actual" not in true_frame.columns:
        from models.deep_sgdf_delta import sgdfnet_bridge as _bridge
        _bridge.lazy_import()
        from sgdfnet.data_contract import RT_COL, DA_COL
        true_frame["rt_actual"] = pd.to_numeric(true_frame[RT_COL], errors="coerce")
        true_frame["da_anchor"] = pd.to_numeric(true_frame[DA_COL], errors="coerce")
        true_frame["delta_target"] = true_frame["rt_actual"] - true_frame["da_anchor"]
    if "business_day" not in true_frame.columns:
        from models.deep_sgdf_delta import sgdfnet_bridge as _bridge
        _bridge.lazy_import()
        from sgdfnet.data_contract import add_business_time_columns
        true_frame = add_business_time_columns(true_frame)

    true_day = true_frame[true_frame["business_day"] == target_day].copy()
    if not true_day.empty and "rt_actual" in true_day.columns:
        true_day = true_day[["business_day", "target_hour", "rt_actual", "delta_target"]].rename(
            columns={"target_hour": "hour"}
        )
        true_day["hour"] = true_day["hour"].astype(int)
        metrics = compute_v3_metrics(pred_df_valid, true_day)
    else:
        metrics = compute_v3_metrics(pred_df_valid)

    return pred_df_valid, metrics


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    """CLI entry point for V3 prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict with TrendKnight-X v3")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to raw data file")
    parser.add_argument("--target-day", type=str, required=True,
                        help="Target day (YYYY-MM-DD)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save predictions CSV")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--blend-mode", type=str, default="deep_only",
                        choices=["deep_only", "teacher_weighted", "teacher_residual"])
    parser.add_argument("--blend-weight", type=float, default=0.3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from models.deep_sgdf_delta import sgdfnet_bridge as _bridge
    _bridge.lazy_import()
    from sgdfnet.data_contract import load_dataset

    raw_df = load_dataset(args.data_path)
    target_day = pd.Timestamp(args.target_day)

    # Load config from checkpoint
    model, feature_cols = load_model_v3(args.checkpoint, device=args.device)
    model_config = model.config

    from .dataset_v3 import DEFAULT_FEATURE_CONFIG

    pred_df, metrics = run_prediction_v3(
        raw_df, DEFAULT_FEATURE_CONFIG, model_config,
        checkpoint_path=args.checkpoint,
        target_day=target_day,
        device=args.device,
        batch_size=args.batch_size,
        blend_mode=args.blend_mode,
        blend_weight=args.blend_weight,
    )

    # Save predictions
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Predictions saved to {out_path}")

    # Print metrics
    print("\n=== TrendKnight-X v3 Prediction Results ===")
    print(f"Days predicted: {metrics.get('num_days', 'N/A')}")
    print(f"Hours predicted: {metrics.get('num_hours', 'N/A')}")
    if "smape_floor50" in metrics:
        print(f"sMAPE_floor50: {metrics['smape_floor50']:.4f}")
    if "delta_mae" in metrics:
        print(f"Delta MAE: {metrics['delta_mae']:.4f}")
    if "mean_confidence" in metrics:
        print(f"Mean confidence: {metrics['mean_confidence']:.4f}")
    if "mean_shock_sensitivity" in metrics:
        print(f"Mean shock sensitivity: {metrics['mean_shock_sensitivity']:.4f}")

    return pred_df, metrics


if __name__ == "__main__":
    main()
