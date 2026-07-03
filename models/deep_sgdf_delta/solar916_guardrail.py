"""Solar916 Guardrail — Phase 8.

Applies safety guardrails to Solar916 correction predictions:
1. Only triggers for 9_16 period hours (9-16)
2. Reduces/disables correction for negative price risk
3. Hour-specific disable for hours that worsen after no-leak retraining
4. Clips correction magnitude to max_abs_correction
5. Reduces weight when features are missing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Solar916GuardrailConfig:
    """Configuration for Solar916 guardrail."""
    # Max absolute correction (clips residual prediction)
    max_abs_correction: float = 100.0
    # Hours to disable correction for (e.g., if they worsened in no-leak eval)
    disabled_hours: list[int] = field(default_factory=list)
    # Hours to reduce weight for (e.g., if they slightly worsened)
    reduced_weight_hours: dict[int, float] = field(default_factory=dict)
    # Weight multiplier when features are missing
    missing_feature_weight: float = 0.5
    # Weight multiplier for negative price risk (price < 0 or very low)
    negative_risk_weight: float = 0.3
    # Threshold below which negative risk is triggered
    negative_risk_threshold: float = 0.0


def apply_guardrail(
    predictions: pd.DataFrame,
    config: Optional[Solar916GuardrailConfig] = None,
) -> pd.DataFrame:
    """Apply guardrail to Solar916 correction predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must contain: hour_business, solar916_residual_pred, sgdfnet_pred,
                      feature_missing_flag (optional)
    config : Solar916GuardrailConfig, optional
        Guardrail configuration. Uses defaults if None.

    Returns
    -------
    pd.DataFrame with additional columns:
        - solar916_raw_residual_pred: original residual prediction
        - solar916_guardrail_weight: weight applied (0 to 1)
        - solar916_residual_pred_after_guardrail: clipped + weighted residual
        - solar916_corrected_pred: sgdfnet_pred + guarded residual
        - guardrail_reason: why guardrail was applied
    """
    if config is None:
        config = Solar916GuardrailConfig()

    df = predictions.copy()
    n = len(df)

    # Store raw prediction
    df["solar916_raw_residual_pred"] = df["solar916_residual_pred"].copy()

    # Initialize weight to 1.0
    weight = np.ones(n, dtype=float)
    reasons = [""] * n

    # Rule 1: Only apply to 9_16 hours
    if "period" in df.columns:
        non_916 = df["period"] != "9_16"
        weight[non_916.values] = 0.0
        for i in range(n):
            if non_916.iloc[i]:
                reasons[i] = "outside_9_16_period"

    # Rule 2: Disable for configured hours
    for hour in config.disabled_hours:
        mask = df["hour_business"] == hour
        weight[mask.values] = 0.0
        for i in range(n):
            if mask.iloc[i] and reasons[i] == "":
                reasons[i] = f"hour_{hour}_disabled"

    # Rule 3: Reduce weight for configured hours
    for hour, w in config.reduced_weight_hours.items():
        mask = df["hour_business"] == hour
        weight[mask.values] = np.minimum(weight[mask.values], w)
        for i in range(n):
            if mask.iloc[i] and reasons[i] == "":
                reasons[i] = f"hour_{hour}_reduced_weight_{w}"

    # Rule 4: Negative price risk
    if "da_anchor" in df.columns:
        neg_risk = df["da_anchor"].fillna(0) < config.negative_risk_threshold
        weight[neg_risk.values] = np.minimum(weight[neg_risk.values], config.negative_risk_weight)
        for i in range(n):
            if neg_risk.iloc[i] and reasons[i] == "":
                reasons[i] = "negative_price_risk"

    # Rule 5: Missing features
    if "feature_missing_flag" in df.columns:
        missing = df["feature_missing_flag"].fillna(False).astype(bool)
        weight[missing.values] = np.minimum(weight[missing.values], config.missing_feature_weight)
        for i in range(n):
            if missing.iloc[i] and reasons[i] == "":
                reasons[i] = "missing_features"

    # Apply weight
    raw_pred = df["solar916_raw_residual_pred"].fillna(0).values
    guarded_pred = raw_pred * weight

    # Clip magnitude
    guarded_pred = np.clip(guarded_pred, -config.max_abs_correction, config.max_abs_correction)

    df["solar916_guardrail_weight"] = weight
    df["solar916_residual_pred_after_guardrail"] = guarded_pred
    df["solar916_corrected_pred"] = df["sgdfnet_pred"].fillna(0).values + guarded_pred
    df["guardrail_reason"] = reasons

    return df


def compute_guarded_metrics(
    df: pd.DataFrame,
    rt_actual_col: str = "rt_actual",
) -> dict:
    """Compute metrics after guardrail application."""
    from models.deep_sgdf_delta.solar916_model import smape_floor50

    rt = df[rt_actual_col].values
    base_pred = df["sgdfnet_pred"].fillna(0).values
    corrected = df["solar916_corrected_pred"].values

    overall = {
        "baseline_smape": smape_floor50(rt, base_pred),
        "corrected_smape": smape_floor50(rt, corrected),
    }
    overall["improvement"] = overall["baseline_smape"] - overall["corrected_smape"]

    # Per-hour
    hourly = []
    for hour in sorted(df["hour_business"].unique()):
        mask = df["hour_business"] == hour
        if mask.sum() == 0:
            continue
        h_base = smape_floor50(rt[mask], base_pred[mask])
        h_corr = smape_floor50(rt[mask], corrected[mask])
        hourly.append({
            "hour": int(hour),
            "baseline_smape": h_base,
            "corrected_smape": h_corr,
            "improvement": h_base - h_corr,
        })

    # Per-bucket
    buckets = []
    for bucket_name, bucket_fn in [
        ("normal", lambda x: (np.abs(x) <= 500) & (x >= 0)),
        ("spike", lambda x: np.abs(x) > 500),
        ("negative", lambda x: x < 0),
    ]:
        mask = bucket_fn(rt)
        if mask.sum() == 0:
            continue
        b_base = smape_floor50(rt[mask], base_pred[mask])
        b_corr = smape_floor50(rt[mask], corrected[mask])
        buckets.append({
            "bucket": bucket_name,
            "baseline_smape": b_base,
            "corrected_smape": b_corr,
            "improvement": b_base - b_corr,
        })

    return {"overall": overall, "hourly": hourly, "buckets": buckets}
