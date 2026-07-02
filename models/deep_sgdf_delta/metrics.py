"""Metrics for DeepSGDFDelta — sMAPE_floor50 and derived KPIs.

All metrics operate on *realtime price* (rt_actual vs rt_pred), not on delta.
The floor-50 capping matches the business metric used across the project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smape_floor50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 50.0,
    eps: float = 1e-6,
) -> float:
    """sMAPE with floor-50 capping on both y_true and y_pred."""
    yt = np.where(y_true < floor, floor, y_true)
    yp = np.where(y_pred < floor, floor, y_pred)
    denom = np.abs(yt) + np.abs(yp) + eps
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


def smape_floor50_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 50.0,
    eps: float = 1e-6,
) -> float:
    """Same as smape_floor50 but guaranteed non-negative for loss usage."""
    return smape_floor50(y_true, y_pred, floor=floor, eps=eps)


def delta_mae(y_true_delta: np.ndarray, y_pred_delta: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred_delta - y_true_delta)))


def compute_period_mask(hours: np.ndarray, period: str) -> np.ndarray:
    """Return boolean mask for a period label like '1_8', '9_16', '17_24'."""
    if period == "1_8":
        return (hours >= 1) & (hours <= 8)
    if period == "9_16":
        return (hours >= 9) & (hours <= 16)
    if period == "17_24":
        return (hours >= 17) & (hours <= 24)
    raise ValueError(f"Unknown period: {period}")


def classify_spike(y_true: np.ndarray, threshold: float = 500.0) -> np.ndarray:
    """Simple spike label: |y_true| > threshold."""
    return np.abs(y_true) > threshold


def classify_negative(y_true: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Negative price label: y_true < floor."""
    return y_true < floor


def compute_full_metrics(
    df: pd.DataFrame,
    *,
    spike_threshold: float = 500.0,
) -> dict:
    """Compute comprehensive metrics from a prediction DataFrame.

    Required columns: rt_actual, rt_pred, delta_target, delta_pred, hour, da_anchor
    """
    valid = df.dropna(subset=["rt_actual", "rt_pred", "delta_target", "delta_pred"]).copy()
    if valid.empty:
        return {"overall_sMAPE_floor50": float("nan"), "rows_total": 0, "rows_missing": 0}

    hours = valid["hour"].to_numpy(dtype=int)
    yt = valid["rt_actual"].to_numpy(dtype=float)
    yp = valid["rt_pred"].to_numpy(dtype=float)
    dt = valid["delta_target"].to_numpy(dtype=float)
    dp = valid["delta_pred"].to_numpy(dtype=float)

    result: dict = {}
    result["overall_sMAPE_floor50"] = smape_floor50(yt, yp)
    result["delta_mae"] = delta_mae(dt, dp)
    result["rows_total"] = len(valid)
    result["rows_missing"] = int(df.shape[0] - len(valid))

    # Segment metrics
    for period in ("1_8", "9_16", "17_24"):
        mask = compute_period_mask(hours, period)
        if mask.sum() > 0:
            result[f"{period}_sMAPE_floor50"] = smape_floor50(yt[mask], yp[mask])
        else:
            result[f"{period}_sMAPE_floor50"] = float("nan")

    # Normal trend (exclude spike and negative)
    spike_mask = classify_spike(yt, spike_threshold)
    neg_mask = classify_negative(yt)
    normal_mask = ~spike_mask & ~neg_mask
    if normal_mask.sum() > 0:
        result["normal_sMAPE_floor50"] = smape_floor50(yt[normal_mask], yp[normal_mask])
    else:
        result["normal_sMAPE_floor50"] = float("nan")

    # Spike metrics
    if spike_mask.sum() > 0:
        result["spike_sMAPE_floor50"] = smape_floor50(yt[spike_mask], yp[spike_mask])
        result["spike_count"] = int(spike_mask.sum())
    else:
        result["spike_sMAPE_floor50"] = float("nan")
        result["spike_count"] = 0

    # Negative price metrics
    if neg_mask.sum() > 0:
        result["negative_sMAPE_floor50"] = smape_floor50(yt[neg_mask], yp[neg_mask])
        result["negative_count"] = int(neg_mask.sum())
    else:
        result["negative_sMAPE_floor50"] = float("nan")
        result["negative_count"] = 0

    return result


def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-month sMAPE_floor50 from a prediction DataFrame.

    Requires columns: rt_actual, rt_pred, target_month (or business_day)
    """
    valid = df.dropna(subset=["rt_actual", "rt_pred"]).copy()
    if valid.empty:
        return pd.DataFrame()

    if "target_month" not in valid.columns:
        if "business_day" in valid.columns:
            valid["target_month"] = pd.to_datetime(valid["business_day"]).dt.to_period("M").astype(str)
        elif "ds" in valid.columns:
            valid["target_month"] = pd.to_datetime(valid["ds"]).dt.to_period("M").astype(str)
        else:
            valid["target_month"] = "unknown"

    rows = []
    for month, grp in valid.groupby("target_month"):
        yt = grp["rt_actual"].to_numpy(dtype=float)
        yp = grp["rt_pred"].to_numpy(dtype=float)
        rows.append({
            "month": month,
            "sMAPE_floor50": smape_floor50(yt, yp),
            "count": len(grp),
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
