"""Intraday Adaptive Residual Tracker — Phase 9.

Uses same-day observed residuals (from hours that have already occurred)
to estimate corrections for future hours within the 9_16 segment.

This module is strictly for INTRADAY mode only. It must NOT be used for
full-day day-ahead prediction.

Core functions:
  - compute_intraday_residual_state: summarize observed residuals into a state
  - predict_intraday_correction: predict correction for future hours
  - apply_intraday_correction: apply correction with guardrails
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntradayTrackerConfig:
    """Configuration for intraday residual tracker."""
    max_abs_correction: float = 80.0
    min_observed_hours: int = 2
    negative_price_guardrail: bool = True
    negative_price_threshold: float = 0.0
    negative_price_weight: float = 0.3
    apply_only_future_hours: bool = True
    # EWM span for exponential weighted mean
    ewm_span: float = 3.0
    # Distance decay: correction weight decreases for hours far from cutoff
    distance_decay: float = 0.1
    # Confidence threshold: below this, don't apply correction
    min_confidence: float = 0.2


@dataclass
class IntradayResidualState:
    """State computed from observed residuals within a business day."""
    business_day: pd.Timestamp
    cutoff_hour: int
    n_observed: int
    mean_residual_today: float
    median_residual_today: float
    ewm_residual_today: float
    last_residual: float
    residual_std_today: float
    bias_direction: str  # "positive", "negative", "mixed", "insufficient"
    confidence: float
    observed_hours: list[int] = field(default_factory=list)
    observed_residuals: list[float] = field(default_factory=list)


def compute_intraday_residual_state(
    observed_rows: pd.DataFrame,
    business_day: pd.Timestamp | str,
    cutoff_hour: int,
) -> IntradayResidualState:
    """Compute residual state from observed hours.

    Parameters
    ----------
    observed_rows : pd.DataFrame
        Must contain: hour_business, sgdfnet_pred, rt_actual.
        Only rows with hour_business <= cutoff_hour should be included.
    business_day : pd.Timestamp or str
        The business day.
    cutoff_hour : int
        The cutoff hour (hours <= cutoff are observed).

    Returns
    -------
    IntradayResidualState
    """
    bd = pd.Timestamp(business_day)
    df = observed_rows.copy()
    df["hour_business"] = df["hour_business"].astype(int)

    # Filter to observed hours only
    df = df[df["hour_business"] <= cutoff_hour].copy()
    df = df.sort_values("hour_business")

    # Compute residual = rt_actual - sgdfnet_pred
    df["residual"] = df["rt_actual"] - df["sgdfnet_pred"]

    # Drop NaN residuals
    df = df.dropna(subset=["residual"])
    n_observed = len(df)

    if n_observed == 0:
        return IntradayResidualState(
            business_day=bd,
            cutoff_hour=cutoff_hour,
            n_observed=0,
            mean_residual_today=0.0,
            median_residual_today=0.0,
            ewm_residual_today=0.0,
            last_residual=0.0,
            residual_std_today=0.0,
            bias_direction="insufficient",
            confidence=0.0,
        )

    residuals = df["residual"].values
    hours = df["hour_business"].values

    mean_res = float(np.mean(residuals))
    median_res = float(np.median(residuals))
    std_res = float(np.std(residuals, ddof=1)) if n_observed > 1 else 0.0
    last_res = float(residuals[-1])

    # EWM: exponential weighted mean (more weight on recent hours)
    ewm_series = pd.Series(residuals).ewm(span=3.0, adjust=False).mean()
    ewm_res = float(ewm_series.iloc[-1])

    # Bias direction
    if n_observed < 2:
        bias_dir = "insufficient"
    elif np.mean(residuals > 0) > 0.7:
        bias_dir = "positive"
    elif np.mean(residuals < 0) > 0.7:
        bias_dir = "negative"
    else:
        bias_dir = "mixed"

    # Confidence: based on number of observations and consistency
    # More observations → higher confidence (up to a point)
    n_factor = min(1.0, n_observed / 6.0)  # saturates at 6 observed hours
    # Lower std → higher confidence
    std_factor = max(0.0, 1.0 - std_res / 200.0) if std_res > 0 else 0.5
    # Consistency of direction
    if bias_dir in ("positive", "negative"):
        dir_factor = 0.8
    elif bias_dir == "mixed":
        dir_factor = 0.3
    else:
        dir_factor = 0.0
    confidence = float(n_factor * 0.5 + std_factor * 0.3 + dir_factor * 0.2)
    confidence = max(0.0, min(1.0, confidence))

    return IntradayResidualState(
        business_day=bd,
        cutoff_hour=cutoff_hour,
        n_observed=n_observed,
        mean_residual_today=mean_res,
        median_residual_today=median_res,
        ewm_residual_today=ewm_res,
        last_residual=last_res,
        residual_std_today=std_res,
        bias_direction=bias_dir,
        confidence=confidence,
        observed_hours=hours.tolist(),
        observed_residuals=residuals.tolist(),
    )


def predict_intraday_correction(
    future_rows: pd.DataFrame,
    state: IntradayResidualState,
    config: Optional[IntradayTrackerConfig] = None,
) -> pd.DataFrame:
    """Predict correction for future hours based on observed state.

    Parameters
    ----------
    future_rows : pd.DataFrame
        Must contain: hour_business, sgdfnet_pred.
        Only hours > cutoff_hour should be included.
    state : IntradayResidualState
        State from compute_intraday_residual_state.
    config : IntradayTrackerConfig, optional
        Tracker configuration.

    Returns
    -------
    pd.DataFrame with added columns:
        - intraday_raw_correction: unguarded correction
        - intraday_correction_weight: weight applied
        - intraday_correction: guarded correction
        - intraday_corrected_pred: sgdfnet_pred + correction
    """
    if config is None:
        config = IntradayTrackerConfig()

    df = future_rows.copy()

    if state.n_observed < config.min_observed_hours:
        # Not enough observed data — no correction
        df["intraday_raw_correction"] = 0.0
        df["intraday_correction_weight"] = 0.0
        df["intraday_correction"] = 0.0
        df["intraday_corrected_pred"] = df["sgdfnet_pred"]
        return df

    # Base correction: weighted combination of state signals
    # Weights: mean=0.4, ewm=0.35, last=0.25
    base_correction = (
        0.40 * state.mean_residual_today
        + 0.35 * state.ewm_residual_today
        + 0.25 * state.last_residual
    )

    # Per-hour adjustments
    corrections = []
    weights = []
    for _, row in df.iterrows():
        target_hour = int(row["hour_business"])
        distance = target_hour - state.cutoff_hour  # hours ahead

        # Distance decay: further from cutoff → less confident
        decay_weight = np.exp(-config.distance_decay * distance)

        # Std penalty: high variability → less correction
        std_penalty = max(0.3, 1.0 - state.residual_std_today / 300.0)

        # Combined weight
        w = state.confidence * decay_weight * std_penalty
        w = max(0.0, min(1.0, w))

        corrections.append(base_correction * w)
        weights.append(w)

    df["intraday_raw_correction"] = corrections
    df["intraday_correction_weight"] = weights

    # Clip correction magnitude
    clipped = np.clip(corrections, -config.max_abs_correction, config.max_abs_correction)
    df["intraday_correction"] = clipped
    df["intraday_corrected_pred"] = df["sgdfnet_pred"].values + clipped

    return df


def apply_intraday_correction(
    future_rows: pd.DataFrame,
    state: IntradayResidualState,
    config: Optional[IntradayTrackerConfig] = None,
) -> pd.DataFrame:
    """Apply intraday correction with full guardrail.

    Combines predict_intraday_correction with additional guardrails:
    - Negative price guardrail
    - Only future hours
    - Minimum observed hours check

    Parameters
    ----------
    future_rows : pd.DataFrame
        Must contain: hour_business, sgdfnet_pred, da_anchor (optional for neg guardrail).
    state : IntradayResidualState
        State from compute_intraday_residual_state.
    config : IntradayTrackerConfig, optional

    Returns
    -------
    pd.DataFrame with correction columns plus:
        - guardrail_reason: why guardrail was applied (empty if none)
    """
    if config is None:
        config = IntradayTrackerConfig()

    df = predict_intraday_correction(future_rows, state, config)
    n = len(df)
    reasons = [""] * n
    weight = df["intraday_correction_weight"].values.copy()

    # Guardrail 1: only future hours (target > cutoff)
    if config.apply_only_future_hours:
        past_mask = df["hour_business"].values <= state.cutoff_hour
        weight[past_mask] = 0.0
        for i in range(n):
            if past_mask[i] and reasons[i] == "":
                reasons[i] = "past_hour_not_corrected"

    # Guardrail 2: negative price risk
    if config.negative_price_guardrail and "da_anchor" in df.columns:
        neg_mask = df["da_anchor"].fillna(0).values < config.negative_price_threshold
        weight[neg_mask] *= config.negative_price_weight
        for i in range(n):
            if neg_mask[i] and reasons[i] == "":
                reasons[i] = "negative_price_risk"

    # Apply weight
    raw_corr = df["intraday_raw_correction"].values
    final_corr = raw_corr * weight
    final_corr = np.clip(final_corr, -config.max_abs_correction, config.max_abs_correction)

    # Confidence floor
    if state.confidence < config.min_confidence:
        final_corr[:] = 0.0
        for i in range(n):
            if reasons[i] == "":
                reasons[i] = f"low_confidence_{state.confidence:.2f}"

    df["intraday_correction_weight"] = weight
    df["intraday_correction"] = final_corr
    df["intraday_corrected_pred"] = df["sgdfnet_pred"].values + final_corr
    df["guardrail_reason"] = reasons

    return df
