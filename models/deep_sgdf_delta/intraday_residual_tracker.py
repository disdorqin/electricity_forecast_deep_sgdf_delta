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

    Correction pipeline (Phase 10 clarified naming):
      intraday_base_correction: unweighted base signal from state
      intraday_model_weight: confidence * distance_decay * std_penalty
      intraday_pre_guardrail_correction: base * model_weight

    Parameters
    ----------
    future_rows : pd.DataFrame
        Must contain: hour_business, sgdfnet_pred.
    state : IntradayResidualState
    config : IntradayTrackerConfig, optional

    Returns
    -------
    pd.DataFrame with columns:
        intraday_base_correction, intraday_model_weight,
        intraday_pre_guardrail_correction, intraday_corrected_pred (preliminary)
    """
    if config is None:
        config = IntradayTrackerConfig()

    df = future_rows.copy()

    if state.n_observed < config.min_observed_hours:
        df["intraday_base_correction"] = 0.0
        df["intraday_model_weight"] = 0.0
        df["intraday_pre_guardrail_correction"] = 0.0
        df["intraday_corrected_pred"] = df["sgdfnet_pred"]
        return df

    # Base correction: weighted combination of state signals (UNWEIGHTED by distance/confidence)
    base_correction = (
        0.40 * state.mean_residual_today
        + 0.35 * state.ewm_residual_today
        + 0.25 * state.last_residual
    )

    # Per-hour model weight
    model_weights = []
    pre_guardrail_corrections = []
    for _, row in df.iterrows():
        target_hour = int(row["hour_business"])
        distance = target_hour - state.cutoff_hour

        # Distance decay: further from cutoff → less confident
        decay_weight = np.exp(-config.distance_decay * distance)

        # Std penalty: high variability → less correction
        std_penalty = max(0.3, 1.0 - state.residual_std_today / 300.0)

        # Model weight = confidence * decay * std_penalty
        w = state.confidence * decay_weight * std_penalty
        w = max(0.0, min(1.0, w))

        model_weights.append(w)
        pre_guardrail_corrections.append(base_correction * w)

    df["intraday_base_correction"] = base_correction
    df["intraday_model_weight"] = model_weights
    df["intraday_pre_guardrail_correction"] = np.clip(
        pre_guardrail_corrections, -config.max_abs_correction, config.max_abs_correction
    )
    df["intraday_corrected_pred"] = (
        df["sgdfnet_pred"].values + df["intraday_pre_guardrail_correction"].values
    )

    return df


def apply_intraday_correction(
    future_rows: pd.DataFrame,
    state: IntradayResidualState,
    config: Optional[IntradayTrackerConfig] = None,
) -> pd.DataFrame:
    """Apply intraday correction with full guardrail.

    Correction pipeline (Phase 10):
      intraday_base_correction: unweighted base signal
      intraday_model_weight: confidence * distance_decay * std_penalty
      intraday_pre_guardrail_correction: base * model_weight
      intraday_guardrail_weight: negative/cutoff/confidence guardrail
      intraday_final_correction: pre_guardrail * guardrail_weight
      intraday_corrected_pred: sgdfnet_pred + final_correction
      intraday_correction: alias for intraday_final_correction (backward compat)

    Parameters
    ----------
    future_rows : pd.DataFrame
        Must contain: hour_business, sgdfnet_pred, da_anchor (optional).
    state : IntradayResidualState
    config : IntradayTrackerConfig, optional

    Returns
    -------
    pd.DataFrame with all correction columns plus guardrail_reason.
    """
    if config is None:
        config = IntradayTrackerConfig()

    df = predict_intraday_correction(future_rows, state, config)
    n = len(df)
    reasons = [""] * n
    guardrail_weight = np.ones(n, dtype=float)

    # Guardrail 1: only future hours (target > cutoff)
    if config.apply_only_future_hours:
        past_mask = df["hour_business"].values <= state.cutoff_hour
        guardrail_weight[past_mask] = 0.0
        for i in range(n):
            if past_mask[i] and reasons[i] == "":
                reasons[i] = "past_hour_not_corrected"

    # Guardrail 2: negative price risk
    if config.negative_price_guardrail and "da_anchor" in df.columns:
        neg_mask = df["da_anchor"].fillna(0).values < config.negative_price_threshold
        guardrail_weight[neg_mask] *= config.negative_price_weight
        for i in range(n):
            if neg_mask[i] and reasons[i] == "":
                reasons[i] = "negative_price_risk"

    # Confidence floor
    if state.confidence < config.min_confidence:
        guardrail_weight[:] = 0.0
        for i in range(n):
            if reasons[i] == "":
                reasons[i] = f"low_confidence_{state.confidence:.2f}"

    # Final correction = pre_guardrail * guardrail_weight
    pre_guardrail = df["intraday_pre_guardrail_correction"].values
    final_corr = pre_guardrail * guardrail_weight
    final_corr = np.clip(final_corr, -config.max_abs_correction, config.max_abs_correction)

    df["intraday_guardrail_weight"] = guardrail_weight
    df["intraday_final_correction"] = final_corr
    df["intraday_correction"] = final_corr  # backward compat alias
    df["intraday_corrected_pred"] = df["sgdfnet_pred"].values + final_corr
    df["guardrail_reason"] = reasons

    return df
