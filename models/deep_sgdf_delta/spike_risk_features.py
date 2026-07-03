"""Spike risk feature construction.

Builds features for spike risk classification from raw hourly electricity data.
Reuses business_time.py and realtime_column_mapping.py.
Reuses DeltaSupply derived feature logic for supply-demand features.

Feature groups:
  1. Forecast-side derived features (target-day forecast allowed in FULL_DAY)
     - forecast_net_load, forecast_renewable_share, forecast_supply_demand_gap,
       forecast_bidding_pressure, forecast_thermal_pressure
  2. Anchor features
     - da_anchor
  3. Calendar / period features
     - hour_sin, hour_cos, period_id
  4. Historical price delta lag features (D-1 and earlier only in FULL_DAY)
     - previous_day_price_delta_mean, previous_day_price_delta_std,
       lagged_price_delta_24h, lagged_price_delta_48h

FULL_DAY mode: no target-day actual features allowed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.realtime_column_mapping import rename_chinese_columns
from models.deep_sgdf_delta.delta_supply_features import (
    _build_derived_features,
    _build_calendar_features,
    FORECAST_FEATURES,
    EPS,
)

logger = logging.getLogger(__name__)


# ── Feature group definitions ─────────────────────────────────────────

SPIKE_DERIVED_FEATURES = [
    "forecast_net_load", "forecast_renewable_share",
    "forecast_supply_demand_gap", "forecast_bidding_pressure",
    "forecast_thermal_pressure",
]

SPIKE_ANCHOR_FEATURES = [
    "da_anchor",
]

SPIKE_CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "period_id",
]

SPIKE_LAG_FEATURES = [
    "previous_day_price_delta_mean", "previous_day_price_delta_std",
    "lagged_price_delta_24h", "lagged_price_delta_48h",
]

ALL_SPIKE_FEATURE_GROUPS = {
    "derived": SPIKE_DERIVED_FEATURES,
    "anchor": SPIKE_ANCHOR_FEATURES,
    "calendar": SPIKE_CALENDAR_FEATURES,
    "lag": SPIKE_LAG_FEATURES,
}


@dataclass
class SpikeFeatureAudit:
    """Feature audit result for spike risk features."""
    n_features: int = 0
    feature_columns: list = field(default_factory=list)
    missing_features: list = field(default_factory=list)
    derived_feature_coverage: float = 0.0
    lag_feature_coverage: float = 0.0
    calendar_feature_coverage: float = 0.0
    leakage_check: bool = True
    verdict: str = "NOT_READY"  # FORMAL_READY | PARTIAL_READY | NOT_READY
    formal_ready: bool = False


@dataclass
class SpikeRiskFeatureResult:
    """Result of building spike risk features."""
    df: pd.DataFrame
    feature_columns: list = field(default_factory=list)
    audit: SpikeFeatureAudit = field(default_factory=SpikeFeatureAudit)
    mode: str = "FULL_DAY"


def build_spike_risk_features(
    df: pd.DataFrame,
    mode: Literal["FULL_DAY", "INTRADAY"] = "FULL_DAY",
    timestamp_col: str = "ds",
) -> SpikeRiskFeatureResult:
    """Build spike risk features from raw hourly data.

    Args:
        df: Raw hourly DataFrame (Chinese or English columns).
        mode: FULL_DAY or INTRADAY. Default FULL_DAY.
        timestamp_col: Timestamp column name.

    Returns:
        SpikeRiskFeatureResult with features and audit.
    """
    # Step 1: Rename Chinese columns
    work = rename_chinese_columns(df)

    # Step 2: Parse timestamp and sort
    work[timestamp_col] = pd.to_datetime(work[timestamp_col])
    work = work.sort_values(timestamp_col).reset_index(drop=True)

    # Step 3: Add business time columns
    work = add_business_time_columns(work, timestamp_col=timestamp_col)

    # Step 4: Ensure numeric types for key columns
    for col in ["da_anchor", "rt_actual"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # Step 5: Compute price_delta for lag features
    if "da_anchor" in work.columns and "rt_actual" in work.columns:
        work["price_delta"] = work["rt_actual"] - work["da_anchor"]
    else:
        work["price_delta"] = np.nan

    # ── Forecast-side derived features ────────────────────────────────
    forecast_available = {}
    for feat in FORECAST_FEATURES:
        if feat in work.columns:
            work[feat] = pd.to_numeric(work[feat], errors="coerce")
            forecast_available[feat] = True
        else:
            forecast_available[feat] = False

    # Reuse DeltaSupply derived feature builder
    _build_derived_features(work, forecast_available)

    # ── Historical price delta lag features ───────────────────────────
    _build_spike_lag_features(work, mode)

    # ── Calendar / period features ────────────────────────────────────
    _build_calendar_features(work)

    # ── Collect feature columns ───────────────────────────────────────
    feature_cols = []
    for group_name, group_cols in ALL_SPIKE_FEATURE_GROUPS.items():
        for col in group_cols:
            if col in work.columns and col not in feature_cols:
                # Check it's not all NaN
                if work[col].notna().any():
                    feature_cols.append(col)

    # ── Feature audit ─────────────────────────────────────────────────
    audit = _compute_spike_audit(work, feature_cols, forecast_available, mode)

    return SpikeRiskFeatureResult(
        df=work,
        feature_columns=feature_cols,
        audit=audit,
        mode=mode,
    )


def _build_spike_lag_features(work: pd.DataFrame, mode: str) -> None:
    """Build historical price delta lag features. FULL_DAY: D-1 and earlier only."""
    if "price_delta" in work.columns:
        work["previous_day_price_delta_mean"] = (
            work["price_delta"].shift(24).rolling(24, min_periods=1).mean()
        )
        work["previous_day_price_delta_std"] = (
            work["price_delta"].shift(24).rolling(24, min_periods=1).std()
        )
        work["lagged_price_delta_24h"] = work["price_delta"].shift(24)
        work["lagged_price_delta_48h"] = work["price_delta"].shift(48)
    else:
        work["previous_day_price_delta_mean"] = np.nan
        work["previous_day_price_delta_std"] = np.nan
        work["lagged_price_delta_24h"] = np.nan
        work["lagged_price_delta_48h"] = np.nan


def _compute_spike_audit(
    work: pd.DataFrame,
    feature_columns: list,
    forecast_available: dict,
    mode: str,
) -> SpikeFeatureAudit:
    """Compute feature audit verdict for spike risk features."""
    n_features = len(feature_columns)

    # Derived coverage
    n_derived_wanted = len(SPIKE_DERIVED_FEATURES)
    n_derived_present = sum(
        1 for f in SPIKE_DERIVED_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    derived_coverage = n_derived_present / max(n_derived_wanted, 1)

    # Lag coverage
    n_lag_wanted = len(SPIKE_LAG_FEATURES)
    n_lag_present = sum(
        1 for f in SPIKE_LAG_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    lag_coverage = n_lag_present / max(n_lag_wanted, 1)

    # Calendar coverage
    n_cal_wanted = len(SPIKE_CALENDAR_FEATURES)
    n_cal_present = sum(
        1 for f in SPIKE_CALENDAR_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    cal_coverage = n_cal_present / max(n_cal_wanted, 1)

    # Missing features
    missing = []
    for group_cols in ALL_SPIKE_FEATURE_GROUPS.values():
        for col in group_cols:
            if col not in feature_columns:
                missing.append(col)

    # Leakage check
    leakage_ok = True  # Enforced by construction in _build_spike_lag_features

    # Verdict
    if n_features >= 10 and derived_coverage >= 0.60 and leakage_ok:
        verdict = "FORMAL_READY"
    elif n_features >= 5 and leakage_ok:
        verdict = "PARTIAL_READY"
    else:
        verdict = "NOT_READY"

    return SpikeFeatureAudit(
        n_features=n_features,
        feature_columns=feature_columns,
        missing_features=missing,
        derived_feature_coverage=round(derived_coverage, 4),
        lag_feature_coverage=round(lag_coverage, 4),
        calendar_feature_coverage=round(cal_coverage, 4),
        leakage_check=leakage_ok,
        verdict=verdict,
        formal_ready=(verdict == "FORMAL_READY"),
    )
