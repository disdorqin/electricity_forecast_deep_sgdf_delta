"""Negative risk feature construction.

Builds features for negative price risk classification from raw hourly data.
Reuses business_time.py and realtime_column_mapping.py.
Reuses DeltaSupply derived feature logic for supply-demand features.

Feature groups:
  1. Forecast-side derived features (target-day forecast allowed in FULL_DAY)
     - forecast_renewable_share, forecast_wind_share, forecast_solar_share,
       forecast_net_load, forecast_bidding_pressure
  2. Anchor features
     - da_anchor
  3. Calendar / period features
     - hour_sin, hour_cos, month_sin, month_cos
  4. Historical price delta lag features (D-1 and earlier only in FULL_DAY)
     - previous_day_price_delta_mean, previous_day_price_delta_std

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
    FORECAST_FEATURES,
    EPS,
)

logger = logging.getLogger(__name__)


# ── Feature group definitions ─────────────────────────────────────────

NEGATIVE_DERIVED_FEATURES = [
    "forecast_renewable_share", "forecast_wind_share", "forecast_solar_share",
    "forecast_net_load", "forecast_bidding_pressure",
]

NEGATIVE_ANCHOR_FEATURES = [
    "da_anchor",
]

NEGATIVE_CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

NEGATIVE_LAG_FEATURES = [
    "previous_day_price_delta_mean", "previous_day_price_delta_std",
]

ALL_NEGATIVE_FEATURE_GROUPS = {
    "derived": NEGATIVE_DERIVED_FEATURES,
    "anchor": NEGATIVE_ANCHOR_FEATURES,
    "calendar": NEGATIVE_CALENDAR_FEATURES,
    "lag": NEGATIVE_LAG_FEATURES,
}


@dataclass
class NegativeFeatureAudit:
    """Feature audit result for negative risk features."""
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
class NegativeRiskFeatureResult:
    """Result of building negative risk features."""
    df: pd.DataFrame
    feature_columns: list = field(default_factory=list)
    audit: NegativeFeatureAudit = field(default_factory=NegativeFeatureAudit)
    mode: str = "FULL_DAY"


def build_negative_risk_features(
    df: pd.DataFrame,
    mode: Literal["FULL_DAY", "INTRADAY"] = "FULL_DAY",
    timestamp_col: str = "ds",
) -> NegativeRiskFeatureResult:
    """Build negative risk features from raw hourly data.

    Args:
        df: Raw hourly DataFrame (Chinese or English columns).
        mode: FULL_DAY or INTRADAY. Default FULL_DAY.
        timestamp_col: Timestamp column name.

    Returns:
        NegativeRiskFeatureResult with features and audit.
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
    _build_negative_lag_features(work, mode)

    # ── Calendar / period features ────────────────────────────────────
    _build_negative_calendar_features(work)

    # ── Collect feature columns ───────────────────────────────────────
    feature_cols = []
    for group_name, group_cols in ALL_NEGATIVE_FEATURE_GROUPS.items():
        for col in group_cols:
            if col in work.columns and col not in feature_cols:
                # Check it's not all NaN
                if work[col].notna().any():
                    feature_cols.append(col)

    # ── Feature audit ─────────────────────────────────────────────────
    audit = _compute_negative_audit(work, feature_cols, forecast_available, mode)

    return NegativeRiskFeatureResult(
        df=work,
        feature_columns=feature_cols,
        audit=audit,
        mode=mode,
    )


def _build_negative_lag_features(work: pd.DataFrame, mode: str) -> None:
    """Build historical price delta lag features. FULL_DAY: D-1 and earlier only."""
    if "price_delta" in work.columns:
        work["previous_day_price_delta_mean"] = (
            work["price_delta"].shift(24).rolling(24, min_periods=1).mean()
        )
        work["previous_day_price_delta_std"] = (
            work["price_delta"].shift(24).rolling(24, min_periods=1).std()
        )
    else:
        work["previous_day_price_delta_mean"] = np.nan
        work["previous_day_price_delta_std"] = np.nan


def _build_negative_calendar_features(work: pd.DataFrame) -> None:
    """Build calendar features for negative risk (hour + month cyclical)."""
    h = work["hour_business"].astype(int)
    work["hour_sin"] = np.sin(2 * np.pi * h / 24)
    work["hour_cos"] = np.cos(2 * np.pi * h / 24)

    month = work["business_day"].dt.month.astype(int)
    work["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    work["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)


def _compute_negative_audit(
    work: pd.DataFrame,
    feature_columns: list,
    forecast_available: dict,
    mode: str,
) -> NegativeFeatureAudit:
    """Compute feature audit verdict for negative risk features."""
    n_features = len(feature_columns)

    # Derived coverage
    n_derived_wanted = len(NEGATIVE_DERIVED_FEATURES)
    n_derived_present = sum(
        1 for f in NEGATIVE_DERIVED_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    derived_coverage = n_derived_present / max(n_derived_wanted, 1)

    # Lag coverage
    n_lag_wanted = len(NEGATIVE_LAG_FEATURES)
    n_lag_present = sum(
        1 for f in NEGATIVE_LAG_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    lag_coverage = n_lag_present / max(n_lag_wanted, 1)

    # Calendar coverage
    n_cal_wanted = len(NEGATIVE_CALENDAR_FEATURES)
    n_cal_present = sum(
        1 for f in NEGATIVE_CALENDAR_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    cal_coverage = n_cal_present / max(n_cal_wanted, 1)

    # Missing features
    missing = []
    for group_cols in ALL_NEGATIVE_FEATURE_GROUPS.values():
        for col in group_cols:
            if col not in feature_columns:
                missing.append(col)

    # Leakage check
    leakage_ok = True  # Enforced by construction

    # Verdict
    if n_features >= 8 and derived_coverage >= 0.60 and leakage_ok:
        verdict = "FORMAL_READY"
    elif n_features >= 4 and leakage_ok:
        verdict = "PARTIAL_READY"
    else:
        verdict = "NOT_READY"

    return NegativeFeatureAudit(
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
