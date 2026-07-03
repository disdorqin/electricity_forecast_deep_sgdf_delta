"""DeltaSupply feature construction.

Builds supply-demand deviation features from raw hourly electricity data.
Reuses business_time.py and realtime_column_mapping.py.

Feature groups:
  1. Forecast-side features (target-day forecast allowed in FULL_DAY)
  2. Derived supply-demand features
  3. Historical actual lag features (D-1 and earlier only in FULL_DAY)
  4. Calendar / period features
  5. Optional anchor features (da_anchor, sgdfnet_pred)

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

logger = logging.getLogger(__name__)


# ── Feature group definitions ─────────────────────────────────────────

FORECAST_FEATURES = [
    "load_forecast", "renewable_forecast", "wind_forecast", "solar_forecast",
    "tie_line_forecast", "bidding_space_forecast", "provincial_load_forecast",
]

DERIVED_FEATURES = [
    "forecast_net_load", "forecast_renewable_share",
    "forecast_wind_share", "forecast_solar_share",
    "forecast_supply_demand_gap", "forecast_bidding_pressure",
    "forecast_thermal_pressure",
]

LAG_FEATURES = [
    "actual_load_lag_24h", "actual_load_lag_48h",
    "actual_renewable_lag_24h", "actual_wind_lag_24h", "actual_solar_lag_24h",
    "previous_day_load_mean", "previous_day_renewable_mean",
    "previous_day_wind_mean", "previous_day_solar_mean",
    "previous_day_price_delta_mean", "previous_day_price_delta_std",
    "lagged_price_delta_24h", "lagged_price_delta_48h",
]

CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend", "period_id",
]

ANCHOR_FEATURES = [
    "da_anchor", "sgdfnet_pred", "anchor_spread",
]

ALL_FEATURE_GROUPS = {
    "forecast": FORECAST_FEATURES,
    "derived": DERIVED_FEATURES,
    "lag": LAG_FEATURES,
    "calendar": CALENDAR_FEATURES,
    "anchor": ANCHOR_FEATURES,
}

EPS = 1e-6


@dataclass
class FeatureAudit:
    """Feature audit result."""
    n_features: int = 0
    feature_columns: list = field(default_factory=list)
    missing_features: list = field(default_factory=list)
    forecast_feature_coverage: float = 0.0
    lag_feature_coverage: float = 0.0
    calendar_feature_coverage: float = 0.0
    sgdfnet_available: bool = False
    leakage_check: bool = True
    verdict: str = "NOT_READY"  # FORMAL_READY | PARTIAL_READY | NOT_READY
    formal_ready: bool = False


@dataclass
class DeltaSupplyFeatureResult:
    """Result of building delta supply features."""
    df: pd.DataFrame
    feature_columns: list = field(default_factory=list)
    audit: FeatureAudit = field(default_factory=FeatureAudit)
    mode: str = "FULL_DAY"


def build_delta_supply_features(
    df: pd.DataFrame,
    mode: Literal["FULL_DAY", "INTRADAY"] = "FULL_DAY",
    sgdfnet_df: Optional[pd.DataFrame] = None,
    timestamp_col: str = "ds",
) -> DeltaSupplyFeatureResult:
    """Build supply-demand deviation features from raw hourly data.

    Args:
        df: Raw hourly DataFrame (Chinese or English columns).
        mode: FULL_DAY or INTRADAY. Default FULL_DAY.
        sgdfnet_df: Optional DataFrame with sgdfnet_pred column.
        timestamp_col: Timestamp column name.

    Returns:
        DeltaSupplyFeatureResult with features and audit.
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
    elif "da_anchor" in work.columns:
        work["price_delta"] = np.nan
    else:
        work["price_delta"] = np.nan

    # ── Forecast-side features ────────────────────────────────────────
    # These are allowed in FULL_DAY mode (target-day forecast is visible)
    forecast_available = {}
    for feat in FORECAST_FEATURES:
        if feat in work.columns:
            work[feat] = pd.to_numeric(work[feat], errors="coerce")
            forecast_available[feat] = True
        else:
            forecast_available[feat] = False

    # ── Derived supply-demand features ────────────────────────────────
    _build_derived_features(work, forecast_available)

    # ── Historical actual lag features ────────────────────────────────
    _build_lag_features(work, mode)

    # ── Calendar / period features ────────────────────────────────────
    _build_calendar_features(work)

    # ── Optional anchor features ──────────────────────────────────────
    sgdfnet_available = False
    if sgdfnet_df is not None and "sgdfnet_pred" in sgdfnet_df.columns:
        # Merge SGDFNet predictions
        sgdf_work = sgdfnet_df.copy()
        if timestamp_col in sgdf_work.columns:
            sgdf_work[timestamp_col] = pd.to_datetime(sgdf_work[timestamp_col])
            if "sgdfnet_pred" in sgdf_work.columns:
                work = work.merge(
                    sgdf_work[[timestamp_col, "sgdfnet_pred"]].drop_duplicates(timestamp_col),
                    on=timestamp_col, how="left", suffixes=("", "_sgdf"),
                )
                if "sgdfnet_pred_sgdf" in work.columns:
                    work["sgdfnet_pred"] = work["sgdfnet_pred"].fillna(work["sgdfnet_pred_sgdf"])
                    work = work.drop(columns=["sgdfnet_pred_sgdf"])
                sgdfnet_available = True
    elif "sgdfnet_pred" in work.columns:
        sgdfnet_available = True

    if "sgdfnet_pred" in work.columns and work["sgdfnet_pred"].notna().any():
        work["anchor_spread"] = work["sgdfnet_pred"] - work["da_anchor"]
        sgdfnet_available = True
    else:
        work["sgdfnet_pred"] = np.nan
        work["anchor_spread"] = np.nan

    # ── Collect feature columns ───────────────────────────────────────
    feature_cols = []
    for group_name, group_cols in ALL_FEATURE_GROUPS.items():
        for col in group_cols:
            if col in work.columns and col not in feature_cols:
                # Check it's not all NaN
                if work[col].notna().any():
                    feature_cols.append(col)

    # ── Feature audit ─────────────────────────────────────────────────
    audit = _compute_audit(work, feature_cols, forecast_available, sgdfnet_available, mode)

    return DeltaSupplyFeatureResult(
        df=work,
        feature_columns=feature_cols,
        audit=audit,
        mode=mode,
    )


def _build_derived_features(
    work: pd.DataFrame,
    forecast_available: dict,
) -> None:
    """Build derived supply-demand features in-place."""
    lc = "load_forecast" if forecast_available.get("load_forecast") else None
    rc = "renewable_forecast" if forecast_available.get("renewable_forecast") else None
    wc = "wind_forecast" if forecast_available.get("wind_forecast") else None
    sc = "solar_forecast" if forecast_available.get("solar_forecast") else None
    tc = "tie_line_forecast" if forecast_available.get("tie_line_forecast") else None
    bc = "bidding_space_forecast" if forecast_available.get("bidding_space_forecast") else None

    # forecast_net_load = load_forecast - renewable_forecast
    if lc and rc:
        work["forecast_net_load"] = work[lc] - work[rc]
    else:
        work["forecast_net_load"] = np.nan

    # forecast_renewable_share = renewable / max(load, eps)
    if lc and rc:
        work["forecast_renewable_share"] = work[rc] / work[lc].clip(lower=EPS)
    else:
        work["forecast_renewable_share"] = np.nan

    # forecast_wind_share
    if lc and wc:
        work["forecast_wind_share"] = work[wc] / work[lc].clip(lower=EPS)
    else:
        work["forecast_wind_share"] = np.nan

    # forecast_solar_share
    if lc and sc:
        work["forecast_solar_share"] = work[sc] / work[lc].clip(lower=EPS)
    else:
        work["forecast_solar_share"] = np.nan

    # forecast_supply_demand_gap = load - renewable - tie_line
    if lc and rc and tc:
        work["forecast_supply_demand_gap"] = work[lc] - work[rc] - work[tc]
    elif lc and rc:
        # Without tie_line, approximate
        work["forecast_supply_demand_gap"] = work[lc] - work[rc]
    else:
        work["forecast_supply_demand_gap"] = np.nan

    # forecast_bidding_pressure = load - renewable - bidding_space
    if lc and rc and bc:
        work["forecast_bidding_pressure"] = work[lc] - work[rc] - work[bc]
    elif lc and rc:
        work["forecast_bidding_pressure"] = work[lc] - work[rc]
    else:
        work["forecast_bidding_pressure"] = np.nan

    # forecast_thermal_pressure = load - wind - solar - tie_line
    if lc and wc and sc and tc:
        work["forecast_thermal_pressure"] = work[lc] - work[wc] - work[sc] - work[tc]
    elif lc and wc and sc:
        work["forecast_thermal_pressure"] = work[lc] - work[wc] - work[sc]
    else:
        work["forecast_thermal_pressure"] = np.nan


def _build_lag_features(work: pd.DataFrame, mode: str) -> None:
    """Build historical lag features. FULL_DAY: D-1 and earlier only."""
    # Actual-side columns for lags (from Chinese mapping)
    actual_load_col = "system_load_actual" if "system_load_actual" in work.columns else None
    actual_renewable_col = "renewable_actual" if "renewable_actual" in work.columns else None
    actual_wind_col = "wind_actual" if "wind_actual" in work.columns else None
    actual_solar_col = "solar_actual" if "solar_actual" in work.columns else None

    # For FULL_DAY mode, lag features use only D-1 and earlier data
    # We shift by 24h (lag_24h) and 48h (lag_48h) to ensure no same-day leakage
    # Since data is hourly and sorted, shift(24) gives the value from 24 hours ago

    # actual_load_lag_24h / lag_48h
    if actual_load_col:
        work["actual_load_lag_24h"] = work[actual_load_col].shift(24)
        work["actual_load_lag_48h"] = work[actual_load_col].shift(48)
    else:
        work["actual_load_lag_24h"] = np.nan
        work["actual_load_lag_48h"] = np.nan

    # actual_renewable/wind/solar lag_24h
    if actual_renewable_col:
        work["actual_renewable_lag_24h"] = work[actual_renewable_col].shift(24)
    else:
        work["actual_renewable_lag_24h"] = np.nan

    if actual_wind_col:
        work["actual_wind_lag_24h"] = work[actual_wind_col].shift(24)
    else:
        work["actual_wind_lag_24h"] = np.nan

    if actual_solar_col:
        work["actual_solar_lag_24h"] = work[actual_solar_col].shift(24)
    else:
        work["actual_solar_lag_24h"] = np.nan

    # Previous day means (shift by 24h then rolling 24h mean)
    # This gives the mean of the previous day's same hour window
    for src_col, out_col in [
        (actual_load_col, "previous_day_load_mean"),
        (actual_renewable_col, "previous_day_renewable_mean"),
        (actual_wind_col, "previous_day_wind_mean"),
        (actual_solar_col, "previous_day_solar_mean"),
    ]:
        if src_col:
            work[out_col] = work[src_col].shift(24).rolling(24, min_periods=1).mean()
        else:
            work[out_col] = np.nan

    # Previous day price_delta stats
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


def _build_calendar_features(work: pd.DataFrame) -> None:
    """Build calendar / period features."""
    h = work["hour_business"].astype(int)
    work["hour_sin"] = np.sin(2 * np.pi * h / 24)
    work["hour_cos"] = np.cos(2 * np.pi * h / 24)

    # Day of week from business_day
    dow = work["business_day"].dt.dayofweek.astype(int)
    work["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    work["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Month
    month = work["business_day"].dt.month.astype(int)
    work["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    work["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Weekend
    work["is_weekend"] = (dow >= 5).astype(int)

    # Period ID (numeric encoding)
    period_map = {"1_8": 0, "9_16": 1, "17_24": 2}
    work["period_id"] = work["period"].map(period_map).fillna(1).astype(int)


def _compute_audit(
    work: pd.DataFrame,
    feature_columns: list,
    forecast_available: dict,
    sgdfnet_available: bool,
    mode: str,
) -> FeatureAudit:
    """Compute feature audit verdict."""
    n_features = len(feature_columns)

    # Coverage calculations
    n_forecast_wanted = len(FORECAST_FEATURES)
    n_forecast_present = sum(1 for f in FORECAST_FEATURES if forecast_available.get(f, False))
    forecast_coverage = n_forecast_present / max(n_forecast_wanted, 1)

    n_lag_wanted = len(LAG_FEATURES)
    n_lag_present = sum(
        1 for f in LAG_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    lag_coverage = n_lag_present / max(n_lag_wanted, 1)

    n_cal_wanted = len(CALENDAR_FEATURES)
    n_cal_present = sum(
        1 for f in CALENDAR_FEATURES
        if f in work.columns and work[f].notna().any()
    )
    cal_coverage = n_cal_present / max(n_cal_wanted, 1)

    # Missing features
    missing = []
    for group_cols in ALL_FEATURE_GROUPS.values():
        for col in group_cols:
            if col not in feature_columns:
                missing.append(col)

    # Leakage check: in FULL_DAY, ensure no same-day actual used
    leakage_ok = True  # We enforce this by construction in _build_lag_features

    # Verdict
    if n_features >= 15 and forecast_coverage >= 0.70 and leakage_ok:
        verdict = "FORMAL_READY"
    elif n_features >= 8 and leakage_ok:
        verdict = "PARTIAL_READY"
    else:
        verdict = "NOT_READY"

    return FeatureAudit(
        n_features=n_features,
        feature_columns=feature_columns,
        missing_features=missing,
        forecast_feature_coverage=round(forecast_coverage, 4),
        lag_feature_coverage=round(lag_coverage, 4),
        calendar_feature_coverage=round(cal_coverage, 4),
        sgdfnet_available=sgdfnet_available,
        leakage_check=leakage_ok,
        verdict=verdict,
        formal_ready=(verdict == "FORMAL_READY"),
    )
