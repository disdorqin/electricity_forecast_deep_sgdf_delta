"""Feature engineering for DeepRT-SOTA v2.

This module implements feature building for the standalone realtime price deep model.

Feature groups (as specified in DEEP_RT_SOTA_V2_SCOPE.md):
1. Price history features
2. Anchor / forecast features
3. Calendar features
4. Risk features (optional)
5. Forecast-side power market features (optional)

Strictly uses business_time.py for time alignment.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.deep_sgdf_delta.business_time import add_business_time_columns

logger = logging.getLogger(__name__)


# ── Feature Group Definitions ───────────────────────────────────────────

PRICE_HISTORY_FEATURES = [
    "rt_lag_24h",
    "rt_lag_48h",
    "rt_lag_72h",
    "rt_lag_168h",
    "previous_day_rt_mean",
    "previous_day_rt_std",
    "previous_7d_same_hour_mean",
    "previous_7d_same_hour_std",
]

ANCHOR_FORECAST_FEATURES = [
    "da_anchor",
    "forecast_price",
    "anchor_spread",
]

CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "period_id",
]

RISK_FEATURES = [
    "negative_prob",
    "negative_risk_score",
    "spike_prob",
    "spike_risk_score",
    "deviation_down_prob",
    "deviation_up_prob",
    "deviation_risk_score",
    "relative_spike_prob",
    "relative_down_prob",
]

FORECAST_SIDE_FEATURES = [
    "load_forecast",
    "renewable_forecast",
    "wind_forecast",
    "solar_forecast",
    "tie_line_forecast",
    "bidding_space_forecast",
    "forecast_net_load",
    "forecast_renewable_share",
    "forecast_supply_demand_gap",
    "forecast_bidding_pressure",
]

RESIDUAL_HISTORY_FEATURES = [
    "residual_lag_24h",
    "residual_lag_48h",
    "residual_lag_72h",
    "residual_lag_168h",
    "residual_prev_day_mean",
    "residual_prev_day_std",
    "residual_prev_7d_same_hour_mean",
    "residual_prev_14d_same_hour_mean",
    "residual_prev_7d_period_mean",
]


def build_deep_rt_sota_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "ds",
    rt_actual_col: str = "rt_actual",
    da_anchor_col: str = "da_anchor",
    risk_features: str = "off",  # "off" | "real" | "synthetic"
    forecast_features: bool = False,
    use_residual_history_features: bool = False,
    risk_df: Optional[pd.DataFrame] = None,
    base_pred_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:

    """Build feature table for DeepRT-SOTA v2.

    Args:
        df: Raw hourly DataFrame with at least columns:
            ["ds", "rt_actual", "da_anchor"].
        timestamp_col: Name of timestamp column.
        rt_actual_col: Name of realtime actual column.
        da_anchor_col: Name of day-ahead anchor column.
        risk_features: Whether to include risk features.
        forecast_features: Whether to include forecast-side features.
        risk_df: Optional risk feature pack.
        base_pred_df: Optional base prediction DataFrame.

    Returns:
        Tuple of (feature_df, feature_manifest).
    """
    df = df.copy()

    # Ensure business time columns
    if "business_day" not in df.columns or "hour_business" not in df.columns:
        df = add_business_time_columns(df, timestamp_col=timestamp_col)

    feature_manifest = {
        "price_history_features": [],
        "anchor_forecast_features": [],
        "calendar_features": [],
        "risk_features": [],
        "forecast_side_features": [],
        "missing_features": [],
    }

    # ── 1. Price history features ──────────────────────────────────────
    logger.info("Building price history features...")
    df, ph_manifest = _build_price_history_features(df, rt_actual_col=rt_actual_col)
    feature_manifest["price_history_features"] = ph_manifest

    # ── 2. Anchor / forecast features ──────────────────────────────────
    logger.info("Building anchor/forecast features...")
    df, af_manifest = _build_anchor_forecast_features(
        df, da_anchor_col=da_anchor_col, base_pred_df=base_pred_df
    )
    feature_manifest["anchor_forecast_features"] = af_manifest

    # ── 3. Calendar features ──────────────────────────────────────────
    logger.info("Building calendar features...")
    df, cal_manifest = _build_calendar_features(df)
    feature_manifest["calendar_features"] = cal_manifest

    # ── 4. Risk features (optional) ────────────────────────────────────
    if risk_features in ("real", "synthetic"):
        logger.info(f"Building risk features (mode={risk_features})...")
        df, risk_manifest = _build_risk_features(
            df, risk_df=risk_df, mode=risk_features
        )
        feature_manifest["risk_features"] = risk_manifest
        feature_manifest["risk_features_source"] = risk_features
    else:
        logger.info("Risk features disabled (off), skipping...")
        feature_manifest["risk_features_source"] = "off"

    # ── 5. Residual history features (optional, for residual_to_da mode) ──
    if use_residual_history_features:
        logger.info("Building residual history features...")
        df, rh_manifest = _build_residual_history_features(
            df, rt_actual_col=rt_actual_col, da_anchor_col=da_anchor_col
        )
        feature_manifest["residual_history_features"] = rh_manifest
        feature_manifest["residual_history_features_enabled"] = True
    else:
        logger.info("Residual history features disabled, skipping...")
        feature_manifest["residual_history_features_enabled"] = False

    # ── 6. Forecast-side features (optional) ──────────────────────────
    if forecast_features:
        logger.info("Building forecast-side features...")
        df, fs_manifest = _build_forecast_side_features(df)
        feature_manifest["forecast_side_features"] = fs_manifest
    else:
        logger.info("Forecast-side features disabled, skipping...")

    return df, feature_manifest


def _build_price_history_features(
    df: pd.DataFrame, rt_actual_col: str = "rt_actual"
) -> Tuple[pd.DataFrame, List[str]]:
    """Build price history features.

    Args:
        df: Input DataFrame (must be sorted by ds).
        rt_actual_col: Name of realtime actual column.

    Returns:
        Tuple of (df with features, list of built feature names).
    """
    built_features = []

    # Ensure sorted
    df = df.sort_values("ds").reset_index(drop=True)

    # rt_lag_24h: same hour yesterday
    df["rt_lag_24h"] = df[rt_actual_col].shift(24)
    built_features.append("rt_lag_24h")

    # rt_lag_48h: same hour two days ago
    df["rt_lag_48h"] = df[rt_actual_col].shift(48)
    built_features.append("rt_lag_48h")

    # rt_lag_72h: same hour three days ago
    df["rt_lag_72h"] = df[rt_actual_col].shift(72)
    built_features.append("rt_lag_72h")

    # rt_lag_168h: same hour one week ago
    df["rt_lag_168h"] = df[rt_actual_col].shift(168)
    built_features.append("rt_lag_168h")

    # previous_day_rt_mean: mean of previous day (24 hours before)
    # For each row, compute mean of rt_actual for the previous 24 hours
    df["previous_day_rt_mean"] = (
        df[rt_actual_col].shift(1).rolling(window=24, min_periods=24).mean()
    )
    built_features.append("previous_day_rt_mean")

    # previous_day_rt_std
    df["previous_day_rt_std"] = (
        df[rt_actual_col].shift(1).rolling(window=24, min_periods=24).std()
    )
    built_features.append("previous_day_rt_std")

    # previous_7d_same_hour_mean: mean of same hour for previous 7 days
    df["previous_7d_same_hour_mean"] = (
        df[rt_actual_col].shift(24).rolling(window=7 * 24, min_periods=7 * 24).mean()
    )
    built_features.append("previous_7d_same_hour_mean")

    # previous_7d_same_hour_std
    df["previous_7d_same_hour_std"] = (
        df[rt_actual_col].shift(24).rolling(window=7 * 24, min_periods=7 * 24).std()
    )
    built_features.append("previous_7d_same_hour_std")

    return df, built_features


def _build_anchor_forecast_features(
    df: pd.DataFrame,
    da_anchor_col: str = "da_anchor",
    base_pred_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build anchor/forecast features.

    Args:
        df: Input DataFrame.
        da_anchor_col: Name of day-ahead anchor column.
        base_pred_df: Optional base prediction DataFrame.

    Returns:
        Tuple of (df with features, list of built feature names).
    """
    built_features = []

    # da_anchor
    if da_anchor_col in df.columns:
        built_features.append("da_anchor")
    else:
        logger.warning(f"da_anchor_col '{da_anchor_col}' not found, skipping...")
        df["da_anchor"] = np.nan
        built_features.append("da_anchor")

    # forecast_price (same as da_anchor if not available)
    if "forecast_price" in df.columns:
        built_features.append("forecast_price")
    else:
        # Use da_anchor as fallback
        df["forecast_price"] = df["da_anchor"]
        built_features.append("forecast_price")

    # anchor_spread: difference between da_anchor and previous day da_anchor
    df["anchor_spread"] = df["da_anchor"].diff()
    built_features.append("anchor_spread")

    # sgdfnet_pred (optional)
    if base_pred_df is not None:
        # Join base prediction
        df = df.merge(
            base_pred_df[["business_day", "hour_business", "base_pred"]],
            on=["business_day", "hour_business"],
            how="left",
        )
        df = df.rename(columns={"base_pred": "sgdfnet_pred"})
        built_features.append("sgdfnet_pred")
    else:
        logger.info("No base_pred_df provided, sgdfnet_pred will be missing...")
        df["sgdfnet_pred"] = np.nan

    return df, built_features


def _build_calendar_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build calendar features.

    Args:
        df: Input DataFrame (must have business_time columns).

    Returns:
        Tuple of (df with features, list of built feature names).
    """
    built_features = []

    # hour_sin, hour_cos
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_business"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_business"] / 24.0)
    built_features.extend(["hour_sin", "hour_cos"])

    # dow_sin, dow_cos (day of week)
    df["dow"] = pd.to_datetime(df["business_day"]).dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    built_features.extend(["dow_sin", "dow_cos"])

    # month_sin, month_cos
    df["month"] = pd.to_datetime(df["business_day"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    built_features.extend(["month_sin", "month_cos"])

    # is_weekend
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    built_features.append("is_weekend")

    # period_id (1_8=0, 9_16=1, 17_24=2)
    df["period_id"] = 0
    df.loc[(df["hour_business"] >= 9) & (df["hour_business"] <= 16), "period_id"] = 1
    df.loc[df["hour_business"] >= 17, "period_id"] = 2
    built_features.append("period_id")

    return df, built_features


def _build_risk_features(
    df: pd.DataFrame,
    risk_df: Optional[pd.DataFrame] = None,
    mode: str = "real",  # "real" | "synthetic"
) -> Tuple[pd.DataFrame, List[str]]:
    """Build risk features.

    Args:
        df: Input DataFrame.
        risk_df: Optional risk feature pack (for "real" mode).
        mode: "real" (load from risk_df) or "synthetic" (generate synthetic).

    Returns:
        Tuple of (df with features, list of built feature names).
    """
    built_features = []

    if mode == "real":
        if risk_df is not None:
            risk_cols = [col for col in risk_df.columns if col in RISK_FEATURES]
            df = df.merge(
                risk_df[["business_day", "hour_business"] + risk_cols],
                on=["business_day", "hour_business"],
                how="left",
            )
            built_features.extend(risk_cols)
            logger.info(f"  Loaded real risk features: {risk_cols}")
        else:
            logger.warning("  mode='real' but no risk_df provided, setting to NaN...")
            for col in RISK_FEATURES:
                df[col] = np.nan
    elif mode == "synthetic":
        logger.info("  Generating SYNTHETIC risk features (debug only)...")
        for col in RISK_FEATURES:
            df[col] = np.random.rand(len(df))
        built_features.extend(RISK_FEATURES)
    else:
        raise ValueError(f"Unknown risk mode: {mode}")

    return df, built_features


def _build_forecast_side_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build forecast-side power market features.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (df with features, list of built feature names).
    """
    built_features = []

    # Try to auto-detect forecast-side features
    for col in FORECAST_SIDE_FEATURES:
        if col in df.columns:
            built_features.append(col)
        else:
            logger.warning(f"Forecast feature '{col}' not found, setting to NaN...")
            df[col] = np.nan
            built_features.append(col)

    return df, built_features


def _build_residual_history_features(
    df: pd.DataFrame,
    rt_actual_col: str = "rt_actual",
    da_anchor_col: str = "da_anchor",
) -> Tuple[pd.DataFrame, List[str]]:
    """Build residual history features (strictly no target-day leakage).

    Residual = rt_actual - da_anchor.
    All lag features use ONLY data from D-1 and earlier.

    Args:
        df: Input DataFrame (must be sorted by ds, have business_time columns).
        rt_actual_col: Name of realtime actual column.
        da_anchor_col: Name of day-ahead anchor column.

    Returns:
        Tuple of (df with features, list of built feature names).
    """
    built_features = []
    df = df.sort_values("ds").reset_index(drop=True)

    # Compute residual (rt_actual - da_anchor)
    residual = df[rt_actual_col] - df[da_anchor_col]

    # residual_lag_24h: residual at same hour on previous day (D-1)
    df["residual_lag_24h"] = residual.shift(24)
    built_features.append("residual_lag_24h")

    # residual_lag_48h: residual at same hour two days ago (D-2)
    df["residual_lag_48h"] = residual.shift(48)
    built_features.append("residual_lag_48h")

    # residual_lag_72h: residual at same hour three days ago (D-3)
    df["residual_lag_72h"] = residual.shift(72)
    built_features.append("residual_lag_72h")

    # residual_lag_168h: residual at same hour one week ago (D-7)
    df["residual_lag_168h"] = residual.shift(168)
    built_features.append("residual_lag_168h")

    # residual_prev_day_mean: mean of residual for previous day (D-1, all 24h)
    df["residual_prev_day_mean"] = (
        residual.shift(1).rolling(window=24, min_periods=24).mean()
    )
    built_features.append("residual_prev_day_mean")

    # residual_prev_day_std
    df["residual_prev_day_std"] = (
        residual.shift(1).rolling(window=24, min_periods=24).std()
    )
    built_features.append("residual_prev_day_std")

    # residual_prev_7d_same_hour_mean: mean of residual at same hour for previous 7 days
    df["residual_prev_7d_same_hour_mean"] = (
        residual.shift(24).rolling(window=7 * 24, min_periods=7 * 24).mean()
    )
    built_features.append("residual_prev_7d_same_hour_mean")

    # residual_prev_14d_same_hour_mean
    df["residual_prev_14d_same_hour_mean"] = (
        residual.shift(24).rolling(window=14 * 24, min_periods=14 * 24).mean()
    )
    built_features.append("residual_prev_14d_same_hour_mean")

    # residual_prev_7d_period_mean: mean of residual for same period (peak/mid/off)
    # peak=hours 9-11,17-20; mid=hours 7-8,12-16; off=hours 0-6,21-24
    period_dummy = df["hour_business"].apply(
        lambda h: "peak" if h in [9, 10, 11, 17, 18, 19, 20] else ("mid" if h in [7, 8, 12, 13, 14, 15, 16] else "off")
    )
    for period_name in ["peak", "mid", "off"]:
        col_name = f"residual_prev_7d_{period_name}_mean"
        # Compute rolling mean for each period separately
        mask = period_dummy == period_name
        period_residual = residual.where(mask)
        # Shift by 1 to avoid using current hour, then roll
        rolled = period_residual.shift(1).rolling(window=7 * 24, min_periods=1).mean()
        df[col_name] = rolled
        built_features.append(col_name)

    return df, built_features


def get_feature_columns(
    risk_features: str = "off",  # "off" | "real" | "synthetic"
    forecast_features: bool = False,
    use_residual_history_features: bool = False,
) -> List[str]:
    """Get list of feature columns based on configuration.

    Args:
        risk_features: "off" | "real" | "synthetic".
        forecast_features: Whether forecast-side features are included.
        use_residual_history_features: Whether residual history features are included.

    Returns:
        List of feature column names.
    """
    features = []
    features.extend(PRICE_HISTORY_FEATURES)
    features.extend(ANCHOR_FORECAST_FEATURES)
    features.extend(CALENDAR_FEATURES)

    if risk_features in ("real", "synthetic"):
        features.extend(RISK_FEATURES)

    if use_residual_history_features:
        features.extend(RESIDUAL_HISTORY_FEATURES)

    if forecast_features:
        features.extend(FORECAST_SIDE_FEATURES)

    return features
