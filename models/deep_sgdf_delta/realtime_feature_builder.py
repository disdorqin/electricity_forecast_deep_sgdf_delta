"""Realtime feature builder for TrendKnightRT (Phase DeepFinal-2).

Builds a complete feature table from raw hourly Shandong electricity data
for realtime (intra-day) delta / residual prediction.

Input:
    - Raw hourly DataFrame (Shandong PMOS format, Chinese or English cols).
    - Optional SGDFNet prediction DataFrame.
    - Optional teacher prediction DataFrame.

Output:
    Augmented DataFrame with ALL features required by the feature contract,
    ready for ``RealtimeDayDataset`` consumption.

Feature groups produced:

    1. **Business time** — ``business_day``, ``hour_business``, ``period``.
    2. **Anchor / target** — ``da_anchor``, ``rt_actual``, ``delta_target``.
    3. **Calendar** — ``hour_sin`` / ``hour_cos``, ``dow_sin`` / ``dow_cos``,
       ``month_sin`` / ``month_cos``, ``is_weekend``, ``is_holiday``.
    4. **Lag** — ``rt_lag_1h`` / ``2h`` / ``3h`` / ``24h`` / ``48h``,
       ``rt_mean_6h`` / ``24h``, ``rt_std_6h`` / ``24h``,
       ``delta_lag_24h`` / ``48h``.
    5. **SGDFNet** — ``sgdfnet_pred``, ``sgdfnet_residual_lag_1h``,
       ``sgdfnet_residual_lag_24h``, ``sgdfnet_residual_mean_7d``.
    6. **Forecast-side features** — from Chinese forecast columns mapped
       via :mod:`realtime_column_mapping`.

Leakage safety:
    - Lag features reference only *historical* data (hours / days before
      the target hour).
    - FULL_DAY mode: only D-1 and earlier data are used.
    - INTRADAY mode: same-day data before cutoff_hour is also available.
    - The builder automatically detects and marks which mode is active.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from .business_time import add_business_time_columns, compute_period
from .realtime_column_mapping import (
    CN_FORECAST_MAP,
    CN_ACTUAL_MAP,
    CN_CORE,
    audit_chinese_column_mapping,
    rename_chinese_columns,
)
from .realtime_feature_contract import (
    ALL_FEATURES,
    REQUIRED_FEATURES,
    OPTIONAL_FEATURES,
    FEATURE_VERSION,
    check_leakage,
)

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────

HOLIDAY_SCHEDULE: set[str] = {
    # 2025-2026 Chinese public holidays (simplified set)
    # Spring Festival (春节) 2025
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03",
    # Spring Festival (春节) 2026
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-02-21", "2026-02-22",
    # National Day (国庆节) 2025
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04",
    "2025-10-05", "2025-10-06", "2025-10-07",
    # National Day (国庆节) 2026
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04",
    "2026-10-05", "2026-10-06", "2026-10-07",
    # Qingming (清明节) 2025
    "2025-04-04", "2025-04-05",
    # Qingming 2026
    "2026-04-04", "2026-04-05",
    # Labour Day (劳动节) 2025
    "2025-05-01", "2025-05-02", "2025-05-03",
    # Labour Day 2026
    "2026-05-01", "2026-05-02", "2026-05-03",
    # Dragon Boat (端午节) 2025
    "2025-05-31", "2025-06-01",
    # Dragon Boat 2026
    "2026-06-19", "2026-06-20",
    # Mid-Autumn (中秋节) 2025
    "2025-10-06",
    # Mid-Autumn 2026
    "2026-09-26",
}

REQUIRED_LAG_FEATURES: list[str] = [
    "rt_lag_1h", "rt_lag_2h", "rt_lag_3h",
    "rt_lag_24h", "rt_lag_48h",
    "rt_mean_6h", "rt_std_6h",
    "rt_mean_24h", "rt_std_24h",
    "delta_lag_24h", "delta_lag_48h",
    "previous_day_delta_mean_24h",
    "previous_day_delta_std_24h",
]

REQUIRED_CALENDAR_FEATURES: list[str] = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    # is_holiday is optional
]

REQUIRED_SGDFNET_FEATURES: list[str] = [
    "sgdfnet_pred",
]

REQUIRED_FORECAST_FEATURES: list[str] = [
    "forecast_price",
    "load_forecast",
    "renewable_forecast",
    "wind_forecast",
    "solar_forecast",
    "tie_line_forecast",
    "bidding_space_forecast",
]


def build_realtime_features(  # noqa: C901 (complexity ok — feature builder)
    raw_df: pd.DataFrame,
    *,
    sgdfnet_pred_df: pd.DataFrame | None = None,
    teacher_pred_df: pd.DataFrame | None = None,
    timestamp_col: str = "ds",
    da_anchor_col: str = "da_anchor",
    rt_actual_col: str = "rt_actual",
    mode: Literal["FULL_DAY", "INTRADAY"] = "FULL_DAY",
    allow_sgdfnet_fallback: bool = False,
    holiday_schedule: set[str] | None = None,
) -> pd.DataFrame:
    """Build the complete realtime feature table.

    Args:
        raw_df: Raw hourly DataFrame.  May have Chinese or English column
            names.  Must contain at minimum a timestamp column and a
            day-ahead price column.
        sgdfnet_pred_df: Optional DataFrame with SGDFNet predictions.
            Must contain ``ds`` and ``sgdfnet_pred`` columns.  If provided,
            predictions are joined on timestamp; if missing and
            *allow_sgdfnet_fallback* is ``False``, a ``ValueError`` is
            raised.
        teacher_pred_df: Optional DataFrame with teacher model predictions
            (rt916_pred, timemixer_pred, timesfm_pred).  Joined on
            timestamp.
        timestamp_col: Column name for the timestamp after renaming.
        da_anchor_col: Column name for day-ahead anchor price.
        rt_actual_col: Column name for realtime actual price.
        mode: ``"FULL_DAY"`` (default) or ``"INTRADAY"``.  Affects which
            lag features are available.
        allow_sgdfnet_fallback: If ``True``, missing ``sgdfnet_pred``
            values are filled from ``da_anchor``.  If ``False`` (default),
            missing ``sgdfnet_pred`` raises ``ValueError``.
        holiday_schedule: Optional set of ``"YYYY-MM-DD"`` holiday date
            strings.  Defaults to the built-in ``HOLIDAY_SCHEDULE``.

    Returns:
        Augmented DataFrame with all feature columns.  The caller is
        responsible for passing this into ``RealtimeDayDataset`` or
        ``build_training_datasets_final``.

    Raises:
        ValueError: If required columns are missing or SGDFNet predictions
            are missing in non-fallback mode.
    """
    if holiday_schedule is None:
        holiday_schedule = HOLIDAY_SCHEDULE

    # ── Step 1: Rename Chinese columns if needed ────────────────────
    df = raw_df.copy()
    df = rename_chinese_columns(df)

    # ── Step 2: Validate core columns ───────────────────────────────
    required_core = {timestamp_col, da_anchor_col, rt_actual_col}
    missing_core = required_core - set(df.columns)
    if missing_core:
        raise ValueError(
            f"Missing core columns: {missing_core}. "
            f"Available: {list(df.columns[:20])}"
        )

    # ── Step 3: Parse timestamp and sort ────────────────────────────
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # ── Step 4: Add business-time alignment ─────────────────────────
    if "business_day" not in df.columns or "hour_business" not in df.columns:
        df = add_business_time_columns(df, timestamp_col=timestamp_col)

    # Ensure numeric types
    df["hour_business"] = df["hour_business"].astype(int)
    df["business_day"] = pd.to_datetime(df["business_day"])

    # ── Step 5: Compute delta target ────────────────────────────────
    if "delta_target" not in df.columns:
        df["delta_target"] = df[rt_actual_col] - df[da_anchor_col]

    # ── Step 6: Generate calendar features ──────────────────────────
    df = _add_calendar_features(df, timestamp_col=timestamp_col,
                                holiday_schedule=holiday_schedule)

    # ── Step 7: Generate lag features ───────────────────────────────
    df = _add_lag_features(df, mode=mode)

    # ── Step 8: Integrate SGDFNet predictions ───────────────────────
    df = _integrate_sgdfnet(
        df,
        sgdfnet_pred_df=sgdfnet_pred_df,
        allow_fallback=allow_sgdfnet_fallback,
        mode=mode,
    )

    # ── Step 9: Add forecast-side features ──────────────────────────
    # These are already renamed in step 1 — just ensure forecast_price alias
    if "forecast_price" not in df.columns and da_anchor_col in df.columns:
        df["forecast_price"] = df[da_anchor_col]

    # ── Step 10: Add teacher predictions if provided ────────────────
    if teacher_pred_df is not None:
        df = _add_teacher_features(df, teacher_pred_df)

    # ── Step 11: Fill NaN in feature columns ────────────────────────
    feature_cols_present = [c for c in ALL_FEATURES if c in df.columns]
    for col in feature_cols_present:
        df[col] = df[col].fillna(0.0)
    return df


# ── Calendar features ──────────────────────────────────────────────────

def _add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = "ds",
    holiday_schedule: set[str] | None = None,
) -> pd.DataFrame:
    """Add cyclical calendar features and holiday/weekend indicators."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    hour = ts.dt.hour
    # For business-hour alignment: midnight (hour 0) → hour_business 24
    hour_for_calendar = np.where(hour == 0, 24, hour)

    dow = ts.dt.dayofweek  # 0=Monday, 6=Sunday
    month = ts.dt.month

    # Cyclical encoding for hour
    df["hour_sin"] = np.sin(2 * np.pi * hour_for_calendar / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_for_calendar / 24)

    # Cyclical encoding for day of week
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Weekend indicator
    df["is_weekend"] = (dow >= 5).astype(int)

    # Holiday indicator
    if holiday_schedule is not None:
        date_strs = ts.dt.strftime("%Y-%m-%d")
        df["is_holiday"] = date_strs.isin(holiday_schedule).astype(int)
    else:
        df["is_holiday"] = 0

    logger.debug(
        "Calendar features added: hour_sin/cos, dow_sin/cos, "
        "month_sin/cos, is_weekend, is_holiday"
    )
    return df


# ── Lag features ───────────────────────────────────────────────────────

def _add_lag_features(
    df: pd.DataFrame,
    mode: Literal["FULL_DAY", "INTRADAY"] = "FULL_DAY",
    rt_actual_col: str = "rt_actual",
    da_anchor_col: str = "da_anchor",
) -> pd.DataFrame:
    """Add lag features from historical actuals.

    **FULL_DAY mode** (default):
        Only D-1 and earlier lags are used.  All features are computed
        from **complete previous-day data only** — no same-day actuals
        are ever consumed.

        Lags generated (all cutoff-safe):
        - ``rt_lag_24h``: same-hour yesterday  (D-1, same hour_business)
        - ``rt_lag_48h``: same-hour two days ago
        - ``rt_mean_24h``: D-1 full-day mean of rt_actual  (previous_day_rt_mean)
        - ``rt_std_24h``: D-1 full-day std of rt_actual   (previous_day_rt_std)
        - ``delta_lag_24h``: delta (rt-da) same-hour yesterday
        - ``delta_lag_48h``: delta two days ago
        - ``previous_day_delta_mean_24h``: D-1 mean of delta
        - ``previous_day_delta_std_24h``: D-1 std of delta

        Same-day rolling features (rt_lag_1h, rt_lag_2h, rt_lag_3h,
        rt_mean_6h, rt_std_6h) are set to 0 — they MUST NOT be used
        in FULL_DAY mode because the target-hour actual is unavailable.

    **INTRADAY mode**:
        Same-day historical actuals (hours before the current target hour)
        are available.  Adds:
        - ``rt_lag_1h``, ``rt_lag_2h``, ``rt_lag_3h``: recent hours
        - ``rt_mean_6h``, ``rt_std_6h``: rolling 6-hour stats

    Edge cases:
        - If a lag feature has zero variance (e.g. no historical data at
          the start of training), it is filled with 0 and the manifest
          records ``zero_variance_lags``.
    """
    df = df.copy()
    df = df.sort_values("ds").reset_index(drop=True)

    n = len(df)
    zero_variance_lags: list[str] = []

    # ── Hour-level lags via shift (same hour previous days) ─────────
    # These are safe: shift(24) = same hour D-1, shift(48) = same hour D-2
    df["rt_lag_24h"] = df[rt_actual_col].shift(24)
    df["rt_lag_48h"] = df[rt_actual_col].shift(48)
    df["delta_lag_24h"] = (df[rt_actual_col] - df[da_anchor_col]).shift(24)
    df["delta_lag_48h"] = (df[rt_actual_col] - df[da_anchor_col]).shift(48)

    # ── Day-level stats (previous day only, no same-day rolling) ────
    # Compute per-business-day stats on D-1, then merge to current day.
    # This avoids any leakage of same-day actuals.
    if "business_day" in df.columns:
        daily_stats = (
            df.groupby("business_day")[rt_actual_col]
            .agg(["mean", "std"])
            .rename(columns={"mean": "rt_mean_24h", "std": "rt_std_24h"})
        )
        # Shift by 1 business_day: D's stats come from D-1
        daily_stats_shifted = daily_stats.shift(1)
        df = df.merge(
            daily_stats_shifted[["rt_mean_24h", "rt_std_24h"]],
            left_on="business_day", right_index=True,
            how="left", suffixes=("", "_daily"),
        )
        # If merge created duplicate columns, prefer the daily-computed ones
        if "rt_mean_24h_daily" in df.columns:
            df["rt_mean_24h"] = df["rt_mean_24h_daily"].fillna(df["rt_mean_24h"])
            df = df.drop(columns=["rt_mean_24h_daily"])
        if "rt_std_24h_daily" in df.columns:
            df["rt_std_24h"] = df["rt_std_24h_daily"].fillna(df["rt_std_24h"])
            df = df.drop(columns=["rt_std_24h_daily"])

        # Same for delta
        df["_delta"] = df[rt_actual_col] - df[da_anchor_col]
        delta_daily = (
            df.groupby("business_day")["_delta"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "previous_day_delta_mean_24h",
                             "std": "previous_day_delta_std_24h"})
        ).shift(1)
        df = df.merge(
            delta_daily[["previous_day_delta_mean_24h", "previous_day_delta_std_24h"]],
            left_on="business_day", right_index=True,
            how="left",
        )
        df = df.drop(columns=["_delta"])
    else:
        # Fallback: shift-based (less accurate but still leak-safe for D-1)
        df["rt_mean_24h"] = df[rt_actual_col].shift(24).rolling(24, min_periods=1).mean()
        df["rt_std_24h"] = df[rt_actual_col].shift(24).rolling(24, min_periods=1).std()
        df["previous_day_delta_mean_24h"] = 0.0
        df["previous_day_delta_std_24h"] = 0.0

    # ── Intra-day lags (mode-dependent) ────────────────────────────
    if mode == "INTRADAY":
        df["rt_lag_1h"] = df[rt_actual_col].shift(1)
        df["rt_lag_2h"] = df[rt_actual_col].shift(2)
        df["rt_lag_3h"] = df[rt_actual_col].shift(3)
        df["rt_mean_6h"] = df[rt_actual_col].shift(1).rolling(6, min_periods=1).mean()
        df["rt_std_6h"] = df[rt_actual_col].shift(1).rolling(6, min_periods=1).std()
    else:
        # FULL_DAY: intra-day lags are NOT available → set to 0
        df["rt_lag_1h"] = 0.0
        df["rt_lag_2h"] = 0.0
        df["rt_lag_3h"] = 0.0
        df["rt_mean_6h"] = 0.0
        df["rt_std_6h"] = 0.0

    # ── Post-processing ────────────────────────────────────────────
    for col in REQUIRED_LAG_FEATURES:
        if col in df.columns:
            # Fill NaN (initial rows where shift produces NaN)
            df[col] = df[col].fillna(0.0)            # Check for zero variance
            if df[col].nunique() <= 1:
                zero_variance_lags.append(col)

    if zero_variance_lags:
        logger.warning("Zero-variance lag features: %s", zero_variance_lags)

    logger.info(
        "Lag features added (mode=%s): %d features, zero-variance=%s",
        mode, len(REQUIRED_LAG_FEATURES), zero_variance_lags or "none",
    )
    return df


# ── SGDFNet integration ─────────────────────────────────────────────────

def _integrate_sgdfnet(
    df: pd.DataFrame,
    sgdfnet_pred_df: pd.DataFrame | None = None,
    allow_fallback: bool = False,
    timestamp_col: str = "ds",
    mode: Literal["FULL_DAY", "INTRADAY"] = "FULL_DAY",
) -> pd.DataFrame:
    """Integrate SGDFNet predictions into the feature table.

    Three cases:

    1. **Real predictions** (``sgdfnet_pred_df`` is provided):
        Joins on timestamp.  Any timestamps without a matching prediction
        are handled according to *allow_fallback*.

    2. **Existing column** (``sgdfnet_pred`` already in *df*):
        Uses it directly.  NaN values are handled per *allow_fallback*.

    3. **Fallback**:
        Only allowed if *allow_fallback* is ``True``.  Sets
        ``sgdfnet_pred = da_anchor`` for missing rows.

    SGDFNet residual features are also computed from the integrated
    predictions.
    """
    df = df.copy()

    # Track coverage
    total_rows = len(df)
    n_fallback = 0
    n_missing = 0

    # Determine if sgdfnet_pred comes from real file or external source
    if sgdfnet_pred_df is not None:
        # Join SGDFNet predictions on timestamp
        sgd = sgdfnet_pred_df.copy()
        if timestamp_col not in sgd.columns:
            # Try common alternatives
            for alt in ["ds", "timestamp", "时刻", "time"]:
                if alt in sgd.columns:
                    sgd = sgd.rename(columns={alt: timestamp_col})
                    break
        sgd[timestamp_col] = pd.to_datetime(sgd[timestamp_col])

        merge_cols = [timestamp_col]
        if "sgdfnet_pred" not in sgd.columns:
            # Try alternative column names
            for alt in ["pred", "prediction", "sgdf_pred", "sgdfnet"]:
                if alt in sgd.columns:
                    sgd = sgd.rename(columns={alt: "sgdfnet_pred"})
                    break

        if "sgdfnet_pred" in sgd.columns:
            # Merge only sgdfnet_pred column
            sgd_merge = sgd[[timestamp_col, "sgdfnet_pred"]].drop_duplicates(
                subset=[timestamp_col]
            )
            df = df.merge(sgd_merge, on=timestamp_col, how="left", suffixes=("", "_sgd"))
            # If there's a conflict (both _x and _y), prefer the explicit merge
            if "sgdfnet_pred_sgd" in df.columns and "sgdfnet_pred" in df.columns:
                # Fill NaN in original with merged value
                df["sgdfnet_pred"] = df["sgdfnet_pred"].fillna(df["sgdfnet_pred_sgd"])
                df = df.drop(columns=["sgdfnet_pred_sgd"])
        else:
            logger.warning(
                "sgdfnet_pred_df provided but no 'sgdfnet_pred' column found. "
                "Available: %s", list(sgd.columns[:10])
            )
    else:
        logger.info("No sgdfnet_pred_df provided; using existing column if present.")

    # Check coverage
    if "sgdfnet_pred" not in df.columns:
        if allow_fallback:
            if "da_anchor" in df.columns:
                df["sgdfnet_pred"] = df["da_anchor"]
                n_fallback = total_rows
                logger.info(
                    "sgdfnet_pred not found — fallback to da_anchor (ALL rows, %d)",
                    total_rows,
                )
            else:
                raise ValueError(
                    "Cannot fallback: neither sgdfnet_pred nor da_anchor present."
                )
        else:
            raise ValueError(
                "Missing sgdfnet_pred for formal training. "
                "Provide SGDFNet predictions or use --allow-sgdfnet-fallback "
                "for smoke only."
            )
    else:
        # sgdfnet_pred exists — handle NaN values
        mask_missing = df["sgdfnet_pred"].isna()
        n_missing = int(mask_missing.sum())
        if n_missing > 0:
            if allow_fallback:
                df.loc[mask_missing, "sgdfnet_pred"] = df.loc[mask_missing, "da_anchor"]
                n_fallback = n_missing
                logger.info(
                    "sgdfnet_pred NaN fallback to da_anchor: %d rows", n_missing,
                )
            else:
                raise ValueError(
                    f"sgdfnet_pred has {n_missing} NaN values in formal training. "
                    "Provide complete SGDFNet predictions or use --allow-sgdfnet-fallback."
                )

    # ── Compute SGDFNet residual features ──────────────────────────
    # residual = rt_actual - sgdfnet_pred
    if "rt_actual" in df.columns and "sgdfnet_pred" in df.columns:
        sgdfnet_residual = df["rt_actual"] - df["sgdfnet_pred"]

        # FULL_DAY: sgdfnet_residual_lag_1h leaks same-day data → must be 0
        if mode == "FULL_DAY":
            df["sgdfnet_residual_lag_1h"] = 0.0
        else:
            df["sgdfnet_residual_lag_1h"] = sgdfnet_residual.shift(1)

        # sgdfnet_residual_lag_24h is D-1 same hour → safe in both modes
        df["sgdfnet_residual_lag_24h"] = sgdfnet_residual.shift(24)

        # sgdfnet_residual_mean_7d: 7-day rolling mean of residual,
        # computed per hour_business across business_days (no same-day leak).
        if "business_day" in df.columns and "hour_business" in df.columns:
            # Group by hour_business, sort by business_day, then rolling(7)
            df["_bd_num"] = df["business_day"].rank(method="dense").astype(int)
            df["_resid"] = sgdfnet_residual
            residual_7d = (
                df.sort_values(["hour_business", "_bd_num"])
                .groupby("hour_business")["_resid"]
                .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
            )
            df["sgdfnet_residual_mean_7d"] = residual_7d.fillna(0.0)
            df = df.drop(columns=["_bd_num", "_resid"])
        else:
            # Fallback: shift(24*7) = 7 days ago, safer than shift(1).rolling(168)
            df["sgdfnet_residual_mean_7d"] = (
                sgdfnet_residual.shift(24 * 7).rolling(24 * 7, min_periods=1).mean()
            )

        # Fill NaN
        for col in ["sgdfnet_residual_lag_1h", "sgdfnet_residual_lag_24h",
                     "sgdfnet_residual_mean_7d"]:
            df[col] = df[col].fillna(0.0)
    # ── Coverage stats ─────────────────────────────────────────────
    present_mask = df["sgdfnet_pred"].notna()
    n_present = int(present_mask.sum())
    effective_coverage = (n_present / total_rows * 100) if total_rows > 0 else 0.0

    # Real coverage: rows where sgdfnet_pred came from a real prediction file
    # (not from fallback).  When fallback_used, the real coverage excludes
    # fallback rows.
    if n_fallback > 0:
        n_real = n_present - n_fallback
    else:
        n_real = n_present
    real_coverage = (n_real / total_rows * 100) if total_rows > 0 else 0.0

    # Attach coverage metadata as DataFrame attributes
    df.attrs["sgdfnet_coverage"] = effective_coverage
    df.attrs["sgdfnet_effective_coverage"] = effective_coverage
    df.attrs["sgdfnet_real_coverage"] = real_coverage
    df.attrs["sgdfnet_missing_rows"] = n_missing
    df.attrs["sgdfnet_fallback_used"] = n_fallback > 0
    df.attrs["sgdfnet_fallback_count"] = n_fallback
    df.attrs["sgdfnet_source"] = "file" if sgdfnet_pred_df is not None else "column"

    logger.info(
        "SGDFNet integration complete: effective_coverage=%.1f%%, "
        "real_coverage=%.1f%%, missing=%d, fallback=%s (%d rows)",
        effective_coverage, real_coverage, n_missing, n_fallback > 0, n_fallback,
    )

    return df


# ── Teacher features (optional) ────────────────────────────────────────

def _add_teacher_features(
    df: pd.DataFrame,
    teacher_df: pd.DataFrame,
    timestamp_col: str = "ds",
) -> pd.DataFrame:
    """Merge teacher model predictions into the feature table.

    Expected columns in *teacher_df*:
    - ``ds`` (or ``timestamp``): timestamp for joining.
    - ``rt916_pred`` (optional), ``timemixer_pred`` (optional),
      ``timesfm_pred`` (optional).
    """
    t = teacher_df.copy()
    if timestamp_col not in t.columns:
        for alt in ["ds", "timestamp", "time"]:
            if alt in t.columns:
                t = t.rename(columns={alt: timestamp_col})
                break

    t[timestamp_col] = pd.to_datetime(t[timestamp_col])

    # Select only known teacher columns + timestamp
    teacher_cols = [timestamp_col]
    for col in ["rt916_pred", "timemixer_pred", "timesfm_pred"]:
        if col in t.columns:
            teacher_cols.append(col)
            if col not in df.columns:
                logger.info("Teacher feature '%s' found and will be merged.", col)

    if len(teacher_cols) > 1:
        t_merge = t[teacher_cols].drop_duplicates(subset=[timestamp_col])
        df = df.merge(t_merge, on=timestamp_col, how="left", suffixes=("", "_t"))
        for col in ["rt916_pred", "timemixer_pred", "timesfm_pred"]:
            if f"{col}_t" in df.columns:
                df[col] = df[col].fillna(df[f"{col}_t"])
                df = df.drop(columns=[f"{col}_t"])
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

    return df


# ── Feature audit ──────────────────────────────────────────────────────

def audit_feature_coverage(  # noqa: C901
    df: pd.DataFrame,
    *,
    required_features: list[str] | None = None,
    optional_features: list[str] | None = None,
) -> dict[str, Any]:
    """Audit which features are present, missing, and their coverage stats.

    Args:
        df: DataFrame to audit (post feature-building).
        required_features: List of required feature names.  Defaults to
            ``REQUIRED_FEATURES`` from the feature contract.
        optional_features: List of optional feature names.  Defaults to
            ``OPTIONAL_FEATURES``.

    Returns:
        Detailed audit dict with keys:

        - ``n_features``
        - ``required_present`` / ``required_missing``
        - ``optional_present``
        - ``sgdfnet_real_coverage`` / ``sgdfnet_effective_coverage`` /
          ``sgdfnet_missing_rows`` / ``sgdfnet_fallback_used``
        - ``lag_feature_coverage``: fraction of lag features present
        - ``calendar_feature_generated``: bool
        - ``leakage_checked``: bool
        - ``formal_train_ready``: bool (True only when verdict is FORMAL_READY)
        - ``verdict``: ``"FORMAL_READY"``, ``"FALLBACK_READY"``,
          ``"PARTIAL_READY"``, or ``"NOT_READY"``.

    Verdict rules::

        FORMAL_READY:
          n_features >= 25
          sgdfnet_real_coverage >= 95
          fallback_used == False
          no high-risk leakage
          required_missing <= 3

        FALLBACK_READY:
          n_features >= 25
          sgdfnet_effective_coverage >= 95
          fallback_used == True
          no high-risk leakage

        PARTIAL_READY:
          n_features >= 15
          sgdfnet_real_coverage >= 80
          fallback_used == False

        NOT_READY:
          otherwise
    """
    if required_features is None:
        required_features = list(REQUIRED_FEATURES)
    if optional_features is None:
        optional_features = list(OPTIONAL_FEATURES)

    present_cols = set(df.columns)

    req_present = [c for c in required_features if c in present_cols]
    req_missing = [c for c in required_features if c not in present_cols]
    opt_present = [c for c in optional_features if c in present_cols]
    all_present = req_present + opt_present

    # Calendar features check
    cal_present = [c for c in REQUIRED_CALENDAR_FEATURES if c in present_cols]
    calendar_ok = len(cal_present) >= len(REQUIRED_CALENDAR_FEATURES)

    # Lag features check
    lag_present = [c for c in REQUIRED_LAG_FEATURES if c in present_cols]
    lag_coverage = len(lag_present) / len(REQUIRED_LAG_FEATURES) if REQUIRED_LAG_FEATURES else 0.0

    # SGDFNet coverage — distinguish real vs effective
    sgdfnet_effective = df.attrs.get("sgdfnet_effective_coverage",
                                      df.attrs.get("sgdfnet_coverage", 0.0))
    sgdfnet_real = df.attrs.get("sgdfnet_real_coverage",
                                 df.attrs.get("sgdfnet_coverage", 0.0))
    sgdfnet_missing = df.attrs.get("sgdfnet_missing_rows", 0)
    sgdfnet_fallback = df.attrs.get("sgdfnet_fallback_used", False)
    sgdfnet_source = df.attrs.get("sgdfnet_source", "unknown")
    sgdfnet_fallback_count = df.attrs.get("sgdfnet_fallback_count", 0)

    # Leakage check
    leakage_ok = check_leakage(df)

    # Formal readiness with new verdict rules
    n_features = len(all_present)
    n_req_missing = len(req_missing)

    # FORMAL_READY: real SGDFNet, no fallback
    if (n_features >= 25 and sgdfnet_real >= 95.0
            and not sgdfnet_fallback
            and leakage_ok
            and n_req_missing <= 3):
        verdict = "FORMAL_READY"
    # FALLBACK_READY: effective coverage via fallback
    elif (n_features >= 25 and sgdfnet_effective >= 95.0
          and sgdfnet_fallback
          and leakage_ok):
        verdict = "FALLBACK_READY"
    # PARTIAL_READY: real SGDFNet at partial coverage
    elif n_features >= 15 and sgdfnet_real >= 80.0 and not sgdfnet_fallback:
        verdict = "PARTIAL_READY"
    else:
        verdict = "NOT_READY"

    return {
        "n_features": n_features,
        "required_present": req_present,
        "required_missing": req_missing,
        "n_required_missing": n_req_missing,
        "optional_present": opt_present,
        "n_optional_present": len(opt_present),
        "sgdfnet_effective_coverage": sgdfnet_effective,
        "sgdfnet_real_coverage": sgdfnet_real,
        "sgdfnet_missing_rows": sgdfnet_missing,
        "sgdfnet_fallback_used": sgdfnet_fallback,
        "sgdfnet_fallback_count": sgdfnet_fallback_count,
        "sgdfnet_source": sgdfnet_source,
        "calendar_feature_generated": calendar_ok,
        "calendar_features_present": cal_present,
        "lag_feature_coverage": lag_coverage,
        "lag_features_present": lag_present,
        "leakage_checked": leakage_ok,
        "formal_train_ready": verdict == "FORMAL_READY",
        "verdict": verdict,
        "feature_version": FEATURE_VERSION,
    }
