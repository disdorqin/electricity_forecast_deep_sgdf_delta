"""Solar916 feature engineering — Phase 8 no-leak version.

Builds feature columns for the 9_16 residual correction specialist.
All features use only information available at prediction time (no future actuals).

CRITICAL FIXES (Phase 8):
  - Lag features use merge-based same-hour previous-day lookup (not simple shift)
  - Rolling features use shift(1) to exclude current row (no target leakage)
  - Same-hour rolling uses groupby(hour_business) + shift(1) + rolling
  - Features must be computed on FULL dataset before filtering to 9_16

Feature categories:
  - Temporal: hour_business, weekday, month
  - Price context: da_anchor, sgdfnet_pred
  - Forecast features: forecast_load, forecast_wind, forecast_solar,
                        forecast_new_energy, bidding_space
  - Derived: net_load, renewable_share
  - Lag features: delta_lag_24, delta_lag_168, residual_lag_24, residual_lag_168
  - Rolling stats: rolling_residual_mean_7d, rolling_residual_std_7d,
                   same_hour_residual_mean_7d, same_hour_residual_std_7d
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column mapping from raw Chinese names to canonical English ───────
RAW_COL_MAP = {
    "光伏总加预测值": "forecast_solar",
    "风电总加预测值": "forecast_wind",
    "新能源总加预测值": "forecast_new_energy",
    "竞价空间预测值": "bidding_space",
    "直调负荷预测值": "forecast_load",
    "日前电价": "da_price",
    "实时电价": "rt_price",
    "系统负荷": "system_load",
    "净负荷": "net_load_raw",
}

# All candidate features the spec requires
ALL_CANDIDATE_FEATURES = [
    "hour_business",
    "weekday",
    "month",
    "da_anchor",
    "sgdfnet_pred",
    "forecast_load",
    "forecast_wind",
    "forecast_solar",
    "forecast_new_energy",
    "bidding_space",
    "net_load",
    "renewable_share",
    "delta_lag_24",
    "delta_lag_168",
    "residual_lag_24",
    "residual_lag_168",
    "rolling_residual_mean_7d",
    "rolling_residual_std_7d",
    "same_hour_residual_mean_7d",
    "same_hour_residual_std_7d",
]


def detect_raw_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detect which raw columns exist and return mapping to canonical names.

    Returns dict: canonical_name -> actual_column_name
    """
    detected = {}
    for raw_name, canonical in RAW_COL_MAP.items():
        if raw_name in df.columns:
            detected[canonical] = raw_name
    return detected


def _build_lag_features_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag features using merge-based same-hour previous-day lookup.

    This is the NO-LEAK approach: instead of shift(1) which just gets the
    previous row (possibly same day, different hour), we explicitly look up
    the value at (business_day - N, same hour_business).

    delta_lag_24: previous business_day, same hour_business
    delta_lag_168: business_day - 7, same hour_business
    residual_lag_24: previous business_day, same hour_business
    residual_lag_168: business_day - 7, same hour_business
    """
    df = df.copy()
    df["business_day"] = pd.to_datetime(df["business_day"])

    # Build history lookup frame
    history = df[["business_day", "hour_business", "delta", "sgdfnet_residual"]].copy()
    history = history.dropna(subset=["business_day", "hour_business"])

    # ── lag_24: previous business_day, same hour ─────────────────────
    lag24 = history[["business_day", "hour_business", "delta", "sgdfnet_residual"]].copy()
    lag24["business_day"] = lag24["business_day"] + pd.Timedelta(days=1)
    lag24 = lag24.rename(columns={
        "delta": "delta_lag_24",
        "sgdfnet_residual": "residual_lag_24",
    })
    df = df.merge(
        lag24[["business_day", "hour_business", "delta_lag_24", "residual_lag_24"]],
        on=["business_day", "hour_business"],
        how="left",
    )

    # ── lag_168: business_day - 7, same hour ─────────────────────────
    lag168 = history[["business_day", "hour_business", "delta", "sgdfnet_residual"]].copy()
    lag168["business_day"] = lag168["business_day"] + pd.Timedelta(days=7)
    lag168 = lag168.rename(columns={
        "delta": "delta_lag_168",
        "sgdfnet_residual": "residual_lag_168",
    })
    df = df.merge(
        lag168[["business_day", "hour_business", "delta_lag_168", "residual_lag_168"]],
        on=["business_day", "hour_business"],
        how="left",
    )

    return df


def _build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rolling residual features with NO target leakage.

    CRITICAL: All rolling features must shift(1) first to exclude current row.
    This ensures the current row's residual (the target) does not leak into features.

    rolling_residual_mean_7d: mean of last 7 residuals BEFORE current row
    rolling_residual_std_7d: std of last 7 residuals BEFORE current row
    same_hour_residual_mean_7d: groupby hour, shift(1), rolling(7).mean()
    same_hour_residual_std_7d: groupby hour, shift(1), rolling(7).std()
    """
    df = df.copy()
    df = df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    # ── Global rolling (all hours, shift(1) to exclude current) ──────
    shifted_residual = df["sgdfnet_residual"].shift(1)
    df["rolling_residual_mean_7d"] = (
        shifted_residual.rolling(window=7, min_periods=1).mean()
    )
    df["rolling_residual_std_7d"] = (
        shifted_residual.rolling(window=7, min_periods=1).std().fillna(0)
    )

    # ── Same-hour rolling (groupby hour, shift(1), rolling) ──────────
    # For each hour_business group, shift(1) then rolling(7)
    grouped = df.groupby("hour_business")["sgdfnet_residual"]
    shifted_by_hour = grouped.shift(1)
    df["same_hour_residual_mean_7d"] = (
        df.groupby("hour_business")["sgdfnet_residual"]
        .shift(1)
        .groupby(df["hour_business"])
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["same_hour_residual_std_7d"] = (
        df.groupby("hour_business")["sgdfnet_residual"]
        .shift(1)
        .groupby(df["hour_business"])
        .rolling(window=7, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    return df


def build_solar916_features(
    df: pd.DataFrame,
    sgdfnet_predictions: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build all Solar916 features on a DataFrame.

    IMPORTANT: This function should be called on the FULL dataset (all hours)
    BEFORE filtering to 9_16. The lag/rolling features require the full
    24-hour context to be correct.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: business_day, hour_business, period, da_price, rt_price.
        Should contain ALL hours (not just 9_16) for correct lag computation.
    sgdfnet_predictions : pd.DataFrame, optional
        Must contain: business_day, hour_business, teacher_pred.
        If None, sgdfnet_pred and residual will be NaN.

    Returns
    -------
    (df_out, info) where info contains missing_features list.
    """
    df_out = df.copy()
    missing_features = []

    # Detect raw columns
    col_map = detect_raw_columns(df)

    # ── Rename raw columns to canonical ──────────────────────────────
    rename_map = {v: k for k, v in col_map.items()}
    df_out = df_out.rename(columns=rename_map)

    # ── SGDFNet predictions ──────────────────────────────────────────
    if sgdfnet_predictions is not None and not sgdfnet_predictions.empty:
        sgdf = sgdfnet_predictions.copy()
        sgdf["business_day"] = pd.to_datetime(sgdf["business_day"])
        sgdf_hour_col = "hour_business" if "hour_business" in sgdf.columns else "hour"

        pred_col = None
        for c in ("teacher_pred", "y_pred", "rt_hat"):
            if c in sgdf.columns:
                pred_col = c
                break

        if pred_col:
            sgdf_merge = sgdf[["business_day", sgdf_hour_col, pred_col]].rename(
                columns={sgdf_hour_col: "hour_business", pred_col: "sgdfnet_pred"}
            )
            df_out = df_out.merge(sgdf_merge, on=["business_day", "hour_business"], how="left")
        else:
            df_out["sgdfnet_pred"] = np.nan
    else:
        df_out["sgdfnet_pred"] = np.nan

    # ── Residual ─────────────────────────────────────────────────────
    df_out["sgdfnet_residual"] = df_out["rt_price"] - df_out["sgdfnet_pred"]

    # ── Temporal features ────────────────────────────────────────────
    bd = pd.to_datetime(df_out["business_day"])
    df_out["weekday"] = bd.dt.weekday  # 0=Monday
    df_out["month"] = bd.dt.month

    # ── Derived features ─────────────────────────────────────────────
    if "forecast_load" in df_out.columns and "forecast_new_energy" in df_out.columns:
        df_out["net_load"] = (
            pd.to_numeric(df_out["forecast_load"], errors="coerce")
            - pd.to_numeric(df_out["forecast_new_energy"], errors="coerce")
        )
    elif "forecast_load" in df_out.columns:
        df_out["net_load"] = pd.to_numeric(df_out["forecast_load"], errors="coerce")
    else:
        df_out["net_load"] = np.nan
        missing_features.append("forecast_load (needed for net_load)")

    has_solar = "forecast_solar" in df_out.columns
    has_wind = "forecast_wind" in df_out.columns
    if has_solar and has_wind and df_out["net_load"].notna().any():
        renewable = (
            pd.to_numeric(df_out["forecast_solar"], errors="coerce")
            + pd.to_numeric(df_out["forecast_wind"], errors="coerce")
        )
        df_out["renewable_share"] = renewable / (df_out["net_load"].abs() + 1e-6)
    else:
        df_out["renewable_share"] = np.nan
        if not has_solar:
            missing_features.append("forecast_solar")
        if not has_wind:
            missing_features.append("forecast_wind")

    for feat in ["forecast_solar", "forecast_new_energy", "bidding_space"]:
        if feat not in df_out.columns:
            missing_features.append(feat)

    # ── Delta (rt - da) ──────────────────────────────────────────────
    df_out["delta"] = df_out["rt_price"] - df_out["da_price"]

    # ── Lag features (Phase 8: merge-based, no-leak) ─────────────────
    df_out = _build_lag_features_merge(df_out)

    # ── Rolling features (Phase 8: shift(1) first, no-leak) ──────────
    df_out = _build_rolling_features(df_out)

    info = {
        "missing_features": list(set(missing_features)),
        "n_samples": len(df_out),
        "feature_columns": [c for c in ALL_CANDIDATE_FEATURES if c in df_out.columns],
    }

    return df_out, info


def write_feature_manifest(info: dict, output_path: str | Path) -> None:
    """Write feature_manifest.json."""
    manifest = {
        "feature_columns": info.get("feature_columns", []),
        "missing_features": info.get("missing_features", []),
        "n_samples": info.get("n_samples", 0),
        "target": "sgdfnet_residual",
        "target_description": "rt_actual - sgdfnet_pred",
        "phase": 8,
        "leak_fix": True,
    }
    Path(output_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Feature manifest written to %s", output_path)
