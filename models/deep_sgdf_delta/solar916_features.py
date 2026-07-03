"""Solar916 feature engineering.

Builds feature columns for the 9_16 residual correction specialist.
All features use only information available at prediction time (no future actuals).

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


def build_solar916_features(
    df: pd.DataFrame,
    sgdfnet_predictions: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build all Solar916 features on a DataFrame that already has
    business_day, hour_business, period, da_price, rt_price columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: business_day, hour_business, period, da_price, rt_price.
        May contain raw Chinese column names for forecast features.
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
    # Net load = forecast_load - forecast_new_energy (if both available)
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

    # Renewable share = (solar + wind) / net_load
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

    # Check other features
    for feat in ["forecast_solar", "forecast_new_energy", "bidding_space"]:
        if feat not in df_out.columns:
            missing_features.append(feat)

    # ── Delta (rt - da) ──────────────────────────────────────────────
    df_out["delta"] = df_out["rt_price"] - df_out["da_price"]

    # ── Lag features (need sorting by business_day + hour_business) ──
    df_out = df_out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    # delta_lag_24: same hour yesterday (24h ago = 1 business day)
    df_out["delta_lag_24"] = df_out["delta"].shift(1)
    # delta_lag_168: same hour last week (168h = 7 business days)
    df_out["delta_lag_168"] = df_out["delta"].shift(7)

    # residual_lag_24, residual_lag_168
    df_out["residual_lag_24"] = df_out["sgdfnet_residual"].shift(1)
    df_out["residual_lag_168"] = df_out["sgdfnet_residual"].shift(7)

    # ── Rolling stats (7-day window) ─────────────────────────────────
    # Rolling mean/std of residual over last 7 days
    df_out["rolling_residual_mean_7d"] = (
        df_out["sgdfnet_residual"].rolling(window=7, min_periods=1).mean()
    )
    df_out["rolling_residual_std_7d"] = (
        df_out["sgdfnet_residual"].rolling(window=7, min_periods=1).std()
    )

    # Same-hour residual mean/std over last 7 days
    # Group by hour_business and compute expanding stats
    for hour_val in range(9, 17):
        mask = df_out["hour_business"] == hour_val
        if mask.sum() == 0:
            continue
        idx = df_out.index[mask]
        rolling_mean = df_out.loc[idx, "sgdfnet_residual"].rolling(
            window=7, min_periods=1
        ).mean()
        rolling_std = df_out.loc[idx, "sgdfnet_residual"].rolling(
            window=7, min_periods=1
        ).std()
        df_out.loc[idx, "same_hour_residual_mean_7d"] = rolling_mean
        df_out.loc[idx, "same_hour_residual_std_7d"] = rolling_std

    # Fill NaN std with 0 (single-sample windows)
    df_out["same_hour_residual_std_7d"] = df_out["same_hour_residual_std_7d"].fillna(0)
    df_out["rolling_residual_std_7d"] = df_out["rolling_residual_std_7d"].fillna(0)

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
    }
    Path(output_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Feature manifest written to %s", output_path)
