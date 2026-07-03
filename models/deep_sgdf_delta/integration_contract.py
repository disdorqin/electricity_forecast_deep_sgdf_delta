"""Integration contract for mainline system接入.

Defines the data schema and validation rules for the trend prediction
output that will be consumed by:
  - Spike module (尖峰模块)
  - Negative price module (负价模块)
  - Ledger fusion (ledger融合)
  - Final delivery report

Two modes:
  1. Online prediction pack: no y_true, no eval-only residuals
  2. Eval pack: includes y_true and residual columns for analysis
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

# ── Online prediction pack columns (no y_true) ──────────────────────

ONLINE_PACK_COLUMNS = [
    "business_day",
    "hour_business",
    "period",
    "ds",
    "trend_pred",
    "trend_model_name",
    "trend_confidence",
    "deep_rt_pred",
    "sgdfnet_pred",
    "blend_pred",
    "da_anchor",
    "normal_trend_flag",
    "high_price_bucket_flag",
    "negative_bucket_flag",
]

# ── Eval pack columns (includes y_true and residuals) ───────────────

EVAL_EXTRA_COLUMNS = [
    "y_true",
    "residual_for_spike_module",
    "residual_for_negative_module",
]

EVAL_PACK_COLUMNS = ONLINE_PACK_COLUMNS + EVAL_EXTRA_COLUMNS

# ── Period mapping ───────────────────────────────────────────────────

def hour_to_period(hour: int) -> str:
    """Map hour_business (1-24) to period string."""
    if 1 <= hour <= 8:
        return "1_8"
    elif 9 <= hour <= 16:
        return "9_16"
    elif 17 <= hour <= 24:
        return "17_24"
    return "unknown"


# ── Validation ───────────────────────────────────────────────────────

def validate_online_pack(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate an online prediction pack DataFrame.

    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []

    # Check required columns
    missing = [c for c in ONLINE_PACK_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Must NOT contain y_true or eval-only columns
    leaked = [c for c in EVAL_EXTRA_COLUMNS if c in df.columns]
    if leaked:
        errors.append(f"Online pack must NOT contain eval columns: {leaked}")

    # Check no NaN in critical columns
    for col in ["business_day", "ds", "trend_pred", "da_anchor"]:
        if col in df.columns and df[col].isna().any():
            n_nan = int(df[col].isna().sum())
            errors.append(f"Column '{col}' has {n_nan} NaN values")

    # Check period values
    if "period" in df.columns:
        valid_periods = {"1_8", "9_16", "17_24"}
        actual_periods = set(df["period"].unique())
        invalid = actual_periods - valid_periods
        if invalid:
            errors.append(f"Invalid period values: {invalid}")

    # Check flag columns are binary
    for flag_col in ["normal_trend_flag", "high_price_bucket_flag", "negative_bucket_flag"]:
        if flag_col in df.columns:
            unique_vals = set(df[flag_col].dropna().unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                errors.append(f"Column '{flag_col}' has non-binary values: {unique_vals}")

    return (len(errors) == 0, errors)


def validate_eval_pack(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate an eval pack DataFrame.

    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []

    # Check required columns (online + eval extras)
    missing = [c for c in EVAL_PACK_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check no NaN in y_true
    if "y_true" in df.columns and df["y_true"].isna().any():
        n_nan = int(df["y_true"].isna().sum())
        errors.append(f"y_true has {n_nan} NaN values")

    return (len(errors) == 0, errors)


# ── Builders ─────────────────────────────────────────────────────────

def build_online_pack_row(
    business_day,
    hour_business: int,
    ds,
    trend_pred: float,
    trend_model_name: str,
    trend_confidence: float,
    deep_rt_pred: float,
    sgdfnet_pred: float,
    blend_pred: float,
    da_anchor: float,
    normal_trend_flag: bool = True,
    high_price_bucket_flag: bool = False,
    negative_bucket_flag: bool = False,
) -> dict:
    """Build a single row for the online prediction pack."""
    return {
        "business_day": business_day,
        "hour_business": hour_business,
        "period": hour_to_period(hour_business),
        "ds": ds,
        "trend_pred": trend_pred,
        "trend_model_name": trend_model_name,
        "trend_confidence": trend_confidence,
        "deep_rt_pred": deep_rt_pred,
        "sgdfnet_pred": sgdfnet_pred,
        "blend_pred": blend_pred,
        "da_anchor": da_anchor,
        "normal_trend_flag": int(normal_trend_flag),
        "high_price_bucket_flag": int(high_price_bucket_flag),
        "negative_bucket_flag": int(negative_bucket_flag),
    }


def add_eval_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add eval-only columns to a prediction DataFrame.

    Computes residual_for_spike_module and residual_for_negative_module
    from y_true and trend_pred.
    """
    result = df.copy()

    if "y_true" not in result.columns:
        raise ValueError("y_true column required for eval columns")

    if "trend_pred" not in result.columns:
        raise ValueError("trend_pred column required for eval columns")

    result["residual_for_spike_module"] = result["y_true"] - result["trend_pred"]
    result["residual_for_negative_module"] = result["y_true"] - result["trend_pred"]

    return result


def strip_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove eval-only columns, producing an online-safe pack."""
    result = df.copy()
    for col in EVAL_EXTRA_COLUMNS:
        if col in result.columns:
            result = result.drop(columns=[col])
    return result
