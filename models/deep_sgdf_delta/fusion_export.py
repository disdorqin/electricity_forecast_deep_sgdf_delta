"""Fusion export module for TrendKnight-X -> mainline system handoff.

Defines the fusion pack schema that bridges the deep trend model output to
the final price prediction pipeline.  The fusion pack carries everything
the downstream modules (日前线, CatBoost, 零样本模型, 产差模块, 负价模块)
need to produce the final realtime price forecast.

Two pack modes:
  - **Online pack**: no eval columns (y_true, residuals) -- safe for serving.
  - **Eval pack**: includes y_true and residual columns for offline analysis.

Typical usage::

    from models.deep_sgdf_delta.fusion_export import (
        FUSION_COLUMNS,
        validate_fusion_pack,
        build_fusion_row,
        strip_eval_columns,
        add_eval_columns,
    )

    row = build_fusion_row(
        business_day="2026-03-15",
        hour_business=10,
        ds="2026-03-15 09:00:00",
        trend_pred=320.5,
        trend_delta_pred=45.2,
        trend_confidence=0.78,
        shock_sensitivity=0.32,
        teacher_used="sgdfnet",
        sgdfnet_pred=318.0,
        rt916_pred=322.1,
        timemixer_pred=315.7,
        runtime_profile="v3_teacher_residual",
    )
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column registry ──────────────────────────────────────────────────

FUSION_COLUMNS: list[str] = [
    # ── Identity ──
    "business_day",        # date of the trading day (Timestamp)
    "hour_business",       # business hour 1-24 (int)
    "period",              # segment label: "1_8" | "9_16" | "17_24"
    "ds",                  # wall-clock timestamp (Timestamp)
    "model_name",          # deep model identifier, e.g. "trendknight_x"

    # ── Trend prediction ──
    "trend_pred",          # final realtime price prediction from deep model
    "trend_delta_pred",    # predicted delta (rt - da) from deep model
    "trend_confidence",    # confidence score in [0.1, 0.95]
    "shock_sensitivity",   # shock sensitivity score in [0, 1]

    # ── Teacher info ──
    "teacher_used",        # which teacher was used: "none" | "sgdfnet" | "sgdfnet+rt916" | etc.
    "sgdfnet_pred",        # SGDFNet teacher prediction (NaN if unavailable)
    "rt916_pred",          # RT916 teacher prediction (NaN if unavailable)
    "timemixer_pred",      # TimeMixer teacher prediction (NaN if unavailable)

    # ── Runtime metadata ──
    "runtime_profile",     # profile name used: "v3_fast_tcn" | "v3_multiscale_tcn" | etc.
]

# Columns that are only meaningful during offline evaluation
_EVAL_ONLY_COLUMNS: list[str] = [
    "y_true",                  # actual realtime price (ground truth)
    "residual_for_spike",      # y_true - trend_pred (for spike module)
    "residual_for_negative",   # y_true - trend_pred (for negative price module)
]

# Period boundaries
_PERIOD_BOUNDS: dict[str, tuple[int, int]] = {
    "1_8": (1, 8),
    "9_16": (9, 16),
    "17_24": (17, 24),
}


# ── Helpers ──────────────────────────────────────────────────────────

def _hour_to_period(hour: int) -> str:
    """Map an integer hour (1-24) to its period label."""
    for label, (lo, hi) in _PERIOD_BOUNDS.items():
        if lo <= hour <= hi:
            return label
    raise ValueError(f"hour_business must be in 1..24, got {hour}")


def _safe_float(val: Any, default: float = float("nan")) -> float:
    """Convert a value to float, returning *default* for None / NaN-like."""
    if val is None:
        return default
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return default


# ── Validation ───────────────────────────────────────────────────────

def validate_fusion_pack(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a fusion pack DataFrame against the schema.

    Returns
    -------
    (is_valid, errors)
        ``is_valid`` is True when no errors are found.
        ``errors`` is a list of human-readable error strings.
    """
    errors: list[str] = []

    # 1. Required columns
    missing = [c for c in FUSION_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # 2. No NaN in critical identity columns
    for col in ("business_day", "ds", "model_name"):
        if col in df.columns:
            n_nan = int(df[col].isna().sum())
            if n_nan > 0:
                errors.append(f"Column '{col}' has {n_nan} NaN values")

    # 3. hour_business range
    if "hour_business" in df.columns:
        hours = pd.to_numeric(df["hour_business"], errors="coerce")
        out_of_range = hours[(hours < 1) | (hours > 24)]
        if len(out_of_range) > 0:
            errors.append(
                f"hour_business has {len(out_of_range)} values outside [1, 24]"
            )

    # 4. period consistency
    if "period" in df.columns and "hour_business" in df.columns:
        hours = pd.to_numeric(df["hour_business"], errors="coerce").dropna().astype(int)
        valid_hours = hours[(hours >= 1) & (hours <= 24)]
        if len(valid_hours) > 0:
            expected_periods = valid_hours.apply(_hour_to_period)
            actual_periods = df.loc[valid_hours.index, "period"].astype(str)
            mismatch = (expected_periods != actual_periods).sum()
            if mismatch > 0:
                errors.append(
                    f"period column mismatches hour_business for {mismatch} rows"
                )

    # 5. trend_confidence range
    if "trend_confidence" in df.columns:
        conf = pd.to_numeric(df["trend_confidence"], errors="coerce")
        out_of_range = conf[conf.notna() & ((conf < 0) | (conf > 1))]
        if len(out_of_range) > 0:
            errors.append(
                f"trend_confidence has {len(out_of_range)} values outside [0, 1]"
            )

    # 6. shock_sensitivity range
    if "shock_sensitivity" in df.columns:
        shock = pd.to_numeric(df["shock_sensitivity"], errors="coerce")
        out_of_range = shock[shock.notna() & ((shock < 0) | (shock > 1))]
        if len(out_of_range) > 0:
            errors.append(
                f"shock_sensitivity has {len(out_of_range)} values outside [0, 1]"
            )

    # 7. teacher_used valid values
    if "teacher_used" in df.columns:
        valid_teachers = {"none", "sgdfnet", "rt916", "timemixer",
                          "sgdfnet+rt916", "sgdfnet+timemixer",
                          "rt916+timemixer", "sgdfnet+rt916+timemixer",
                          "sgdfnet+rt916+timemixer"}
        actual = set(df["teacher_used"].dropna().unique())
        invalid = actual - valid_teachers
        if invalid:
            errors.append(f"teacher_used has invalid values: {invalid}")

    return (len(errors) == 0, errors)


# ── Row builder ──────────────────────────────────────────────────────

def build_fusion_row(
    business_day: Any,
    hour_business: int,
    ds: Any,
    trend_pred: float,
    trend_delta_pred: float,
    trend_confidence: float,
    shock_sensitivity: float,
    model_name: str = "trendknight_x",
    teacher_used: str = "none",
    sgdfnet_pred: Optional[float] = None,
    rt916_pred: Optional[float] = None,
    timemixer_pred: Optional[float] = None,
    runtime_profile: str = "v3_teacher_residual",
) -> dict[str, Any]:
    """Build a single fusion pack row as a plain dict.

    Parameters
    ----------
    business_day : Timestamp-like
        The business day (date only).
    hour_business : int
        Business hour in 1..24.
    ds : Timestamp-like
        Wall-clock timestamp.
    trend_pred : float
        Final realtime price prediction.
    trend_delta_pred : float
        Predicted delta (rt - da).
    trend_confidence : float
        Confidence score in [0.1, 0.95].
    shock_sensitivity : float
        Shock sensitivity in [0, 1].
    model_name : str
        Model identifier string.
    teacher_used : str
        Which teacher(s) were used in the prediction.
    sgdfnet_pred, rt916_pred, timemixer_pred : float or None
        Teacher predictions (None / NaN when unavailable).
    runtime_profile : str
        Runtime profile name.

    Returns
    -------
    dict conforming to FUSION_COLUMNS.
    """
    hour_business = int(hour_business)
    period = _hour_to_period(hour_business)

    return {
        "business_day": pd.Timestamp(business_day),
        "hour_business": hour_business,
        "period": period,
        "ds": pd.Timestamp(ds),
        "model_name": str(model_name),
        "trend_pred": _safe_float(trend_pred),
        "trend_delta_pred": _safe_float(trend_delta_pred),
        "trend_confidence": float(np.clip(_safe_float(trend_confidence, 0.5), 0.0, 1.0)),
        "shock_sensitivity": float(np.clip(_safe_float(shock_sensitivity, 0.0), 0.0, 1.0)),
        "teacher_used": str(teacher_used),
        "sgdfnet_pred": _safe_float(sgdfnet_pred),
        "rt916_pred": _safe_float(rt916_pred),
        "timemixer_pred": _safe_float(timemixer_pred),
        "runtime_profile": str(runtime_profile),
    }


# ── DataFrame builders ───────────────────────────────────────────────

def build_fusion_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble a list of row dicts into a column-ordered DataFrame.

    The returned DataFrame has exactly the columns in FUSION_COLUMNS
    in canonical order.  Missing keys are filled with NaN.
    """
    if not rows:
        return pd.DataFrame(columns=FUSION_COLUMNS)

    df = pd.DataFrame(rows)

    # Ensure all columns exist
    for col in FUSION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Enforce canonical column order
    df = df[FUSION_COLUMNS].copy()

    # Type casting
    df["business_day"] = pd.to_datetime(df["business_day"])
    df["hour_business"] = pd.to_numeric(df["hour_business"], errors="coerce").astype("Int64")
    df["period"] = df["period"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    df["model_name"] = df["model_name"].astype(str)
    df["teacher_used"] = df["teacher_used"].astype(str)
    df["runtime_profile"] = df["runtime_profile"].astype(str)

    float_cols = [
        "trend_pred", "trend_delta_pred", "trend_confidence",
        "shock_sensitivity", "sgdfnet_pred", "rt916_pred", "timemixer_pred",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


# ── Eval column management ───────────────────────────────────────────

def strip_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove eval-only columns, producing an online-safe fusion pack.

    Columns not present are silently ignored.
    """
    return df.drop(columns=_EVAL_ONLY_COLUMNS, errors="ignore").copy()


def add_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add eval-only columns derived from ``y_true`` and ``trend_pred``.

    Requires that both ``y_true`` and ``trend_pred`` are present in *df*.
    Rows where ``y_true`` is NaN keep NaN in all eval columns.

    Columns added:
      - ``y_true``: kept as-is (must already exist)
      - ``residual_for_spike``: y_true - trend_pred
      - ``residual_for_negative``: y_true - trend_pred
    """
    if "y_true" not in df.columns:
        raise ValueError("add_eval_columns requires 'y_true' column")
    if "trend_pred" not in df.columns:
        raise ValueError("add_eval_columns requires 'trend_pred' column")

    out = df.copy()
    yt = pd.to_numeric(out["y_true"], errors="coerce")
    trend = pd.to_numeric(out["trend_pred"], errors="coerce")
    residual = yt - trend

    # Where y_true is NaN, residuals should be NaN
    out["residual_for_spike"] = residual
    out["residual_for_negative"] = residual

    return out


# ── Conversion from prediction CSV ───────────────────────────────────

def convert_predictions_to_fusion(
    pred_df: pd.DataFrame,
    model_name: str = "trendknight_x",
    runtime_profile: str = "v3_teacher_residual",
    teacher_status: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert a deep model predictions CSV to fusion pack format.

    Expected input columns (from predict_v3 or output_contract):
      - business_day, hour (or hour_business), da_anchor
      - delta_pred (or deep_delta_pred)
      - rt_pred (or deep_rt_pred or blend_pred or trend_pred)
      - confidence (or trend_confidence)
      - shock_sensitivity (optional)
      - sgdfnet_pred (optional)
      - rt916_pred (optional)
      - timemixer_pred (optional)

    Returns a DataFrame conforming to FUSION_COLUMNS.
    """
    df = pred_df.copy()

    # Normalize column names
    col_map: dict[str, str] = {}
    if "hour" in df.columns and "hour_business" not in df.columns:
        col_map["hour"] = "hour_business"
    if "deep_delta_pred" in df.columns and "trend_delta_pred" not in df.columns:
        col_map["deep_delta_pred"] = "trend_delta_pred"
    if "deep_rt_pred" in df.columns and "trend_pred" not in df.columns:
        col_map["deep_rt_pred"] = "trend_pred"
    if "blend_pred" in df.columns and "trend_pred" not in df.columns:
        col_map["blend_pred"] = "trend_pred"
    if "trend_model_name" in df.columns:
        col_map["trend_model_name"] = "model_name"
    if df.columns.difference(col_map.keys()).size == len(df.columns):
        pass  # no renaming needed

    df = df.rename(columns=col_map)

    # Derive missing columns
    if "trend_delta_pred" not in df.columns:
        if "trend_pred" in df.columns and "da_anchor" in df.columns:
            df["trend_delta_pred"] = df["trend_pred"] - df["da_anchor"]
        else:
            df["trend_delta_pred"] = np.nan

    if "trend_confidence" not in df.columns:
        df["trend_confidence"] = 0.5  # default

    if "shock_sensitivity" not in df.columns:
        df["shock_sensitivity"] = 0.0  # default

    if "teacher_used" not in df.columns:
        if teacher_status:
            available = [k for k, v in teacher_status.items()
                         if isinstance(v, dict) and v.get("availability") == "available"]
            df["teacher_used"] = "+".join(available) if available else "none"
        else:
            df["teacher_used"] = "none"

    for col in ("sgdfnet_pred", "rt916_pred", "timemixer_pred"):
        if col not in df.columns:
            df[col] = np.nan

    if "model_name" not in df.columns:
        df["model_name"] = model_name

    if "runtime_profile" not in df.columns:
        df["runtime_profile"] = runtime_profile

    # Build fusion DataFrame
    rows: list[dict] = []
    for _, row in df.iterrows():
        hour = int(row.get("hour_business", row.get("hour", 1)))
        bd = row.get("business_day", pd.NaT)
        ds = row.get("ds", pd.NaT)
        if pd.isna(ds):
            ds = pd.Timestamp(bd) + pd.Timedelta(hours=hour - 1)

        rows.append(build_fusion_row(
            business_day=bd,
            hour_business=hour,
            ds=ds,
            trend_pred=row.get("trend_pred", np.nan),
            trend_delta_pred=row.get("trend_delta_pred", np.nan),
            trend_confidence=row.get("trend_confidence", 0.5),
            shock_sensitivity=row.get("shock_sensitivity", 0.0),
            model_name=row.get("model_name", model_name),
            teacher_used=row.get("teacher_used", "none"),
            sgdfnet_pred=row.get("sgdfnet_pred"),
            rt916_pred=row.get("rt916_pred"),
            timemixer_pred=row.get("timemixer_pred"),
            runtime_profile=row.get("runtime_profile", runtime_profile),
        ))

    return build_fusion_dataframe(rows)


# ── Public API ───────────────────────────────────────────────────────

__all__ = [
    "FUSION_COLUMNS",
    "validate_fusion_pack",
    "build_fusion_row",
    "build_fusion_dataframe",
    "strip_eval_columns",
    "add_eval_columns",
    "convert_predictions_to_fusion",
]
