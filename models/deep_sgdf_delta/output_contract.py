"""Output contract for DeepSGDFDelta predictions.csv.

Defines the standard schema that this model produces and that downstream
modules (spike detector, negative-price handler, ledger fusion, final
delivery) consume.  Every prediction row — whether generated during
offline evaluation or online serving — must conform to OUTPUT_COLUMNS.

Eval-only columns (flags and residuals that depend on y_true) are added
by ``add_eval_columns`` and stripped by ``strip_eval_columns`` so that
online predictions never carry stale or fabricated ground-truth signals.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column registry ──────────────────────────────────────────────────

OUTPUT_COLUMNS: list[str] = [
    "business_day",
    "hour_business",
    "period",
    "ds",
    "da_anchor",
    "y_true",
    "deep_delta_pred",
    "deep_rt_pred",
    "sgdfnet_pred",
    "blend_pred",
    "trend_pred",
    "trend_model_name",
    "trend_confidence",
    "normal_trend_flag",
    "high_price_bucket_flag",
    "negative_bucket_flag",
    "residual_for_spike_module",
    "residual_for_negative_module",
]

EVAL_ONLY_COLUMNS: list[str] = [
    "high_price_bucket_flag",
    "negative_bucket_flag",
    "residual_for_spike_module",
    "residual_for_negative_module",
]

# Period boundaries (inclusive on both ends)
_PERIOD_BOUNDS = {
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


# ── Validation ───────────────────────────────────────────────────────


def validate_predictions(df: pd.DataFrame) -> list[str]:
    """Return a list of column names required by the contract but missing
    from *df*.  An empty list means the DataFrame is fully compliant."""
    present = set(df.columns)
    return [c for c in OUTPUT_COLUMNS if c not in present]


# ── Confidence scoring ───────────────────────────────────────────────


def compute_trend_confidence(
    deep_delta_pred: float,
    sgdfnet_pred: Optional[float],
    da_anchor: float,
) -> float:
    """Heuristic confidence score in [0.1, 0.95].

    Logic
    -----
    * **Deep-only** (``sgdfnet_pred`` is ``None`` or NaN):
      Start at 0.6 and penalise proportionally to how large the predicted
      delta is relative to the anchor price.  Large deltas are inherently
      less trustworthy.
    * **Blended** (SGDFNet available):
      Start at 0.5 and add up to 0.4 proportional to how closely the two
      models agree (relative to the anchor magnitude).

    The score is always clamped to [0.1, 0.95] so that downstream modules
    never see degenerate 0 or 1 values.
    """
    _MIN_ANCHOR = 50.0  # avoid division by near-zero anchors
    anchor_ref = max(abs(da_anchor), _MIN_ANCHOR)

    # Detect whether sgdfnet is usable
    sgdf_available = (
        sgdfnet_pred is not None
        and not (isinstance(sgdfnet_pred, float) and math.isnan(sgdfnet_pred))
        and not (isinstance(sgdfnet_pred, np.floating) and np.isnan(sgdfnet_pred))
    )

    if not sgdf_available:
        # Deep-only scoring
        delta_ratio = min(abs(deep_delta_pred) / anchor_ref, 1.0)
        confidence = 0.6 - 0.2 * delta_ratio
    else:
        # Two-model agreement scoring
        deep_rt = da_anchor + deep_delta_pred
        disagreement = abs(deep_rt - sgdfnet_pred) / anchor_ref
        agreement = max(0.0, 1.0 - disagreement)
        confidence = 0.5 + 0.4 * agreement

    return float(np.clip(confidence, 0.1, 0.95))


# ── Row builder ──────────────────────────────────────────────────────


def build_prediction_row(
    business_day: Any,
    hour: int,
    da_anchor: float,
    deep_delta_pred: float,
    *,
    sgdfnet_pred: Optional[float] = None,
    blend_mode: str = "deep_only",
    blend_weight: float = 0.5,
    trend_model_name: str = "DeepSGDFDelta_V2_tcn",
    ds: Any = None,
) -> dict:
    """Build a single prediction row as a plain ``dict``.

    Parameters
    ----------
    business_day : Timestamp-like
        The business day this hour belongs to.
    hour : int
        Business hour in 1..24.
    da_anchor : float
        Day-ahead anchor price for this hour.
    deep_delta_pred : float
        The deep model's predicted delta (rt - da).
    sgdfnet_pred : float or None
        SGDFNet's realtime price prediction (not delta).  ``None`` when
        SGDFNet is unavailable.
    blend_mode : str
        One of ``"deep_only"``, ``"sgdfnet_blend"``, ``"sgdfnet_residual"``.
    blend_weight : float
        Weight given to SGDFNet in ``sgdfnet_blend`` mode.
    trend_model_name : str
        Identifier string written to the output.
    ds : Timestamp-like or None
        The actual wall-clock timestamp.  Derived from *business_day* and
        *hour* when not supplied.

    Returns
    -------
    dict  – one row conforming to OUTPUT_COLUMNS (eval columns set to NaN).
    """
    hour = int(hour)
    period = _hour_to_period(hour)

    # Derived prices
    deep_rt_pred = float(da_anchor) + float(deep_delta_pred)

    # Blend
    if blend_mode == "deep_only" or sgdfnet_pred is None:
        blend_pred = deep_rt_pred
    elif blend_mode == "sgdfnet_blend":
        blend_pred = blend_weight * float(sgdfnet_pred) + (1.0 - blend_weight) * deep_rt_pred
    elif blend_mode == "sgdfnet_residual":
        # Deep predicts residual on top of SGDFNet
        blend_pred = float(sgdfnet_pred) + float(deep_delta_pred)
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode!r}")

    trend_pred = blend_pred

    # Timestamp derivation
    if ds is None:
        base = pd.Timestamp(business_day)
        ds = base + pd.Timedelta(hours=hour - 1)

    confidence = compute_trend_confidence(deep_delta_pred, sgdfnet_pred, da_anchor)

    return {
        "business_day": pd.Timestamp(business_day),
        "hour_business": hour,
        "period": period,
        "ds": pd.Timestamp(ds),
        "da_anchor": float(da_anchor),
        "y_true": float("nan"),  # unknown in prediction mode
        "deep_delta_pred": float(deep_delta_pred),
        "deep_rt_pred": deep_rt_pred,
        "sgdfnet_pred": float(sgdfnet_pred) if sgdfnet_pred is not None else float("nan"),
        "blend_pred": float(blend_pred),
        "trend_pred": float(trend_pred),
        "trend_model_name": str(trend_model_name),
        "trend_confidence": confidence,
        "normal_trend_flag": 1,  # assumed normal until eval reveals otherwise
        "high_price_bucket_flag": float("nan"),
        "negative_bucket_flag": float("nan"),
        "residual_for_spike_module": float("nan"),
        "residual_for_negative_module": float("nan"),
    }


# ── DataFrame builder ────────────────────────────────────────────────


def build_predictions_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Assemble a list of row dicts into a typed, column-ordered DataFrame.

    The returned DataFrame has exactly the columns in OUTPUT_COLUMNS,
    in the canonical order.  Missing keys in individual rows are filled
    with NaN.
    """
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)

    # Ensure all columns exist
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Enforce canonical column order
    df = df[OUTPUT_COLUMNS].copy()

    # ── Type casting ─────────────────────────────────────────────
    df["business_day"] = pd.to_datetime(df["business_day"])
    df["hour_business"] = pd.to_numeric(df["hour_business"], errors="coerce").astype("Int64")
    df["period"] = df["period"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])

    float_cols = [
        "da_anchor", "y_true", "deep_delta_pred", "deep_rt_pred",
        "sgdfnet_pred", "blend_pred", "trend_pred", "trend_confidence",
        "high_price_bucket_flag", "negative_bucket_flag",
        "residual_for_spike_module", "residual_for_negative_module",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df["normal_trend_flag"] = pd.to_numeric(df["normal_trend_flag"], errors="coerce").astype("Int64")
    df["trend_model_name"] = df["trend_model_name"].astype(str)

    return df


# ── Eval column helpers ──────────────────────────────────────────────

_SPIKE_THRESHOLD = 500.0


def add_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add evaluation-only columns derived from ``y_true``.

    Requires that ``y_true`` and ``trend_pred`` are present.  Rows where
    ``y_true`` is NaN keep NaN in all eval columns so that they are
    transparently skipped by downstream metric calculations.

    Columns added / overwritten:
      - ``high_price_bucket_flag``: 1 when |y_true| > 500
      - ``negative_bucket_flag``: 1 when y_true < 0
      - ``normal_trend_flag``: 1 when neither spike nor negative
      - ``residual_for_spike_module``: y_true - trend_pred
      - ``residual_for_negative_module``: y_true - trend_pred
    """
    out = df.copy()

    yt = pd.to_numeric(out["y_true"], errors="coerce")
    trend = pd.to_numeric(out["trend_pred"], errors="coerce")

    residual = yt - trend

    # Bucket flags (NaN-safe via pd.Series comparisons)
    high_price = np.where(yt.notna() & (yt.abs() > _SPIKE_THRESHOLD), 1, 0)
    negative = np.where(yt.notna() & (yt < 0), 1, 0)

    # Where y_true is NaN, flags should be NaN (not 0) to signal "unknown"
    ytrue_nan = yt.isna()
    high_price_series = pd.array(high_price, dtype="Int64")
    negative_series = pd.array(negative, dtype="Int64")
    high_price_series[ytrue_nan] = pd.NA
    negative_series[ytrue_nan] = pd.NA

    out["high_price_bucket_flag"] = high_price_series
    out["negative_bucket_flag"] = negative_series

    # Normal trend = not spike AND not negative
    normal = np.where(
        ytrue_nan,
        pd.NA,
        np.where((high_price == 0) & (negative == 0), 1, 0),
    )
    out["normal_trend_flag"] = pd.array(normal, dtype="Int64")

    # Residuals — NaN where y_true is NaN
    out["residual_for_spike_module"] = residual
    out["residual_for_negative_module"] = residual

    return out


def strip_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove eval-only columns, returning a DataFrame safe for online
    prediction delivery.  Columns not present are silently ignored."""
    return df.drop(columns=EVAL_ONLY_COLUMNS, errors="ignore").copy()
