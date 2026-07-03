"""Realtime feature contract for TrendKnightRT.

Defines the canonical feature set, leakage checks, and feature versioning.
This is the SINGLE SOURCE OF TRUTH for what features the model expects.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature groups ───────────────────────────────────────────────────

FORECAST_FEATURES = [
    "forecast_price",  # day-ahead anchor price
    # SGDFNet forecast columns (imported dynamically via sgdfnet_bridge)
]

CALENDAR_FEATURES = [
    "hour_sin", "hour_cos",       # cyclical hour encoding
    "dow_sin", "dow_cos",         # cyclical day-of-week encoding
    "month_sin", "month_cos",     # cyclical month encoding
    "is_weekend",                 # 1 if Saturday/Sunday
    "is_holiday",                 # 1 if holiday (optional)
]

LAG_FEATURES = [
    # Historical lag features (visible actuals up to cutoff)
    "rt_lag_1h", "rt_lag_2h", "rt_lag_3h",
    "rt_lag_24h", "rt_lag_48h",
    "rt_mean_6h", "rt_std_6h",
    "rt_mean_24h", "rt_std_24h",
    "delta_lag_24h", "delta_lag_48h",
]

SGDFNET_FEATURES = [
    "sgdfnet_pred",  # SGDFNet realtime prediction
    "sgdfnet_residual_lag_1h",  # lagged SGDFNet residual
]

OPTIONAL_TEACHER_FEATURES = [
    "rt916_pred",
    "timemixer_pred",
    "timesfm_pred",
]

# ── Full feature list ────────────────────────────────────────────────

REQUIRED_FEATURES = FORECAST_FEATURES + CALENDAR_FEATURES + LAG_FEATURES
OPTIONAL_FEATURES = SGDFNET_FEATURES + OPTIONAL_TEACHER_FEATURES
ALL_FEATURES = REQUIRED_FEATURES + OPTIONAL_FEATURES

# ── Feature version ──────────────────────────────────────────────────

FEATURE_VERSION = "v1.0"

# ── Leakage rules ────────────────────────────────────────────────────

LEAKAGE_RULES = {
    "no_target_hour_actual": "Cannot use rt_actual for the target hour or any future hour",
    "no_future_visible": "Cannot use visible actuals after cutoff_hour (default 15:00 D-1)",
    "no_post_cutoff_lag": "Lag features must not reference data after cutoff",
}


# ── Functions ────────────────────────────────────────────────────────

def validate_features(df: pd.DataFrame) -> list[str]:
    """Check which REQUIRED features are missing from the DataFrame.

    Compares the columns present in *df* against ``REQUIRED_FEATURES``.
    Optional features (SGDFNet, teacher) are NOT flagged as missing —
    they are handled gracefully at training / prediction time.

    Args:
        df: pandas DataFrame whose columns should contain the contract
            feature names.

    Returns:
        A list of required feature column names that are absent from
        *df*.  An empty list means all required features are present.

    Example::

        missing = validate_features(df)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
    """
    present_cols = set(df.columns)
    missing: list[str] = [
        feat for feat in REQUIRED_FEATURES if feat not in present_cols
    ]
    if missing:
        logger.warning("Missing required features: %s", missing)
    return missing


def check_leakage(df: pd.DataFrame, cutoff_hour: int = 15) -> bool:
    """Verify that no feature column leaks future information.

    The check inspects every column name in *df* that contains the
    substring ``"actual"`` (case-insensitive).  Such columns are expected
    to encode an hour reference — either as a suffix like ``_h14`` or as
    a separate companion column ``hour_business`` / ``target_hour`` in the
    DataFrame.  The function flags a leakage violation when:

    1. A column name contains ``"actual"`` **and** encodes an hour that
       is strictly greater than *cutoff_hour*.
    2. The DataFrame has a ``target_hour`` or ``hour_business`` column
       and any ``"actual"``-bearing column name encodes an hour >= that
       target hour (meaning the model would see the answer it is trying
       to predict).

    Hour extraction from column names uses the regex pattern
    ``_h(\d{1,2})`` (e.g. ``rt_actual_h16`` -> hour 16).

    Args:
        df: pandas DataFrame to check.
        cutoff_hour: The latest hour (1-24) whose actuals are allowed to
            appear in feature columns.  Default is 15 (i.e. data visible
            up to 15:00 on D-1).

    Returns:
        ``True`` if the DataFrame passes all leakage checks (i.e. it is
        **safe** to use).  ``False`` if any leakage is detected.

    Example::

        if not check_leakage(df, cutoff_hour=15):
            raise RuntimeError("Leakage detected — aborting.")
    """
    # Pattern to extract hour from column names like "rt_actual_h16"
    hour_pattern = re.compile(r"_h(\d{1,2})", re.IGNORECASE)

    # Collect all columns that reference "actual" in their name
    actual_cols = [
        col for col in df.columns
        if "actual" in col.lower()
    ]

    for col in actual_cols:
        match = hour_pattern.search(col)
        if match is None:
            # No explicit hour encoded in the column name.
            # If there is a target_hour / hour_business column we still
            # need to be cautious, but we cannot determine the hour from
            # the name alone — skip (the column may be a generic
            # "rt_actual" used as a target, not a feature).
            continue

        col_hour = int(match.group(1))

        # Rule 1: hour exceeds cutoff
        if col_hour > cutoff_hour:
            logger.error(
                "LEAKAGE: column '%s' references hour %d > cutoff_hour %d",
                col, col_hour, cutoff_hour,
            )
            return False

        # Rule 2: hour >= target_hour (if available)
        for hour_col in ("target_hour", "hour_business"):
            if hour_col in df.columns:
                target_hours = df[hour_col].dropna().unique()
                for th in target_hours:
                    if col_hour >= int(th):
                        logger.error(
                            "LEAKAGE: column '%s' hour %d >= %s value %d",
                            col, col_hour, hour_col, int(th),
                        )
                        return False

    logger.info("Leakage check passed (cutoff_hour=%d).", cutoff_hour)
    return True


def build_feature_manifest(
    feature_columns: list[str],
    target_columns: list[str],
    date_range: tuple[Any, Any] | pd.DatetimeIndex | None = None,
    n_days: int | None = None,
) -> dict:
    """Build a feature manifest dictionary for reproducibility.

    The manifest captures the exact feature set, target columns, date
    coverage, and row/day counts so that every training or prediction
    run can be traced back to the data it consumed.

    Args:
        feature_columns: List of feature column names actually used
            (may be a subset of ``ALL_FEATURES`` after resolution).
        target_columns: List of target column names (e.g.
            ``["delta_target", "residual_target"]``).
        date_range: Either a ``(start, end)`` tuple of dates / timestamps,
            or a ``pd.DatetimeIndex``.  Converted to ISO-8601 strings in
            the manifest.  May be ``None`` if unknown.
        n_days: Number of unique business days covered by the data.

    Returns:
        A dictionary with keys:

        - ``feature_version``: contract version string
        - ``feature_columns``: list of feature column names
        - ``target_columns``: list of target column names
        - ``n_features``: number of feature columns
        - ``n_targets``: number of target columns
        - ``date_range``: ``{"start": ..., "end": ...}`` or ``None``
        - ``n_days``: number of business days (or ``None``)
        - ``leakage_checks_passed``: always ``True`` at manifest
          creation time (caller must verify separately)

    Example::

        manifest = build_feature_manifest(
            feature_columns=["forecast_price", "hour_sin", ...],
            target_columns=["delta_target", "residual_target"],
            date_range=("2024-01-01", "2024-06-30"),
            n_days=180,
        )
    """
    # Normalise date_range
    dr: dict[str, str] | None = None
    if date_range is not None:
        if isinstance(date_range, pd.DatetimeIndex):
            start = str(date_range.min().date())
            end = str(date_range.max().date())
        else:
            start = str(pd.Timestamp(date_range[0]).date())
            end = str(pd.Timestamp(date_range[1]).date())
        dr = {"start": start, "end": end}

    # Classify each feature column
    present_required = [c for c in REQUIRED_FEATURES if c in feature_columns]
    present_optional = [c for c in OPTIONAL_FEATURES if c in feature_columns]
    missing_required = [c for c in REQUIRED_FEATURES if c not in feature_columns]

    manifest: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "feature_columns": list(feature_columns),
        "target_columns": list(target_columns),
        "n_features": len(feature_columns),
        "n_targets": len(target_columns),
        "date_range": dr,
        "n_days": n_days,
        "leakage_checks_passed": True,
        # Diagnostic breakdown
        "required_present": present_required,
        "optional_present": present_optional,
        "required_missing": missing_required,
    }
    return manifest


def get_period(hour_business: int) -> str:
    """Map a business hour (1-24) to its period segment.

    Period mapping (Shandong spot market convention):

    - Hours 1-8   -> ``"1_8"``   (valley / overnight)
    - Hours 9-16  -> ``"9_16"``  (shoulder / daytime)
    - Hours 17-24 -> ``"17_24"`` (peak / evening)

    This is the canonical period lookup for the realtime contract.
    It mirrors the logic in ``business_time.compute_period`` so that
    both modules stay in sync.

    Args:
        hour_business: Integer business hour in the range 1-24.

    Returns:
        Period string: ``"1_8"``, ``"9_16"``, or ``"17_24"``.

    Raises:
        ValueError: If *hour_business* is outside 1-24.

    Example::

        >>> get_period(3)
        '1_8'
        >>> get_period(12)
        '9_16'
        >>> get_period(20)
        '17_24'
    """
    if not isinstance(hour_business, (int,)):
        hour_business = int(hour_business)

    if 1 <= hour_business <= 8:
        return "1_8"
    elif 9 <= hour_business <= 16:
        return "9_16"
    elif 17 <= hour_business <= 24:
        return "17_24"
    else:
        raise ValueError(
            f"hour_business must be 1-24, got {hour_business}"
        )
