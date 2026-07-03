"""Unified business-day time alignment for Shandong electricity market.

Rule (Shandong spot market convention):
  - Timestamp D 01:00 ~ 23:00 → business_day = D, hour_business = 1 ~ 23
  - Timestamp D 00:00          → business_day = D-1, hour_business = 24

Period mapping:
  - hour_business 1-8   → "1_8"
  - hour_business 9-16  → "9_16"
  - hour_business 17-24 → "17_24"

This module is the SINGLE SOURCE OF TRUTH for business-day alignment.
All scripts and teacher adapters MUST use this instead of hand-rolled logic.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def add_business_time_columns(
    df: pd.DataFrame,
    timestamp_col: str = "ds",
    business_day_col: str = "business_day",
    hour_col: str = "hour_business",
    period_col: str = "period",
) -> pd.DataFrame:
    """Add business_day, hour_business, and period columns to a DataFrame.

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.
        business_day_col: Output column name for business day.
        hour_col: Output column name for business hour (1-24).
        period_col: Output column name for period segment.

    Returns:
        DataFrame with new columns added (modified in-place copy).
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    # Extract date and hour
    date_part = ts.dt.normalize()
    hour_of_day = ts.dt.hour

    # Rule: hour 0 → business_day = D-1, hour_business = 24
    #        hour 1-23 → business_day = D, hour_business = hour
    is_midnight = hour_of_day == 0
    df[hour_col] = np.where(is_midnight, 24, hour_of_day)
    df[business_day_col] = np.where(
        is_midnight,
        date_part - pd.Timedelta(days=1),
        date_part,
    )
    df[business_day_col] = pd.to_datetime(df[business_day_col])

    # Period mapping
    h = df[hour_col].astype(int)
    df[period_col] = pd.cut(
        h, bins=[0, 8, 16, 24],
        labels=["1_8", "9_16", "17_24"],
        include_lowest=True,
    ).astype(str)

    return df


def compute_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Compute business_day for a single timestamp.

    Args:
        ts: A pandas Timestamp.

    Returns:
        Business day as pd.Timestamp (normalized to midnight).
    """
    date_part = ts.normalize()
    if ts.hour == 0:
        return date_part - pd.Timedelta(days=1)
    return date_part


def compute_hour_business(ts: pd.Timestamp) -> int:
    """Compute hour_business (1-24) for a single timestamp.

    Args:
        ts: A pandas Timestamp.

    Returns:
        Business hour (1-24). Hour 0 maps to 24.
    """
    if ts.hour == 0:
        return 24
    return ts.hour


def compute_period(hour_business: int) -> str:
    """Compute period segment from business hour.

    Args:
        hour_business: Business hour (1-24).

    Returns:
        Period string: "1_8", "9_16", or "17_24".
    """
    if 1 <= hour_business <= 8:
        return "1_8"
    elif 9 <= hour_business <= 16:
        return "9_16"
    elif 17 <= hour_business <= 24:
        return "17_24"
    else:
        raise ValueError(f"hour_business must be 1-24, got {hour_business}")
