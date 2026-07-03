"""Negative risk target definitions.

Computes negative price labels from (da_anchor, rt_actual):
  - negative_label: rt_actual < 0
  - deep_negative_label: rt_actual <= -100
  - relative_down_label: rt_actual - da_anchor <= -200

Business time alignment uses business_time.py (single source of truth).

Output columns:
  business_day, hour_business, ds, period,
  rt_actual, da_anchor,
  negative_label, deep_negative_label, relative_down_label
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from models.deep_sgdf_delta.business_time import add_business_time_columns


@dataclass
class NegativeRiskThresholds:
    """Configurable thresholds for negative price labels."""
    negative: float = 0.0
    deep_negative: float = -100.0
    relative_down: float = -200.0


@dataclass
class NegativeRiskTargetResult:
    """Result of computing negative risk targets."""
    df: pd.DataFrame
    n_rows: int = 0
    n_valid: int = 0
    n_missing_da: int = 0
    n_missing_rt: int = 0
    thresholds: NegativeRiskThresholds = field(default_factory=NegativeRiskThresholds)

    # Label statistics
    negative_rate: float = 0.0
    deep_negative_rate: float = 0.0
    relative_down_rate: float = 0.0
    mean_rt: float = 0.0
    std_rt: float = 0.0


REQUIRED_COLUMNS = {"ds", "da_anchor", "rt_actual"}

OUTPUT_COLUMNS = [
    "business_day", "hour_business", "ds", "period",
    "rt_actual", "da_anchor",
    "negative_label", "deep_negative_label", "relative_down_label",
]


def compute_negative_risk_targets(
    df: pd.DataFrame,
    thresholds: Optional[NegativeRiskThresholds] = None,
    timestamp_col: str = "ds",
) -> NegativeRiskTargetResult:
    """Compute negative risk targets from a DataFrame with da_anchor and rt_actual.

    Args:
        df: DataFrame with at least ds, da_anchor, rt_actual columns.
        thresholds: Negative price thresholds. Uses defaults if None.
        timestamp_col: Name of the timestamp column.

    Returns:
        NegativeRiskTargetResult with labeled DataFrame and statistics.

    Raises:
        ValueError: If required columns are missing.
    """
    if thresholds is None:
        thresholds = NegativeRiskThresholds()

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()

    # Ensure numeric
    work["da_anchor"] = pd.to_numeric(work["da_anchor"], errors="coerce")
    work["rt_actual"] = pd.to_numeric(work["rt_actual"], errors="coerce")

    # Add business time columns
    work = add_business_time_columns(work, timestamp_col=timestamp_col)

    # Count missing
    n_missing_da = int(work["da_anchor"].isna().sum())
    n_missing_rt = int(work["rt_actual"].isna().sum())

    # Negative labels
    work["negative_label"] = (
        work["rt_actual"] < thresholds.negative
    ).astype(int)
    work["deep_negative_label"] = (
        work["rt_actual"] <= thresholds.deep_negative
    ).astype(int)
    work["relative_down_label"] = (
        (work["rt_actual"] - work["da_anchor"]) <= thresholds.relative_down
    ).astype(int)

    # Set NaN rows labels to -1 (unknown)
    invalid_mask = work["da_anchor"].isna() | work["rt_actual"].isna()
    for label_col in [
        "negative_label", "deep_negative_label", "relative_down_label",
    ]:
        work.loc[invalid_mask, label_col] = -1

    # Statistics on valid rows
    valid = work.loc[~invalid_mask]
    n_valid = len(valid)
    n_rows = len(work)

    if n_valid > 0:
        negative_rate = float(valid["negative_label"].mean())
        deep_negative_rate = float(valid["deep_negative_label"].mean())
        relative_down_rate = float(valid["relative_down_label"].mean())
        mean_rt = float(valid["rt_actual"].mean())
        std_rt = float(valid["rt_actual"].std())
    else:
        negative_rate = deep_negative_rate = relative_down_rate = 0.0
        mean_rt = std_rt = 0.0

    return NegativeRiskTargetResult(
        df=work,
        n_rows=n_rows,
        n_valid=n_valid,
        n_missing_da=n_missing_da,
        n_missing_rt=n_missing_rt,
        thresholds=thresholds,
        negative_rate=negative_rate,
        deep_negative_rate=deep_negative_rate,
        relative_down_rate=relative_down_rate,
        mean_rt=mean_rt,
        std_rt=std_rt,
    )
