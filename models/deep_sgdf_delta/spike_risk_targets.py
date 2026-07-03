"""Spike risk target definitions.

Computes spike labels from (da_anchor, rt_actual):
  - spike_label: rt_actual >= 500
  - extreme_spike_label: rt_actual >= 800
  - relative_spike_label: rt_actual - da_anchor >= 200

Business time alignment uses business_time.py (single source of truth).

Output columns:
  business_day, hour_business, ds, period,
  rt_actual, da_anchor,
  spike_label, extreme_spike_label, relative_spike_label
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from models.deep_sgdf_delta.business_time import add_business_time_columns


@dataclass
class SpikeRiskThresholds:
    """Configurable thresholds for spike labels."""
    spike: float = 500.0
    extreme_spike: float = 800.0
    relative_spike: float = 200.0


@dataclass
class SpikeRiskTargetResult:
    """Result of computing spike risk targets."""
    df: pd.DataFrame
    n_rows: int = 0
    n_valid: int = 0
    n_missing_da: int = 0
    n_missing_rt: int = 0
    thresholds: SpikeRiskThresholds = field(default_factory=SpikeRiskThresholds)

    # Label statistics
    spike_rate: float = 0.0
    extreme_spike_rate: float = 0.0
    relative_spike_rate: float = 0.0
    mean_rt: float = 0.0
    std_rt: float = 0.0


REQUIRED_COLUMNS = {"ds", "da_anchor", "rt_actual"}

OUTPUT_COLUMNS = [
    "business_day", "hour_business", "ds", "period",
    "rt_actual", "da_anchor",
    "spike_label", "extreme_spike_label", "relative_spike_label",
]


def compute_spike_risk_targets(
    df: pd.DataFrame,
    thresholds: Optional[SpikeRiskThresholds] = None,
    timestamp_col: str = "ds",
) -> SpikeRiskTargetResult:
    """Compute spike risk targets from a DataFrame with da_anchor and rt_actual.

    Args:
        df: DataFrame with at least ds, da_anchor, rt_actual columns.
        thresholds: Spike thresholds. Uses defaults if None.
        timestamp_col: Name of the timestamp column.

    Returns:
        SpikeRiskTargetResult with labeled DataFrame and statistics.

    Raises:
        ValueError: If required columns are missing.
    """
    if thresholds is None:
        thresholds = SpikeRiskThresholds()

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

    # Spike labels
    work["spike_label"] = (
        work["rt_actual"] >= thresholds.spike
    ).astype(int)
    work["extreme_spike_label"] = (
        work["rt_actual"] >= thresholds.extreme_spike
    ).astype(int)
    work["relative_spike_label"] = (
        (work["rt_actual"] - work["da_anchor"]) >= thresholds.relative_spike
    ).astype(int)

    # Set NaN rows labels to -1 (unknown)
    invalid_mask = work["da_anchor"].isna() | work["rt_actual"].isna()
    for label_col in [
        "spike_label", "extreme_spike_label", "relative_spike_label",
    ]:
        work.loc[invalid_mask, label_col] = -1

    # Statistics on valid rows
    valid = work.loc[~invalid_mask]
    n_valid = len(valid)
    n_rows = len(work)

    if n_valid > 0:
        spike_rate = float(valid["spike_label"].mean())
        extreme_spike_rate = float(valid["extreme_spike_label"].mean())
        relative_spike_rate = float(valid["relative_spike_label"].mean())
        mean_rt = float(valid["rt_actual"].mean())
        std_rt = float(valid["rt_actual"].std())
    else:
        spike_rate = extreme_spike_rate = relative_spike_rate = 0.0
        mean_rt = std_rt = 0.0

    return SpikeRiskTargetResult(
        df=work,
        n_rows=n_rows,
        n_valid=n_valid,
        n_missing_da=n_missing_da,
        n_missing_rt=n_missing_rt,
        thresholds=thresholds,
        spike_rate=spike_rate,
        extreme_spike_rate=extreme_spike_rate,
        relative_spike_rate=relative_spike_rate,
        mean_rt=mean_rt,
        std_rt=std_rt,
    )
