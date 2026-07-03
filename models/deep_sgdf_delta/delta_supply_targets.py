"""DeltaSupply target definitions.

Computes deviation targets from (da_anchor, rt_actual):
  - price_delta = rt_actual - da_anchor
  - upward_deviation_label: price_delta >= upward_threshold
  - downward_deviation_label: price_delta <= downward_threshold
  - large_abs_deviation_label: |price_delta| >= abs_large_threshold
  - deviation_magnitude_target: clipped(price_delta, -clip, +clip)

Business time alignment uses business_time.py (single source of truth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from models.deep_sgdf_delta.business_time import add_business_time_columns


@dataclass
class DeltaSupplyThresholds:
    """Configurable thresholds for deviation labels."""
    upward: float = 100.0
    downward: float = -100.0
    abs_large: float = 150.0
    clip: float = 500.0


@dataclass
class DeltaSupplyTargetResult:
    """Result of computing delta supply targets."""
    df: pd.DataFrame
    n_rows: int = 0
    n_valid: int = 0
    n_missing_da: int = 0
    n_missing_rt: int = 0
    thresholds: DeltaSupplyThresholds = field(default_factory=DeltaSupplyThresholds)

    # Label statistics
    upward_rate: float = 0.0
    downward_rate: float = 0.0
    large_abs_rate: float = 0.0
    mean_delta: float = 0.0
    std_delta: float = 0.0


REQUIRED_COLUMNS = {"ds", "da_anchor", "rt_actual"}

OUTPUT_COLUMNS = [
    "business_day", "hour_business", "ds", "period",
    "rt_actual", "da_anchor", "price_delta",
    "upward_deviation_label", "downward_deviation_label",
    "large_abs_deviation_label", "deviation_magnitude_target",
]


def compute_delta_supply_targets(
    df: pd.DataFrame,
    thresholds: Optional[DeltaSupplyThresholds] = None,
    timestamp_col: str = "ds",
) -> DeltaSupplyTargetResult:
    """Compute deviation targets from a DataFrame with da_anchor and rt_actual.

    Args:
        df: DataFrame with at least ds, da_anchor, rt_actual columns.
        thresholds: Deviation thresholds. Uses defaults if None.
        timestamp_col: Name of the timestamp column.

    Returns:
        DeltaSupplyTargetResult with labeled DataFrame and statistics.

    Raises:
        ValueError: If required columns are missing.
    """
    if thresholds is None:
        thresholds = DeltaSupplyThresholds()

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

    # Compute price_delta
    work["price_delta"] = work["rt_actual"] - work["da_anchor"]

    # Deviation labels
    work["upward_deviation_label"] = (
        work["price_delta"] >= thresholds.upward
    ).astype(int)
    work["downward_deviation_label"] = (
        work["price_delta"] <= thresholds.downward
    ).astype(int)
    work["large_abs_deviation_label"] = (
        work["price_delta"].abs() >= thresholds.abs_large
    ).astype(int)

    # Magnitude target (clipped)
    work["deviation_magnitude_target"] = work["price_delta"].clip(
        lower=-thresholds.clip, upper=thresholds.clip
    )

    # Set NaN rows labels to -1 (unknown)
    invalid_mask = work["da_anchor"].isna() | work["rt_actual"].isna()
    for label_col in [
        "upward_deviation_label", "downward_deviation_label",
        "large_abs_deviation_label",
    ]:
        work.loc[invalid_mask, label_col] = -1
    work.loc[invalid_mask, "deviation_magnitude_target"] = np.nan

    # Statistics on valid rows
    valid = work.loc[~invalid_mask]
    n_valid = len(valid)
    n_rows = len(work)

    if n_valid > 0:
        upward_rate = float(valid["upward_deviation_label"].mean())
        downward_rate = float(valid["downward_deviation_label"].mean())
        large_abs_rate = float(valid["large_abs_deviation_label"].mean())
        mean_delta = float(valid["price_delta"].mean())
        std_delta = float(valid["price_delta"].std())
    else:
        upward_rate = downward_rate = large_abs_rate = 0.0
        mean_delta = std_delta = 0.0

    return DeltaSupplyTargetResult(
        df=work,
        n_rows=n_rows,
        n_valid=n_valid,
        n_missing_da=n_missing_da,
        n_missing_rt=n_missing_rt,
        thresholds=thresholds,
        upward_rate=upward_rate,
        downward_rate=downward_rate,
        large_abs_rate=large_abs_rate,
        mean_delta=mean_delta,
        std_delta=std_delta,
    )
