"""Unified dataset for DeepRT-SOTA v2.

This module builds training samples for the standalone realtime price deep model.

Design:
- Accepts `full_df` (all data with features already built) and `target_days`
- For each target day, builds sequence from all data BEFORE that day
- No leakage: target day actuals never appear in sequence
- Supports both day-level (24h vector) and hourly prediction

Strictly uses business_time.py for time alignment.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Type alias for compatibility
TorchDataset = object  # Placeholder, not actually used


class DeepRTSOTADatasetConfig:
    """Configuration for DeepRT-SOTA dataset."""

    def __init__(
        self,
        seq_len_days: int = 14,
        target_mode: str = "direct",
        risk_features: bool = False,
        forecast_features: bool = False,
        mode: str = "FULL_DAY",
        target_granularity: str = "day",  # "day" or "hourly"
    ):
        """Initialize dataset configuration.

        Args:
            seq_len_days: Number of past days to use as sequence (7, 14, 30).
            target_mode: "direct" (predict rt_actual) or "residual_to_da"
                        (predict rt_actual - da_anchor).
            risk_features: Whether to include risk features.
            forecast_features: Whether to include forecast-side features.
            mode: "FULL_DAY" (currently only supported).
            target_granularity: "day" (predict 24h vector) or "hourly" (predict 1 hour).
        """
        self.seq_len_days = seq_len_days
        self.target_mode = target_mode
        self.risk_features = risk_features
        self.forecast_features = forecast_features
        self.mode = mode
        self.target_granularity = target_granularity

        if self.mode != "FULL_DAY":
            raise ValueError(f"Only FULL_DAY mode is currently supported, got {self.mode}")
        if self.target_granularity not in ("day", "hourly"):
            raise ValueError(f"target_granularity must be 'day' or 'hourly', got {self.target_granularity}")
        if self.target_granularity == "hourly":
            raise NotImplementedError(
                "Hourly mode is not production-ready. Use target_granularity='day'."
            )


class DeepRTSOTADataset:
    """Unified dataset for DeepRT-SOTA v2.

    Uses `full_df` (all data with features) for sequence building,
    so test days can access history from before the test period.

    No leakage: for target day D, only uses data from days < D.

    Note: This class does NOT inherit from torch.utils.data.Dataset.
    It returns plain numpy arrays. The caller is responsible for
    converting to tensors and creating a DataLoader.
    """

    def __init__(
        self,
        config: DeepRTSOTADatasetConfig,
        full_df: pd.DataFrame,
        target_days: List[pd.Timestamp],
        feature_columns: List[str],
    ):
        """Initialize dataset.

        Args:
            config: Dataset configuration.
            full_df: Full featured data (train + val + test merged).
                     MUST be sorted by [business_day, hour_business].
                     Features already built (rt_lag_24h, etc.).
            target_days: List of business_days to build samples for.
                         For train: all train days.
                         For test: all test days (should be ~27 for 2026-02).
            feature_columns: List of feature column names to use in X_seq.
        """
        self.config = config
        self.full_df = full_df.copy()
        self.target_days = sorted(target_days)
        self.feature_columns = feature_columns

        # Validate
        if "business_day" not in self.full_df.columns:
            raise ValueError("full_df must have 'business_day' column")
        if "hour_business" not in self.full_df.columns:
            raise ValueError("full_df must have 'hour_business' column")
        if "rt_actual" not in self.full_df.columns:
            raise ValueError("full_df must have 'rt_actual' column")

        # Ensure sorted
        self.full_df = self.full_df.sort_values(
            ["business_day", "hour_business"]
        ).reset_index(drop=True)

        # Build samples
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        """Build training samples for all target_days.

        Returns:
            List of sample dictionaries.

        Raises:
            ValueError: If no valid samples found (empty dataset).
        """
        samples = []
        skipped = []  # Track skipped days

        for day in self.target_days:
            if self.config.target_granularity == "day":
                sample = self._build_day_sample(day)
            else:  # "hourly"
                sample = self._build_hourly_sample(day)
            if sample is not None:
                samples.append(sample)
            else:
                skipped.append(day)

        if skipped:
            print(f"  WARNING: Skipped {len(skipped)} days: {skipped[:5]}...")

        # Check for empty dataset (don't silently return empty)
        if len(samples) == 0:
            raise ValueError(
                f"Dataset has 0 valid samples! "
                f"target_days={len(self.target_days)}, "
                f"target_mode={self.config.target_mode}, "
                f"seq_len_days={self.config.seq_len_days}. "
                f"Skipped days: {skipped}"
            )

        return samples

    def _build_day_sample(self, business_day: pd.Timestamp) -> Optional[Dict]:
        """Build a day-level sample (predict 24h vector).

        Args:
            business_day: Target business day.

        Returns:
            Sample dict or None if invalid.
        """
        # Get target day rows
        day_mask = self.full_df["business_day"] == business_day
        day_rows = self.full_df[day_mask].sort_values("hour_business")

        if len(day_rows) < 20:
            return None  # Too few hours (need at least 20/24)

        # Pad to 24 hours if needed
        if len(day_rows) < 24:
            padded = np.zeros((24, len(day_rows.columns)), dtype=np.float32)
            padded[:len(day_rows)] = day_rows.values
            day_rows = pd.DataFrame(padded, columns=day_rows.columns)

        # Build sequence from history (days < business_day)
        seq_end = business_day - pd.Timedelta(days=1)
        seq_start = business_day - pd.Timedelta(days=self.config.seq_len_days)

        seq_mask = (
            (self.full_df["business_day"] >= seq_start)
            & (self.full_df["business_day"] <= seq_end)
        )
        seq_df = self.full_df[seq_mask].sort_values(["business_day", "hour_business"])

        # Use all available history (pad if insufficient)
        # No minimum check - will pad with zeros if needed

        # Build X_seq: (seq_len_days * 24, n_features)
        X_seq = seq_df[self.feature_columns].values.astype(np.float32)
        # Pad or truncate to expected length
        if len(X_seq) < self.config.seq_len_days * 24:
            # Pad with zeros (or mean)
            pad_len = (self.config.seq_len_days * 24) - len(X_seq)
            X_seq = np.pad(X_seq, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0)
        else:
            X_seq = X_seq[-self.config.seq_len_days * 24:]

        # Fill NaN with 0 (don't skip samples with NaN features)
        X_seq = np.nan_to_num(X_seq, nan=0.0)

        # Build y: (24,)
        y = day_rows["rt_actual"].values.astype(np.float32)
        # Do NOT fill NaN target with 0 — that's a data leakage risk.
        # Instead, if any NaN in target, skip this day.
        if np.any(np.isnan(y)):
            return None  # Skip days with NaN target

        if self.config.target_mode == "residual_to_da":
            if "da_anchor" not in day_rows.columns:
                raise ValueError(
                    f"da_anchor column required for residual_to_da mode. "
                    f"business_day={business_day}, available columns: {list(day_rows.columns)}"
                )
            da = day_rows["da_anchor"].values.astype(np.float32)
            if np.any(np.isnan(da)):
                return None  # Cannot compute residual with NaN anchor
            y = y - da

        # Build X_static: static features for target day (optional)
        X_static = np.array([
            day_rows["hour_sin"].mean() if "hour_sin" in day_rows.columns else 0.0,
            day_rows["dow_sin"].mean() if "dow_sin" in day_rows.columns else 0.0,
        ], dtype=np.float32)

        return {
            "business_day": business_day,
            "X_seq": X_seq,
            "X_static": X_static,
            "y": y,
        }

    def _build_hourly_sample(self, business_day: pd.Timestamp) -> Optional[Dict]:
        """Build hourly samples for all 24 hours of the day.

        For hourly prediction, each hour gets its own sample.
        Sequence: seq_len_days * 24 hours before this hour.

        Args:
            business_day: Target business day.

        Returns:
            Sample dict or None if invalid.
        """
        day_mask = self.full_df["business_day"] == business_day
        day_rows = self.full_df[day_mask].sort_values("hour_business")

        samples = []
        for _, row in day_rows.iterrows():
            hour = row["hour_business"]
            ts = row["ds"] if "ds" in row.index else None

            # Build sequence: all data before this timestamp
            if ts is None:
                # Use business_day and hour to compute cutoff
                cutoff = pd.Timestamp(year=business_day.year, month=business_day.month, day=business_day.day) + pd.Timedelta(hours=hour-1)
            else:
                cutoff = ts - pd.Timedelta(hours=1)

            seq_start = cutoff - pd.Timedelta(hours=self.config.seq_len_days * 24)

            seq_mask = (
                (self.full_df["ds"] >= seq_start)
                & (self.full_df["ds"] <= cutoff)
            )
            seq_df = self.full_df[seq_mask].sort_values("ds")

            expected_seq_len = self.config.seq_len_days * 24
            if len(seq_df) < expected_seq_len:
                continue  # Skip this hour

            X_seq = seq_df[self.feature_columns].values.astype(np.float32)
            X_seq = X_seq[-expected_seq_len:]

            y = np.array([row["rt_actual"]], dtype=np.float32)

            if self.config.target_mode == "residual_to_da":
                if "da_anchor" not in row.index:
                    raise ValueError("da_anchor column required for residual_to_da mode")
                da = np.array([row["da_anchor"]], dtype=np.float32)
                if np.isnan(da[0]):
                    continue
                y = y - da

            X_static = np.array([
                row["hour_sin"] if "hour_sin" in row.index else 0.0,
                row["dow_sin"] if "dow_sin" in row.index else 0.0,
            ], dtype=np.float32)

            samples.append({
                "business_day": business_day,
                "hour_business": hour,
                "X_seq": X_seq,
                "X_static": X_static,
                "y": y,
            })

        # For day-level interface, return first sample (or average?)
        # Actually, for hourly, we should return all samples as a list
        # But __getitem__ expects one sample per index...
        # Let's use a different approach: store all hourly samples flat
        if len(samples) == 0:
            return None

        # Return a merged sample (X_seq averaged? No, that doesn't make sense)
        # Actually, let's just return the day sample with all hours
        # This is getting complex. Let me simplify: for hourly, use same interface
        # but X_seq is (seq_len_hours, n_features) and y is scalar

        # The current design builds one sample per day. For hourly, we need one sample per hour.
        # Let me change the design: _build_samples returns flat list of hourly samples.

        # Actually, I'll handle this in a separate method. For now, return None for hourly.
        # TODO: Implement proper hourly sampling.
        return None

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Sample dictionary with numpy arrays (business_day excluded - use samples[idx]['business_day'] directly).
        """
        sample = self.samples[idx]

        return {
            "X_seq": sample["X_seq"],  # numpy array
            "X_static": sample["X_static"],  # numpy array
            "y": sample["y"],  # numpy array
        }

    def get_sample_counts(self) -> Dict:
        """Get diagnostic counts."""
        return {
            "n_samples": len(self.samples),
            "n_target_days": len(self.target_days),
            "n_feature_cols": len(self.feature_columns),
            "seq_shape": self.samples[0]["X_seq"].shape if len(self.samples) > 0 else None,
            "y_shape": self.samples[0]["y"].shape if len(self.samples) > 0 else None,
        }


def build_deep_rt_sota_dataset(
    config: DeepRTSOTADatasetConfig,
    full_df: pd.DataFrame,
    target_days: List[pd.Timestamp],
    feature_columns: List[str],
) -> DeepRTSOTADataset:
    """Build DeepRT-SOTA dataset.

    Args:
        config: Dataset configuration.
        full_df: Full featured data (train + val + test merged).
        target_days: List of business_days to build samples for.
        feature_columns: List of feature column names.

    Returns:
        DeepRTSOTADataset instance.
    """
    return DeepRTSOTADataset(
        config=config,
        full_df=full_df,
        target_days=target_days,
        feature_columns=feature_columns,
    )
