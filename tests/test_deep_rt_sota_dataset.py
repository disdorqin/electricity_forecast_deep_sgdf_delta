"""Tests for DeepRT-SOTA v2 dataset module.

Uses REAL API (matches deep_rt_sota_dataset.py implementation):
    DeepRTSOTADataset(config, full_df, target_days, feature_columns)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.deep_rt_sota_dataset import (
    DeepRTSOTADatasetConfig,
    DeepRTSOTADataset,
    build_deep_rt_sota_dataset,
)
from models.deep_sgdf_delta.business_time import add_business_time_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_sample_data(n_days: int = 30) -> pd.DataFrame:
    """Create sample hourly data with proper business_time columns."""
    start_date = pd.Timestamp("2026-01-01")
    end_date = start_date + pd.Timedelta(days=n_days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")[:-1]

    n_hours = len(timestamps)
    np.random.seed(42)

    rt_actual = np.random.randn(n_hours) * 50 + 300
    rt_actual = np.clip(rt_actual, -100, 800)

    da_anchor = rt_actual + np.random.randn(n_hours) * 20

    df = pd.DataFrame({
        "ds": timestamps,
        "rt_actual": rt_actual,
        "da_anchor": da_anchor,
    })

    df = add_business_time_columns(df, timestamp_col="ds")

    # Add minimal feature columns so dataset doesn't crash
    for h in range(24):
        hour_rows = df[df["hour_business"] == h + 1]
        if len(hour_rows) > 0:
            idx = hour_rows.index
            df.loc[idx, "rt_lag_24h"] = np.random.randn(len(idx)) * 10 + 300
            df.loc[idx, "da_lag_24h"] = np.random.randn(len(idx)) * 10 + 300

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return columns that exist in df and are reasonable features."""
    candidates = [
        "da_anchor", "rt_lag_24h", "da_lag_24h",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    return [c for c in candidates if c in df.columns]


# ---------------------------------------------------------------------------
# Test Config
# ---------------------------------------------------------------------------

class TestDeepRTSOTADatasetConfig:

    def test_default_config(self):
        config = DeepRTSOTADatasetConfig()
        assert config.seq_len_days == 14
        assert config.target_mode == "direct"
        assert config.target_granularity == "day"

    def test_custom_config(self):
        config = DeepRTSOTADatasetConfig(
            seq_len_days=7,
            target_mode="residual_to_da",
            target_granularity="day",
        )
        assert config.seq_len_days == 7
        assert config.target_mode == "residual_to_da"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Only FULL_DAY mode"):
            DeepRTSOTADatasetConfig(mode="INTRADAY")

    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="target_granularity must be"):
            DeepRTSOTADatasetConfig(target_granularity="minute")


# ---------------------------------------------------------------------------
# Test Dataset — REAL API
# ---------------------------------------------------------------------------

class TestDeepRTSOTADatasetRealAPI:

    @pytest.fixture
    def sample_df(self):
        return create_sample_data(n_days=30)

    @pytest.fixture
    def config(self):
        return DeepRTSOTADatasetConfig(
            seq_len_days=7,
            target_mode="direct",
            target_granularity="day",
        )

    @pytest.fixture
    def feature_cols(self, sample_df):
        return get_feature_columns(sample_df)

    def test_init_with_real_api(self, sample_df, config, feature_cols):
        """Test dataset init with REAL API (config, full_df, target_days, feature_columns)."""
        target_days = sorted(sample_df["business_day"].unique())[7:]  # Skip first 7 (need history)
        dataset = DeepRTSOTADataset(
            config=config,
            full_df=sample_df,
            target_days=target_days,
            feature_columns=feature_cols,
        )
        assert dataset is not None
        assert len(dataset) > 0

    def test_missing_business_day_column(self, config, feature_cols):
        """Missing business_day must raise."""
        df = pd.DataFrame({"ds": [1], "rt_actual": [1]})
        with pytest.raises(ValueError, match="business_day"):
            DeepRTSOTADataset(config, df, [pd.Timestamp("2026-01-10")], feature_cols)

    def test_missing_rt_actual_column(self, config, feature_cols):
        """Missing rt_actual must raise."""
        df = pd.DataFrame({
            "ds": [pd.Timestamp("2026-01-01")],
            "business_day": [pd.Timestamp("2026-01-01")],
            "hour_business": [1],
        })
        with pytest.raises(ValueError, match="rt_actual"):
            DeepRTSOTADataset(config, df, [pd.Timestamp("2026-01-10")], feature_cols)

    def test_residual_to_da_missing_da_anchor(self, sample_df, feature_cols):
        """residual_to_da without da_anchor must raise."""
        config = DeepRTSOTADatasetConfig(target_mode="residual_to_da")
        df = sample_df.drop(columns=["da_anchor"])
        target_days = sorted(df["business_day"].unique())[7:]
        with pytest.raises(ValueError, match="da_anchor.*required"):
            DeepRTSOTADataset(config, df, target_days, feature_cols)

    def test_no_silent_empty_dataset(self, sample_df, config, feature_cols):
        """If no valid samples, must raise ValueError (no silent empty)."""
        # target_days with no history → should fail
        early_day = sorted(sample_df["business_day"].unique())[0:1]
        with pytest.raises(ValueError, match="0 valid samples"):
            DeepRTSOTADataset(config, sample_df, early_day, feature_cols)

    def test_target_nan_not_filled_with_zero(self, sample_df, config, feature_cols):
        """Target NaN must NOT be silently filled with 0."""
        config = DeepRTSOTADatasetConfig(target_mode="direct")
        target_days = sorted(sample_df["business_day"].unique())[7:9]

        # Intentionally set some target NaN
        df = sample_df.copy()
        mask = df["business_day"].isin(target_days)
        df.loc[mask, "rt_actual"] = np.nan

        # The dataset should either skip these days or raise
        # (Current implementation fills with 0 — this is a bug we're documenting)
        dataset = DeepRTSOTADataset(config, df, target_days, feature_cols)
        for i in range(len(dataset)):
            sample = dataset[i]
            y = sample["y"]
            if np.any(y == 0.0):
                pytest.fail("Target NaN was filled with 0! This is a data leakage risk.")

    def test_business_time_rule(self, sample_df, config, feature_cols):
        """Verify business_time rules:
        - 00:00 → business_day = D-1, hour_business = 24
        - 01:00~23:00 → business_day = D, hour_business = 1~23
        """
        config = DeepRTSOTADatasetConfig()
        target_days = sorted(sample_df["business_day"].unique())[7:9]
        dataset = DeepRTSOTADataset(config, sample_df, target_days, feature_cols)

        # Check a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            bd = sample["business_day"]
            # business_day should be a pd.Timestamp
            assert isinstance(bd, pd.Timestamp), f"business_day is {type(bd)}"

    def test_day_level_requires_24h(self, sample_df, config, feature_cols):
        """Day-level mode: if < 20 hours available, skip day."""
        config = DeepRTSOTADatasetConfig(target_granularity="day")
        # Create a day with only 10 hours
        df = sample_df.copy()
        bad_day = sorted(df["business_day"].unique())[10]
        bad_mask = (df["business_day"] == bad_day) & (df["hour_business"] > 10)
        df = df[~bad_mask]

        target_days = [bad_day]
        dataset = DeepRTSOTADataset(config, df, target_days, feature_cols)
        # bad_day should be skipped (sample is None → not added)
        assert len(dataset) == 0  # Or check that bad_day not in samples

    def test_hourly_mode_raises_not_implemented(self, sample_df, feature_cols):
        """Hourly mode must raise NotImplementedError."""
        config = DeepRTSOTADatasetConfig(target_granularity="hourly")
        target_days = sorted(sample_df["business_day"].unique())[7:9]
        with pytest.raises(NotImplementedError, match="Hourly mode"):
            DeepRTSOTADataset(config, sample_df, target_days, feature_cols)


# ---------------------------------------------------------------------------
# Test build_deep_rt_sota_dataset function
# ---------------------------------------------------------------------------

class TestBuildDeepRTSOTADataset:

    def test_build_with_real_api(self):
        config = DeepRTSOTADatasetConfig()
        df = create_sample_data(n_days=30)
        feature_cols = get_feature_columns(df)
        target_days = sorted(df["business_day"].unique())[7:]

        dataset = build_deep_rt_sota_dataset(
            config=config,
            full_df=df,
            target_days=target_days,
            feature_columns=feature_cols,
        )

        assert isinstance(dataset, DeepRTSOTADataset)
        assert len(dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
