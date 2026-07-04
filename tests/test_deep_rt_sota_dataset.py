"""Tests for DeepRT-SOTA v2 dataset module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.deep_rt_sota_dataset import (
    DeepRTSOTADatasetConfig,
    DeepRTSOTADataset,
    build_deep_rt_sota_dataset,
)
from models.deep_sgdf_delta.business_time import add_business_time_columns


def create_sample_data(n_days: int = 30) -> pd.DataFrame:
    """Create sample hourly data for testing.

    Args:
        n_days: Number of days of data to generate.

    Returns:
        DataFrame with sample data.
    """
    # Generate hourly timestamps
    start_date = pd.Timestamp("2026-01-01")
    end_date = start_date + pd.Timedelta(days=n_days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")[:-1]  # Exclude last

    # Generate sample data
    n_hours = len(timestamps)
    np.random.seed(42)

    # Generate rt_actual with some pattern
    rt_actual = np.random.randn(n_hours) * 50 + 300
    rt_actual = np.clip(rt_actual, -100, 800)  # Clip to reasonable range

    # Generate da_anchor (correlated with rt_actual)
    da_anchor = rt_actual + np.random.randn(n_hours) * 20

    df = pd.DataFrame({
        "ds": timestamps,
        "rt_actual": rt_actual,
        "da_anchor": da_anchor,
    })

    # Add business time columns
    df = add_business_time_columns(df, timestamp_col="ds")

    return df


class TestDeepRTSOTADatasetConfig:
    """Tests for DeepRTSOTADatasetConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeepRTSOTADatasetConfig()
        assert config.seq_len_days == 14
        assert config.target_mode == "direct"
        assert config.risk_features is False
        assert config.forecast_features is False
        assert config.mode == "FULL_DAY"

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeepRTSOTADatasetConfig(
            seq_len_days=7,
            target_mode="residual_to_da",
            risk_features=True,
            forecast_features=True,
            mode="FULL_DAY",
        )
        assert config.seq_len_days == 7
        assert config.target_mode == "residual_to_da"
        assert config.risk_features is True
        assert config.forecast_features is True

    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="Only FULL_DAY mode is currently supported"):
            DeepRTSOTADatasetConfig(mode="INTRADAY")


class TestDeepRTSOTADataset:
    """Tests for DeepRTSOTADataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return create_sample_data(n_days=30)

    @pytest.fixture
    def config(self):
        """Create default config for tests."""
        return DeepRTSOTADatasetConfig(
            seq_len_days=7,
            target_mode="direct",
            risk_features=False,
            forecast_features=False,
        )

    def test_init(self, sample_data, config):
        """Test dataset initialization."""
        dataset = DeepRTSOTADataset(
            config=config,
            data_df=sample_data,
            split="train",
        )
        assert dataset is not None
        assert dataset.config == config
        assert dataset.split == "train"

    def test_missing_required_column(self):
        """Test missing required column raises error."""
        config = DeepRTSOTADatasetConfig()
        df = pd.DataFrame({"rt_actual": [1, 2, 3]})  # Missing "ds"

        with pytest.raises(ValueError, match="Missing required column: ds"):
            DeepRTSOTADataset(config=config, data_df=df, split="train")

    def test_preprocess_data(self, sample_data, config):
        """Test data preprocessing."""
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")

        # Check business time columns added
        assert "business_day" in dataset.data_df.columns
        assert "hour_business" in dataset.data_df.columns
        assert "period" in dataset.data_df.columns

        # Check business time rule: hour 0 -> business_day = D-1, hour_business = 24
        midnight_rows = dataset.data_df[dataset.data_df["ds"].dt.hour == 0]
        if len(midnight_rows) > 0:
            for _, row in midnight_rows.iterrows():
                expected_business_day = row["ds"].normalize() - pd.Timedelta(days=1)
                expected_hour = 24
                assert row["business_day"] == expected_business_day
                assert row["hour_business"] == expected_hour

    def test_build_samples(self, sample_data, config):
        """Test sample building."""
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")

        # Check samples built
        assert len(dataset.samples) > 0

        # Check sample structure
        sample = dataset.samples[0]
        assert "business_day" in sample
        assert "X_seq" in sample
        assert "X_static" in sample
        assert "y" in sample
        assert "feature_manifest" in sample

    def test_len(self, sample_data, config):
        """Test __len__ method."""
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")
        assert len(dataset) == len(dataset.samples)

    def test_getitem(self, sample_data, config):
        """Test __getitem__ method."""
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")

        if len(dataset) > 0:
            sample = dataset[0]
            assert "business_day" in sample
            assert "X_seq" in sample
            assert "X_static" in sample
            assert "y" in sample

            # Check tensor types
            import torch
            assert isinstance(sample["X_seq"], torch.Tensor)
            assert isinstance(sample["X_static"], torch.Tensor)
            assert isinstance(sample["y"], torch.Tensor)

    def test_target_mode_direct(self, sample_data):
        """Test direct target mode."""
        config = DeepRTSOTADatasetConfig(target_mode="direct")
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")

        if len(dataset) > 0:
            sample = dataset[0]
            y = sample["y"]

            # y should be rt_actual
            day = sample["business_day"]
            day_hours = sample_data[sample_data["business_day"] == day]
            expected_y = day_hours["rt_actual"].values[:24]

            np.testing.assert_allclose(y.numpy(), expected_y, rtol=1e-5)

    def test_target_mode_residual_to_da(self, sample_data):
        """Test residual_to_da target mode."""
        config = DeepRTSOTADatasetConfig(target_mode="residual_to_da")
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")

        if len(dataset) > 0:
            sample = dataset[0]
            y = sample["y"]

            # y should be rt_actual - da_anchor
            day = sample["business_day"]
            day_hours = sample_data[sample_data["business_day"] == day]
            rt_actual = day_hours["rt_actual"].values[:24]
            da_anchor = day_hours["da_anchor"].values[:24]
            expected_y = rt_actual - da_anchor

            np.testing.assert_allclose(y.numpy(), expected_y, rtol=1e-5)

    def test_missing_da_anchor_for_residual(self, sample_data):
        """Test missing da_anchor raises error for residual_to_da mode."""
        config = DeepRTSOTADatasetConfig(target_mode="residual_to_da")
        df = sample_data.drop(columns=["da_anchor"])

        with pytest.raises(ValueError, match="da_anchor column required for residual_to_da mode"):
            DeepRTSOTADataset(config=config, data_df=df, split="train")

    def test_leakage_check(self, sample_data, config):
        """Test leakage check: target day actual should not be used for features."""
        # TODO: Implement proper leakage check test
        # For now, just check that the dataset can be built
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")
        assert dataset is not None

    def test_save_feature_manifest(self, sample_data, config, tmp_path):
        """Test saving feature manifest."""
        dataset = DeepRTSOTADataset(config=config, data_df=sample_data, split="train")

        manifest_path = tmp_path / "feature_manifest.json"
        dataset.save_feature_manifest(str(manifest_path))

        assert manifest_path.exists()
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert isinstance(manifest, dict)


class TestBuildDeepRTSOTA:
    """Tests for build_deep_rt_sota_dataset function."""

    def test_build_dataset(self):
        """Test building dataset."""
        config = DeepRTSOTADatasetConfig()
        df = create_sample_data(n_days=30)

        dataset = build_deep_rt_sota_dataset(config=config, data_df=df, split="train")

        assert isinstance(dataset, DeepRTSOTADataset)
        assert len(dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
