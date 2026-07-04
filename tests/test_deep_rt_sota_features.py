"""Tests for DeepRT-SOTA v2 features module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.deep_rt_sota_features import (
    build_deep_rt_sota_features,
    get_feature_columns,
    PRICE_HISTORY_FEATURES,
    ANCHOR_FORECAST_FEATURES,
    CALENDAR_FEATURES,
    RISK_FEATURES,
    FORECAST_SIDE_FEATURES,
)
from models.deep_sgdf_delta.business_time import add_business_time_columns


def create_sample_data(n_days: int = 30) -> pd.DataFrame:
    """Create sample hourly data for testing."""
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

    return df


class TestBuildDeepRTSOTAFeatures:
    """Tests for build_deep_rt_sota_features function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return create_sample_data(n_days=30)

    def test_build_features_basic(self, sample_data):
        """Test basic feature building."""
        df, manifest = build_deep_rt_sota_features(
            sample_data,
            risk_features=False,
            forecast_features=False,
        )

        # Check price history features
        for col in PRICE_HISTORY_FEATURES:
            assert col in df.columns, f"Missing price history feature: {col}"

        # Check anchor/forecast features
        for col in ANCHOR_FORECAST_FEATURES:
            assert col in df.columns, f"Missing anchor feature: {col}"

        # Check calendar features
        for col in CALENDAR_FEATURES:
            assert col in df.columns, f"Missing calendar feature: {col}"

        # Check manifest
        assert isinstance(manifest, dict)
        assert "price_history_features" in manifest
        assert "anchor_forecast_features" in manifest
        assert "calendar_features" in manifest

    def test_build_features_with_risk(self, sample_data):
        """Test feature building with risk features."""
        # Create dummy risk df
        risk_df = sample_data[["business_day", "hour_business"]].copy()
        for col in RISK_FEATURES:
            risk_df[col] = np.random.rand(len(risk_df))

        df, manifest = build_deep_rt_sota_features(
            sample_data,
            risk_features=True,
            forecast_features=False,
            risk_df=risk_df,
        )

        # Check risk features
        for col in RISK_FEATURES:
            assert col in df.columns, f"Missing risk feature: {col}"

        assert len(manifest["risk_features"]) > 0

    def test_build_features_with_forecast(self, sample_data):
        """Test feature building with forecast-side features."""
        df, manifest = build_deep_rt_sota_features(
            sample_data,
            risk_features=False,
            forecast_features=True,
        )

        # Check forecast-side features (may be NaN if not in input)
        for col in FORECAST_SIDE_FEATURES:
            assert col in df.columns, f"Missing forecast feature: {col}"

        assert len(manifest["forecast_side_features"]) > 0

    def test_price_history_features(self, sample_data):
        """Test price history feature values."""
        df, _ = build_deep_rt_sota_features(sample_data)

        # Check lag features
        # rt_lag_24h at row 24 should equal rt_actual at row 0
        assert np.isclose(df["rt_lag_24h"].iloc[24], sample_data["rt_actual"].iloc[0])

        # rt_lag_48h at row 48 should equal rt_actual at row 0
        assert np.isclose(df["rt_lag_48h"].iloc[48], sample_data["rt_actual"].iloc[0])

    def test_calendar_features(self, sample_data):
        """Test calendar feature values."""
        df, _ = build_deep_rt_sota_features(sample_data)

        # Check hour_sin, hour_cos
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns

        # Check is_weekend
        assert set(df["is_weekend"].unique()).issubset({0, 1})

        # Check period_id
        assert set(df["period_id"].unique()).issubset({0, 1, 2})

    def test_missing_da_anchor(self):
        """Test handling missing da_anchor."""
        df = create_sample_data(n_days=10)
        df = df.drop(columns=["da_anchor"])

        df, manifest = build_deep_rt_sota_features(
            df,
            da_anchor_col="da_anchor",
        )

        # Should still have da_anchor column (filled with NaN)
        assert "da_anchor" in df.columns

    def test_business_time_columns_added(self, sample_data):
        """Test that business time columns are added if missing."""
        df_no_bt = sample_data.drop(columns=["business_day", "hour_business", "period"])

        df, _ = build_deep_rt_sota_features(df_no_bt)

        # Should have business time columns now
        assert "business_day" in df.columns
        assert "hour_business" in df.columns
        assert "period" in df.columns


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""

    def test_no_optional_features(self):
        """Test without optional features."""
        features = get_feature_columns(
            risk_features=False,
            forecast_features=False,
        )

        # Should have basic features
        assert len(features) > 0

        # Should not have risk features
        for col in RISK_FEATURES:
            assert col not in features

        # Should not have forecast features
        for col in FORECAST_SIDE_FEATURES:
            assert col not in features

    def test_with_risk_features(self):
        """Test with risk features."""
        features = get_feature_columns(
            risk_features=True,
            forecast_features=False,
        )

        # Should have risk features
        for col in RISK_FEATURES:
            assert col in features

    def test_with_forecast_features(self):
        """Test with forecast features."""
        features = get_feature_columns(
            risk_features=False,
            forecast_features=True,
        )

        # Should have forecast features
        for col in FORECAST_SIDE_FEATURES:
            assert col in features

    def test_with_all_features(self):
        """Test with all features."""
        features = get_feature_columns(
            risk_features=True,
            forecast_features=True,
        )

        # Should have all features
        assert len(features) > 0
        assert len(features) == (
            len(PRICE_HISTORY_FEATURES)
            + len(ANCHOR_FORECAST_FEATURES)
            + len(CALENDAR_FEATURES)
            + len(RISK_FEATURES)
            + len(FORECAST_SIDE_FEATURES)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
