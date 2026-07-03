"""Tests for spike_risk_targets module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.spike_risk_targets import (
    SpikeRiskThresholds,
    compute_spike_risk_targets,
    OUTPUT_COLUMNS,
)


def _make_df(n=48, da=300.0, rt=350.0, start="2026-02-01"):
    """Create a simple test DataFrame."""
    dates = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "ds": dates,
        "da_anchor": da,
        "rt_actual": rt,
    })


class TestBusinessTimeMapping:
    """Test 00:00 -> previous business_day, hour 24."""

    def test_midnight_maps_to_previous_day_hour_24(self):
        df = pd.DataFrame({
            "ds": [pd.Timestamp("2026-02-02 00:00:00")],
            "da_anchor": [300.0],
            "rt_actual": [350.0],
        })
        result = compute_spike_risk_targets(df)
        row = result.df.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-02-01")
        assert row["hour_business"] == 24

    def test_hour_1_maps_to_same_day_hour_1(self):
        df = pd.DataFrame({
            "ds": [pd.Timestamp("2026-02-02 01:00:00")],
            "da_anchor": [300.0],
            "rt_actual": [350.0],
        })
        result = compute_spike_risk_targets(df)
        row = result.df.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-02-02")
        assert row["hour_business"] == 1


class TestSpikeLabel:
    def test_spike_when_rt_above_500(self):
        df = _make_df(n=24, da=300.0, rt=500.0)  # rt = 500 >= 500
        result = compute_spike_risk_targets(df)
        assert (result.df["spike_label"] == 1).all()

    def test_spike_when_rt_above_600(self):
        df = _make_df(n=24, da=300.0, rt=600.0)  # rt = 600 >= 500
        result = compute_spike_risk_targets(df)
        assert (result.df["spike_label"] == 1).all()

    def test_no_spike_when_rt_below_500(self):
        df = _make_df(n=24, da=300.0, rt=499.0)  # rt = 499 < 500
        result = compute_spike_risk_targets(df)
        assert (result.df["spike_label"] == 0).all()


class TestExtremeSpikeLabel:
    def test_extreme_spike_when_rt_above_800(self):
        df = _make_df(n=24, da=300.0, rt=800.0)  # rt = 800 >= 800
        result = compute_spike_risk_targets(df)
        assert (result.df["extreme_spike_label"] == 1).all()

    def test_extreme_spike_when_rt_above_1000(self):
        df = _make_df(n=24, da=300.0, rt=1000.0)  # rt = 1000 >= 800
        result = compute_spike_risk_targets(df)
        assert (result.df["extreme_spike_label"] == 1).all()

    def test_no_extreme_spike_when_rt_below_800(self):
        df = _make_df(n=24, da=300.0, rt=799.0)  # rt = 799 < 800
        result = compute_spike_risk_targets(df)
        assert (result.df["extreme_spike_label"] == 0).all()


class TestRelativeSpikeLabel:
    def test_relative_spike_when_delta_above_200(self):
        df = _make_df(n=24, da=300.0, rt=500.0)  # delta = 200 >= 200
        result = compute_spike_risk_targets(df)
        assert (result.df["relative_spike_label"] == 1).all()

    def test_relative_spike_when_delta_above_300(self):
        df = _make_df(n=24, da=300.0, rt=700.0)  # delta = 400 >= 200
        result = compute_spike_risk_targets(df)
        assert (result.df["relative_spike_label"] == 1).all()

    def test_no_relative_spike_when_delta_below_200(self):
        df = _make_df(n=24, da=300.0, rt=499.0)  # delta = 199 < 200
        result = compute_spike_risk_targets(df)
        assert (result.df["relative_spike_label"] == 0).all()


class TestMissingData:
    def test_missing_da_anchor_raises(self):
        df = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=24, freq="h"),
                           "rt_actual": 300.0})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_spike_risk_targets(df)

    def test_missing_rt_actual_raises(self):
        df = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=24, freq="h"),
                           "da_anchor": 300.0})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_spike_risk_targets(df)

    def test_missing_ds_raises(self):
        df = pd.DataFrame({"da_anchor": [300.0], "rt_actual": [500.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_spike_risk_targets(df)

    def test_nan_values_produce_minus1_labels(self):
        df = _make_df(n=24, da=300.0, rt=500.0)
        df.loc[0, "da_anchor"] = np.nan
        result = compute_spike_risk_targets(df)
        assert result.df.iloc[0]["spike_label"] == -1
        assert result.df.iloc[0]["extreme_spike_label"] == -1
        assert result.df.iloc[0]["relative_spike_label"] == -1


class TestStatistics:
    def test_statistics_computed(self):
        df = _make_df(n=48, da=300.0, rt=500.0)
        result = compute_spike_risk_targets(df)
        assert result.n_valid == 48
        assert result.spike_rate == 1.0  # rt=500 >= 500
        assert result.mean_rt == 500.0

    def test_extreme_spike_rate(self):
        df = _make_df(n=48, da=300.0, rt=800.0)
        result = compute_spike_risk_targets(df)
        assert result.extreme_spike_rate == 1.0  # rt=800 >= 800


class TestOutputColumns:
    def test_output_has_required_columns(self):
        df = _make_df(n=24)
        result = compute_spike_risk_targets(df)
        for col in OUTPUT_COLUMNS:
            assert col in result.df.columns, f"Missing output column: {col}"
