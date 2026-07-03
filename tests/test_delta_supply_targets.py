"""Tests for delta_supply_targets module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.delta_supply_targets import (
    DeltaSupplyThresholds,
    compute_delta_supply_targets,
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
    """Test 00:00 → previous business_day, hour 24."""

    def test_midnight_maps_to_previous_day_hour_24(self):
        df = pd.DataFrame({
            "ds": [pd.Timestamp("2026-02-02 00:00:00")],
            "da_anchor": [300.0],
            "rt_actual": [350.0],
        })
        result = compute_delta_supply_targets(df)
        row = result.df.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-02-01")
        assert row["hour_business"] == 24

    def test_hour_1_maps_to_same_day_hour_1(self):
        df = pd.DataFrame({
            "ds": [pd.Timestamp("2026-02-02 01:00:00")],
            "da_anchor": [300.0],
            "rt_actual": [350.0],
        })
        result = compute_delta_supply_targets(df)
        row = result.df.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-02-02")
        assert row["hour_business"] == 1


class TestUpwardLabel:
    def test_upward_when_delta_above_threshold(self):
        df = _make_df(n=24, da=300.0, rt=450.0)  # delta = 150 >= 100
        result = compute_delta_supply_targets(df)
        assert (result.df["upward_deviation_label"] == 1).all()

    def test_no_upward_when_delta_below_threshold(self):
        df = _make_df(n=24, da=300.0, rt=350.0)  # delta = 50 < 100
        result = compute_delta_supply_targets(df)
        assert (result.df["upward_deviation_label"] == 0).all()


class TestDownwardLabel:
    def test_downward_when_delta_below_negative_threshold(self):
        df = _make_df(n=24, da=300.0, rt=150.0)  # delta = -150 <= -100
        result = compute_delta_supply_targets(df)
        assert (result.df["downward_deviation_label"] == 1).all()

    def test_no_downward_when_delta_above_threshold(self):
        df = _make_df(n=24, da=300.0, rt=350.0)  # delta = 50 > -100
        result = compute_delta_supply_targets(df)
        assert (result.df["downward_deviation_label"] == 0).all()


class TestLargeAbsLabel:
    def test_large_abs_when_abs_delta_above_threshold(self):
        df = _make_df(n=24, da=300.0, rt=500.0)  # |delta| = 200 >= 150
        result = compute_delta_supply_targets(df)
        assert (result.df["large_abs_deviation_label"] == 1).all()

    def test_no_large_abs_when_small_delta(self):
        df = _make_df(n=24, da=300.0, rt=350.0)  # |delta| = 50 < 150
        result = compute_delta_supply_targets(df)
        assert (result.df["large_abs_deviation_label"] == 0).all()


class TestMagnitudeClipping:
    def test_clipping_at_500(self):
        df = _make_df(n=24, da=0.0, rt=1000.0)  # delta = 1000, clipped to 500
        result = compute_delta_supply_targets(df)
        assert (result.df["deviation_magnitude_target"] == 500.0).all()

    def test_negative_clipping(self):
        df = _make_df(n=24, da=1000.0, rt=0.0)  # delta = -1000, clipped to -500
        result = compute_delta_supply_targets(df)
        assert (result.df["deviation_magnitude_target"] == -500.0).all()

    def test_no_clipping_within_range(self):
        df = _make_df(n=24, da=300.0, rt=400.0)  # delta = 100, within [-500, 500]
        result = compute_delta_supply_targets(df)
        assert (result.df["deviation_magnitude_target"] == 100.0).all()


class TestConfigurableThresholds:
    def test_custom_thresholds(self):
        thresholds = DeltaSupplyThresholds(upward=50, downward=-50, abs_large=75, clip=200)
        df = _make_df(n=24, da=300.0, rt=360.0)  # delta = 60
        result = compute_delta_supply_targets(df, thresholds=thresholds)
        assert (result.df["upward_deviation_label"] == 1).all()  # 60 >= 50
        assert (result.df["downward_deviation_label"] == 0).all()
        assert (result.df["large_abs_deviation_label"] == 0).all()  # |60| < 75


class TestMissingData:
    def test_missing_da_anchor_raises(self):
        df = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=24, freq="h"),
                           "rt_actual": 300.0})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_delta_supply_targets(df)

    def test_missing_rt_actual_raises(self):
        df = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=24, freq="h"),
                           "da_anchor": 300.0})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_delta_supply_targets(df)

    def test_nan_values_produce_minus1_labels(self):
        df = _make_df(n=24, da=300.0, rt=350.0)
        df.loc[0, "da_anchor"] = np.nan
        result = compute_delta_supply_targets(df)
        assert result.df.iloc[0]["upward_deviation_label"] == -1
        assert np.isnan(result.df.iloc[0]["deviation_magnitude_target"])

    def test_statistics_computed(self):
        df = _make_df(n=48, da=300.0, rt=450.0)
        result = compute_delta_supply_targets(df)
        assert result.n_valid == 48
        assert result.upward_rate == 1.0  # delta=150 >= 100
        assert result.mean_delta == 150.0


class TestOutputColumns:
    def test_output_has_required_columns(self):
        df = _make_df(n=24)
        result = compute_delta_supply_targets(df)
        for col in OUTPUT_COLUMNS:
            assert col in result.df.columns, f"Missing output column: {col}"
