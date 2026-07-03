"""Tests for negative_risk_targets module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.negative_risk_targets import (
    NegativeRiskThresholds,
    compute_negative_risk_targets,
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
        result = compute_negative_risk_targets(df)
        row = result.df.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-02-01")
        assert row["hour_business"] == 24


class TestNegativeLabel:
    def test_negative_when_rt_below_zero(self):
        df = _make_df(n=24, da=300.0, rt=-10.0)  # rt = -10 < 0
        result = compute_negative_risk_targets(df)
        assert (result.df["negative_label"] == 1).all()

    def test_negative_when_rt_deeply_negative(self):
        df = _make_df(n=24, da=300.0, rt=-200.0)  # rt = -200 < 0
        result = compute_negative_risk_targets(df)
        assert (result.df["negative_label"] == 1).all()

    def test_no_negative_when_rt_zero(self):
        df = _make_df(n=24, da=300.0, rt=0.0)  # rt = 0, not < 0
        result = compute_negative_risk_targets(df)
        assert (result.df["negative_label"] == 0).all()

    def test_no_negative_when_rt_positive(self):
        df = _make_df(n=24, da=300.0, rt=100.0)  # rt = 100 > 0
        result = compute_negative_risk_targets(df)
        assert (result.df["negative_label"] == 0).all()


class TestDeepNegativeLabel:
    def test_deep_negative_when_rt_at_minus_100(self):
        df = _make_df(n=24, da=300.0, rt=-100.0)  # rt = -100 <= -100
        result = compute_negative_risk_targets(df)
        assert (result.df["deep_negative_label"] == 1).all()

    def test_deep_negative_when_rt_below_minus_100(self):
        df = _make_df(n=24, da=300.0, rt=-200.0)  # rt = -200 <= -100
        result = compute_negative_risk_targets(df)
        assert (result.df["deep_negative_label"] == 1).all()

    def test_no_deep_negative_when_rt_above_minus_100(self):
        df = _make_df(n=24, da=300.0, rt=-99.0)  # rt = -99 > -100
        result = compute_negative_risk_targets(df)
        assert (result.df["deep_negative_label"] == 0).all()


class TestRelativeDownLabel:
    def test_relative_down_when_delta_at_minus_200(self):
        df = _make_df(n=24, da=300.0, rt=100.0)  # delta = -200 <= -200
        result = compute_negative_risk_targets(df)
        assert (result.df["relative_down_label"] == 1).all()

    def test_relative_down_when_delta_below_minus_200(self):
        df = _make_df(n=24, da=300.0, rt=50.0)  # delta = -250 <= -200
        result = compute_negative_risk_targets(df)
        assert (result.df["relative_down_label"] == 1).all()

    def test_no_relative_down_when_delta_above_minus_200(self):
        df = _make_df(n=24, da=300.0, rt=101.0)  # delta = -199 > -200
        result = compute_negative_risk_targets(df)
        assert (result.df["relative_down_label"] == 0).all()


class TestMissingData:
    def test_missing_da_anchor_raises(self):
        df = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=24, freq="h"),
                           "rt_actual": 300.0})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_negative_risk_targets(df)

    def test_missing_rt_actual_raises(self):
        df = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=24, freq="h"),
                           "da_anchor": 300.0})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_negative_risk_targets(df)

    def test_nan_values_produce_minus1_labels(self):
        df = _make_df(n=24, da=300.0, rt=-10.0)
        df.loc[0, "da_anchor"] = np.nan
        result = compute_negative_risk_targets(df)
        assert result.df.iloc[0]["negative_label"] == -1
        assert result.df.iloc[0]["deep_negative_label"] == -1
        assert result.df.iloc[0]["relative_down_label"] == -1


class TestStatistics:
    def test_statistics_computed(self):
        df = _make_df(n=48, da=300.0, rt=-10.0)
        result = compute_negative_risk_targets(df)
        assert result.n_valid == 48
        assert result.negative_rate == 1.0  # rt=-10 < 0
        assert result.mean_rt == -10.0


class TestOutputColumns:
    def test_output_has_required_columns(self):
        df = _make_df(n=24)
        result = compute_negative_risk_targets(df)
        for col in OUTPUT_COLUMNS:
            assert col in result.df.columns, f"Missing output column: {col}"
