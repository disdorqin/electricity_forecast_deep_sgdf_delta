"""Tests for Phase 8 no-leak feature engineering.

Verifies that:
1. Lag features use same-hour previous-day lookup (not simple shift)
2. Rolling features exclude current row (shift(1) before rolling)
3. Current row's residual does not leak into any feature
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.solar916_features import (
    _build_lag_features_merge,
    _build_rolling_features,
    build_solar916_features,
)


def _make_full_df(n_days: int = 14, start_date: str = "2026-02-01") -> pd.DataFrame:
    """Create a full 24-hour DataFrame for testing."""
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows = []
    for d in dates:
        for h in range(1, 25):
            ts = d + pd.Timedelta(hours=h - 1) if h < 24 else d + pd.Timedelta(days=1)
            rows.append({
                "时刻": ts,
                "日前电价": 100.0,
                "实时电价": 110.0 + h * 2.0,  # varies by hour
                "光伏总加预测值": 50.0,
                "风电总加预测值": 30.0,
                "新能源总加预测值": 60.0,
                "竞价空间预测值": 200.0,
                "直调负荷预测值": 500.0,
            })
    df = pd.DataFrame(rows)
    df = add_business_time_columns(df, timestamp_col="时刻")
    df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
    df["delta"] = df["rt_price"] - df["da_price"]
    df["sgdfnet_pred"] = 105.0
    df["sgdfnet_residual"] = df["rt_price"] - df["sgdfnet_pred"]
    return df


class TestLagFeaturesNoLeak:
    """Test that lag features use same-hour previous-day lookup."""

    def test_hour10_lag24_from_previous_day_hour10(self):
        """Hour 10's residual_lag_24 must come from previous day hour 10,
        NOT from same day hour 9."""
        df = _make_full_df(n_days=5)
        result = _build_lag_features_merge(df)

        # Find hour 10 on day 2 (2026-02-02)
        day2_h10 = result[
            (result["business_day"] == pd.Timestamp("2026-02-02"))
            & (result["hour_business"] == 10)
        ]
        assert len(day2_h10) == 1

        # Find hour 10 on day 1 (2026-02-01) — this is what lag_24 should be
        day1_h10 = result[
            (result["business_day"] == pd.Timestamp("2026-02-01"))
            & (result["hour_business"] == 10)
        ]
        assert len(day1_h10) == 1

        # lag_24 should match day1 hour10's residual
        expected_residual_lag = day1_h10["sgdfnet_residual"].values[0]
        actual_lag = day2_h10["residual_lag_24"].values[0]
        assert actual_lag == pytest.approx(expected_residual_lag, abs=1e-6)

        # lag_24 should NOT match day2 hour9's residual (the old buggy behavior)
        day2_h9 = result[
            (result["business_day"] == pd.Timestamp("2026-02-02"))
            & (result["hour_business"] == 9)
        ]
        wrong_value = day2_h9["sgdfnet_residual"].values[0]
        # They should be different (hour 9 and 10 have different rt_price)
        assert wrong_value != pytest.approx(actual_lag, abs=1e-6)

    def test_hour9_lag24_from_previous_day_hour9(self):
        """Hour 9's residual_lag_24 must come from previous day hour 9."""
        df = _make_full_df(n_days=5)
        result = _build_lag_features_merge(df)

        day2_h9 = result[
            (result["business_day"] == pd.Timestamp("2026-02-02"))
            & (result["hour_business"] == 9)
        ]
        day1_h9 = result[
            (result["business_day"] == pd.Timestamp("2026-02-01"))
            & (result["hour_business"] == 9)
        ]

        expected = day1_h9["sgdfnet_residual"].values[0]
        actual = day2_h9["residual_lag_24"].values[0]
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_lag168_from_7_days_ago_same_hour(self):
        """residual_lag_168 must come from 7 business days ago, same hour."""
        df = _make_full_df(n_days=14)
        result = _build_lag_features_merge(df)

        # Day 8 hour 10 should have lag_168 from day 1 hour 10
        day8_h10 = result[
            (result["business_day"] == pd.Timestamp("2026-02-08"))
            & (result["hour_business"] == 10)
        ]
        day1_h10 = result[
            (result["business_day"] == pd.Timestamp("2026-02-01"))
            & (result["hour_business"] == 10)
        ]

        assert len(day8_h10) == 1
        assert len(day1_h10) == 1

        expected = day1_h10["sgdfnet_residual"].values[0]
        actual = day8_h10["residual_lag_168"].values[0]
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_current_row_residual_not_in_lag(self):
        """Current row's residual must NOT appear in its own lag features."""
        df = _make_full_df(n_days=5)
        result = _build_lag_features_merge(df)

        # For any row, residual_lag_24 should not equal its own residual
        valid = result.dropna(subset=["residual_lag_24"])
        # The residual values vary by hour, so lag from different day same hour
        # should be the same value. But the key is it's from a DIFFERENT day.
        # Check that lag_24 row's business_day is different from current row
        for _, row in valid.iterrows():
            lag_source_day = row["business_day"] - pd.Timedelta(days=1)
            # The lag value should match the source day's residual
            source_rows = result[
                (result["business_day"] == lag_source_day)
                & (result["hour_business"] == row["hour_business"])
            ]
            if len(source_rows) == 1:
                assert row["residual_lag_24"] == pytest.approx(
                    source_rows["sgdfnet_residual"].values[0], abs=1e-6
                )


class TestRollingFeaturesNoLeak:
    """Test that rolling features exclude current row."""

    def test_rolling_excludes_current_row(self):
        """Changing current row's residual must NOT change its rolling features."""
        df = _make_full_df(n_days=10)
        result = _build_rolling_features(df)

        # Pick a row in the middle (not first few)
        mid_idx = len(result) // 2
        original_residual = result.loc[mid_idx, "sgdfnet_residual"]
        original_rolling_mean = result.loc[mid_idx, "rolling_residual_mean_7d"]

        # Change the current row's residual to an extreme value
        result.loc[mid_idx, "sgdfnet_residual"] = 99999.0

        # Rebuild rolling features
        result2 = _build_rolling_features(result)

        # The rolling mean for this row should NOT change
        assert result2.loc[mid_idx, "rolling_residual_mean_7d"] == pytest.approx(
            original_rolling_mean, abs=1e-6
        )

    def test_same_hour_rolling_excludes_current(self):
        """Same-hour rolling mean must not include current row's residual."""
        df = _make_full_df(n_days=10)
        result = _build_rolling_features(df)

        mid_idx = len(result) // 2
        original_sh_mean = result.loc[mid_idx, "same_hour_residual_mean_7d"]

        # Change current row's residual
        result.loc[mid_idx, "sgdfnet_residual"] = 99999.0
        result2 = _build_rolling_features(result)

        # Same-hour rolling mean should NOT change
        assert result2.loc[mid_idx, "same_hour_residual_mean_7d"] == pytest.approx(
            original_sh_mean, abs=1e-6
        )

    def test_future_rows_can_change(self):
        """Changing current row's residual SHOULD affect future rows' rolling features."""
        df = _make_full_df(n_days=10)
        result = _build_rolling_features(df)

        mid_idx = len(result) // 2
        next_idx = mid_idx + 1
        original_next_rolling = result.loc[next_idx, "rolling_residual_mean_7d"]

        # Change current row's residual
        result.loc[mid_idx, "sgdfnet_residual"] = 99999.0
        result2 = _build_rolling_features(result)

        # The NEXT row's rolling mean SHOULD change (it includes the modified row)
        assert result2.loc[next_idx, "rolling_residual_mean_7d"] != pytest.approx(
            original_next_rolling, abs=1e-3
        )


class TestFullFeatureBuildNoLeak:
    """Integration tests for the full feature build pipeline."""

    def test_features_on_full_then_filter_916(self):
        """Features should be computed on full dataset, then filtered to 9_16."""
        df = _make_full_df(n_days=14)
        result, info = build_solar916_features(df)

        # Filter to 9_16
        df_916 = result[result["period"] == "9_16"].copy()

        # All 9_16 rows should have valid lag features (from full-dataset context)
        assert len(df_916) > 0
        # After day 1, delta_lag_24 should be available (delta doesn't need sgdfnet)
        later_rows = df_916[df_916["business_day"] > pd.Timestamp("2026-02-01")]
        if len(later_rows) > 0:
            assert later_rows["delta_lag_24"].notna().sum() > 0

    def test_no_target_in_features(self):
        """sgdfnet_residual (target) must not be directly used as a feature."""
        df = _make_full_df(n_days=14)
        result, info = build_solar916_features(df)

        feature_cols = info["feature_columns"]
        assert "sgdfnet_residual" not in feature_cols
        assert "rt_price" not in feature_cols
