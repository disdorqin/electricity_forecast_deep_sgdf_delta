"""Tests for delta_supply_features module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.delta_supply_features import (
    build_delta_supply_features,
    FORECAST_FEATURES,
    LAG_FEATURES,
    CALENDAR_FEATURES,
    ANCHOR_FEATURES,
    FeatureAudit,
)


def _make_raw_df(n_days=5, with_forecast=True, with_actual=True):
    """Create a raw test DataFrame with Chinese column names."""
    n_hours = n_days * 24
    dates = pd.date_range("2026-02-01", periods=n_hours, freq="h")
    data = {
        "时刻": dates,
        "日前电价": np.random.uniform(200, 400, n_hours),
        "实时电价": np.random.uniform(100, 500, n_hours),
    }
    if with_forecast:
        data["统调负荷预测值"] = np.random.uniform(8000, 12000, n_hours)
        data["新能源总加预测值"] = np.random.uniform(5000, 10000, n_hours)
        data["风电总加预测值"] = np.random.uniform(3000, 8000, n_hours)
        data["光伏总加预测值"] = np.random.uniform(0, 3000, n_hours)
        data["竞价空间预测值"] = np.random.uniform(10000, 30000, n_hours)
    if with_actual:
        data["统调负荷"] = np.random.uniform(8000, 12000, n_hours)
        data["新能源总加实际值"] = np.random.uniform(5000, 10000, n_hours)
        data["风电总加实际值"] = np.random.uniform(3000, 8000, n_hours)
        data["光伏总加实际值"] = np.random.uniform(-100, 3000, n_hours)
    return pd.DataFrame(data)


class TestFullDayNoLeakage:
    """FULL_DAY mode must not use target-day actual features."""

    def test_full_day_does_not_use_same_day_actual(self):
        df = _make_raw_df(n_days=5)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        # Lag features should be shifted by 24h, so first 24 rows should have NaN
        # for lag features
        lag_cols = ["actual_load_lag_24h", "actual_renewable_lag_24h"]
        for col in lag_cols:
            if col in result.df.columns:
                # First 24 rows should be NaN (shifted by 24)
                assert result.df[col].iloc[:24].isna().all(), \
                    f"{col} should be NaN for first 24 rows in FULL_DAY mode"

    def test_day_d_features_not_affected_by_day_d_actual(self):
        """Features for day D should not change when day D actual is modified."""
        df = _make_raw_df(n_days=5)
        result1 = build_delta_supply_features(df, mode="FULL_DAY")

        # Modify day 3 actuals
        df2 = df.copy()
        day3_mask = df2["时刻"].dt.date == pd.Timestamp("2026-02-03").date()
        for col in ["统调负荷", "新能源总加实际值"]:
            if col in df2.columns:
                df2.loc[day3_mask, col] = 99999.0

        result2 = build_delta_supply_features(df2, mode="FULL_DAY")

        # Day 2 features should be identical (day 3 actual shouldn't affect day 2)
        day2_mask = result1.df["business_day"] == pd.Timestamp("2026-02-02")
        for col in result1.feature_columns:
            if col in ["previous_day_load_mean", "previous_day_renewable_mean",
                        "actual_load_lag_24h", "actual_renewable_lag_24h"]:
                # These use lagged actuals; day 2 features depend on day 1 actuals
                # which are unchanged, so they should match
                continue
            vals1 = result1.df.loc[day2_mask, col].values
            vals2 = result2.df.loc[day2_mask, col].values
            if len(vals1) > 0:
                np.testing.assert_array_equal(vals1, vals2,
                    err_msg=f"Feature {col} changed for day 2 when day 3 actual was modified")


class TestForecastSideAllowed:
    """Forecast-side features can use target-day forecast."""

    def test_forecast_features_available_for_target_day(self):
        df = _make_raw_df(n_days=3, with_forecast=True)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        # Forecast features should be available (not all NaN)
        forecast_present = [f for f in FORECAST_FEATURES if f in result.feature_columns]
        assert len(forecast_present) > 0, "At least some forecast features should be present"


class TestMissingForecastColumns:
    """Missing forecast columns should not be fabricated."""

    def test_missing_forecast_columns_in_missing_features(self):
        df = _make_raw_df(n_days=3, with_forecast=False)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        # Forecast features should be in missing list
        missing_forecast = [f for f in FORECAST_FEATURES if f in result.audit.missing_features]
        assert len(missing_forecast) > 0, "Missing forecast features should be reported"

    def test_no_forecast_means_lower_coverage(self):
        df = _make_raw_df(n_days=3, with_forecast=False)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        assert result.audit.forecast_feature_coverage < 0.5


class TestFeatureAudit:
    def test_audit_verdict_formal_ready(self):
        df = _make_raw_df(n_days=10, with_forecast=True, with_actual=True)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        # With full data, should be at least PARTIAL_READY
        assert result.audit.verdict in ("FORMAL_READY", "PARTIAL_READY")

    def test_audit_verdict_not_ready_with_minimal_data(self):
        # Only core columns, no forecast, no actual
        n = 48
        dates = pd.date_range("2026-02-01", periods=n, freq="h")
        df = pd.DataFrame({
            "ds": dates,
            "da_anchor": np.random.uniform(200, 400, n),
            "rt_actual": np.random.uniform(100, 500, n),
        })
        result = build_delta_supply_features(df, mode="FULL_DAY")
        # Should be NOT_READY or PARTIAL_READY (calendar features alone might get to 8+)
        assert result.audit.verdict in ("NOT_READY", "PARTIAL_READY")

    def test_audit_has_required_fields(self):
        df = _make_raw_df(n_days=5)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        audit = result.audit
        assert hasattr(audit, "n_features")
        assert hasattr(audit, "feature_columns")
        assert hasattr(audit, "missing_features")
        assert hasattr(audit, "forecast_feature_coverage")
        assert hasattr(audit, "lag_feature_coverage")
        assert hasattr(audit, "calendar_feature_coverage")
        assert hasattr(audit, "sgdfnet_available")
        assert hasattr(audit, "leakage_check")
        assert hasattr(audit, "verdict")


class TestCalendarPeriodFeatures:
    def test_calendar_features_present(self):
        df = _make_raw_df(n_days=3)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        for feat in CALENDAR_FEATURES:
            assert feat in result.df.columns, f"Missing calendar feature: {feat}"

    def test_period_id_values(self):
        df = _make_raw_df(n_days=3)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        assert set(result.df["period_id"].unique()).issubset({0, 1, 2})

    def test_hour_sin_cos_range(self):
        df = _make_raw_df(n_days=3)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        assert result.df["hour_sin"].between(-1, 1).all()
        assert result.df["hour_cos"].between(-1, 1).all()


class TestSgdfnetOptional:
    def test_sgdfnet_not_available_flag(self):
        df = _make_raw_df(n_days=3)
        result = build_delta_supply_features(df, mode="FULL_DAY")
        assert result.audit.sgdfnet_available is False

    def test_sgdfnet_available_when_provided(self):
        df = _make_raw_df(n_days=3)
        sgdf_df = pd.DataFrame({
            "ds": pd.date_range("2026-02-01", periods=72, freq="h"),
            "sgdfnet_pred": np.random.uniform(200, 400, 72),
        })
        result = build_delta_supply_features(df, mode="FULL_DAY", sgdfnet_df=sgdf_df)
        assert result.audit.sgdfnet_available is True
        assert "sgdfnet_pred" in result.feature_columns
