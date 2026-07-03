"""Tests for negative_risk_features module."""
import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.negative_risk_features import (
    build_negative_risk_features,
    NEGATIVE_DERIVED_FEATURES,
    NEGATIVE_CALENDAR_FEATURES,
    NEGATIVE_LAG_FEATURES,
    NEGATIVE_ANCHOR_FEATURES,
    NegativeFeatureAudit,
)


def _make_raw_df(n_days=5, with_forecast=True):
    """Create a raw test DataFrame with Chinese column names."""
    n_hours = n_days * 24
    dates = pd.date_range("2026-02-01", periods=n_hours, freq="h")
    data = {
        "时刻": dates,
        "日前电价": np.random.uniform(200, 400, n_hours),
        "实时电价": np.random.uniform(-100, 500, n_hours),
    }
    if with_forecast:
        data["统调负荷预测值"] = np.random.uniform(8000, 12000, n_hours)
        data["新能源总加预测值"] = np.random.uniform(5000, 10000, n_hours)
        data["风电总加预测值"] = np.random.uniform(3000, 8000, n_hours)
        data["光伏总加预测值"] = np.random.uniform(0, 3000, n_hours)
        data["竞价空间预测值"] = np.random.uniform(10000, 30000, n_hours)
    return pd.DataFrame(data)


class TestFullDayNoLeakage:
    """FULL_DAY mode must not use target-day actual features."""

    def test_full_day_lag_features_are_shifted(self):
        df = _make_raw_df(n_days=5)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        # Lag features should be shifted by 24h, so first 24 rows should have NaN
        lag_cols = ["previous_day_price_delta_mean", "previous_day_price_delta_std"]
        for col in lag_cols:
            if col in result.df.columns:
                # First 24 rows should be NaN (shifted by 24)
                assert result.df[col].iloc[:24].isna().all(), \
                    f"{col} should be NaN for first 24 rows in FULL_DAY mode"


class TestFeatureAudit:
    def test_audit_verdict_with_full_data(self):
        df = _make_raw_df(n_days=10, with_forecast=True)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        # With full forecast data, should be at least PARTIAL_READY
        assert result.audit.verdict in ("FORMAL_READY", "PARTIAL_READY")

    def test_audit_verdict_with_minimal_data(self):
        n = 48
        dates = pd.date_range("2026-02-01", periods=n, freq="h")
        df = pd.DataFrame({
            "ds": dates,
            "da_anchor": np.random.uniform(200, 400, n),
            "rt_actual": np.random.uniform(-100, 500, n),
        })
        result = build_negative_risk_features(df, mode="FULL_DAY")
        # Should be NOT_READY or PARTIAL_READY
        assert result.audit.verdict in ("NOT_READY", "PARTIAL_READY")

    def test_audit_has_required_fields(self):
        df = _make_raw_df(n_days=5)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        audit = result.audit
        assert hasattr(audit, "n_features")
        assert hasattr(audit, "feature_columns")
        assert hasattr(audit, "missing_features")
        assert hasattr(audit, "derived_feature_coverage")
        assert hasattr(audit, "lag_feature_coverage")
        assert hasattr(audit, "calendar_feature_coverage")
        assert hasattr(audit, "leakage_check")
        assert hasattr(audit, "verdict")

    def test_audit_leakage_check_passes(self):
        df = _make_raw_df(n_days=5)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        assert result.audit.leakage_check is True


class TestCalendarFeatures:
    def test_calendar_features_present(self):
        df = _make_raw_df(n_days=3)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        for feat in NEGATIVE_CALENDAR_FEATURES:
            assert feat in result.df.columns, f"Missing calendar feature: {feat}"

    def test_hour_sin_cos_range(self):
        df = _make_raw_df(n_days=3)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        assert result.df["hour_sin"].between(-1, 1).all()
        assert result.df["hour_cos"].between(-1, 1).all()

    def test_month_sin_cos_range(self):
        df = _make_raw_df(n_days=3)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        assert result.df["month_sin"].between(-1, 1).all()
        assert result.df["month_cos"].between(-1, 1).all()


class TestMissingForecastColumns:
    def test_missing_forecast_columns_in_missing_features(self):
        df = _make_raw_df(n_days=3, with_forecast=False)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        missing_derived = [f for f in NEGATIVE_DERIVED_FEATURES if f in result.audit.missing_features]
        assert len(missing_derived) > 0, "Missing derived features should be reported"

    def test_no_forecast_means_lower_coverage(self):
        df = _make_raw_df(n_days=3, with_forecast=False)
        result = build_negative_risk_features(df, mode="FULL_DAY")
        assert result.audit.derived_feature_coverage < 0.5
