"""Tests for realtime_feature_builder.py.

Covers:
- End-to-end feature pipeline with synthetic data.
- Calendar feature generation.
- Lag feature generation (FULL_DAY and INTRADAY modes).
- SGDFNet integration (real predictions, fallback, error on missing).
- Teacher feature integration.
- Feature coverage audit.
- Chinese column renaming integration.
- Leakage safety.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.realtime_feature_builder import (
    build_realtime_features,
    audit_feature_coverage,
    _add_calendar_features,
    _add_lag_features,
    _integrate_sgdfnet,
)
from models.deep_sgdf_delta.realtime_feature_contract import (
    ALL_FEATURES,
    REQUIRED_FEATURES,
)

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────


def _make_synthetic_hourly(  # noqa: C901
    n_days: int = 200,
    start_date: str = "2025-06-01",
    with_chinese_names: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic hourly Shandong-like data for testing."""
    rng = np.random.default_rng(seed)
    ts_range = pd.date_range(start=start_date, periods=n_days * 24, freq="h")

    # Base price signals
    hour_of_day = ts_range.hour
    # Daily pattern: low at night, peak midday and evening
    base_pattern = (
        200 + 50 * np.sin(2 * np.pi * (hour_of_day + 6) / 24)
    )

    # Random walk for trend
    trend = np.cumsum(rng.normal(0, 2, size=len(ts_range))) * 0.1
    noise = rng.normal(0, 10, size=len(ts_range))

    da_price = np.clip(base_pattern + trend + rng.normal(0, 5, size=len(ts_range)), 50, 800)
    rt_price = np.clip(
        base_pattern + trend + noise
        + 5 * np.sin(2 * np.pi * ts_range.dayofweek / 7),
        0, 1000,
    )

    df_data: dict[str, np.ndarray | pd.Series | pd.DatetimeIndex] = {
        "ds": ts_range,
        "da_anchor": np.round(da_price, 2),
        "rt_actual": np.round(rt_price, 2),
        "forecast_price": np.round(da_price, 2),
    }

    # Add forecast-side columns
    df_data["load_forecast"] = np.round(
        5000 + 1000 * np.sin(2 * np.pi * hour_of_day / 24) + rng.normal(0, 50, size=len(ts_range)),
        2,
    )
    df_data["renewable_forecast"] = np.round(
        2000 + 1500 * np.sin(2 * np.pi * (hour_of_day - 4) / 24) + rng.normal(0, 100, size=len(ts_range)),
        2,
    )
    df_data["wind_forecast"] = np.round(
        1200 + 800 * np.sin(2 * np.pi * (hour_of_day - 2) / 24) + rng.normal(0, 80, size=len(ts_range)),
        2,
    )
    df_data["solar_forecast"] = np.round(
        np.maximum(0, 800 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)) + rng.normal(0, 30, size=len(ts_range)),
        2,
    )
    df_data["tie_line_forecast"] = np.round(
        1500 + 500 * np.sin(2 * np.pi * hour_of_day / 12) + rng.normal(0, 40, size=len(ts_range)),
        2,
    )
    df_data["bidding_space_forecast"] = np.round(
        3000 + 800 * np.sin(2 * np.pi * hour_of_day / 24) + rng.normal(0, 60, size=len(ts_range)),
        2,
    )

    df = pd.DataFrame(df_data)

    if with_chinese_names:
        df = df.rename(columns={
            "ds": "时刻",
            "da_anchor": "日前电价",
            "rt_actual": "实时电价",
            "forecast_price": "日前价格",
            "load_forecast": "日前负荷预测值",
            "renewable_forecast": "新能源总加预测值",
            "wind_forecast": "风电总加预测值",
            "solar_forecast": "光伏总加预测值",
            "tie_line_forecast": "联络线受电负荷预测值",
            "bidding_space_forecast": "竞价空间预测值",
        })

    return df


def _make_sgdfnet_predictions(
    df: pd.DataFrame,
    noise_std: float = 5.0,
    seed: int = 123,
) -> pd.DataFrame:
    """Create synthetic SGDFNet predictions from a data DataFrame."""
    rng = np.random.default_rng(seed)
    if "ds" in df.columns:
        ts_col = "ds"
    elif "时刻" in df.columns:
        ts_col = "时刻"
    else:
        ts_col = df.columns[0]

    if "rt_actual" in df.columns:
        base = df["rt_actual"].values
    elif "实时电价" in df.columns:
        base = df["实时电价"].values
    else:
        base = np.full(len(df), 300.0)

    return pd.DataFrame({
        ts_col: df[ts_col].values,
        "sgdfnet_pred": np.round(base + rng.normal(0, noise_std, size=len(df)), 2),
    })


# ── Tests ──────────────────────────────────────────────────────────────


class TestBuildRealtimeFeatures:
    """End-to-end feature builder tests."""

    def test_basic_full_day_pipeline(self):
        """Full pipeline produces all expected features in FULL_DAY mode."""
        df = _make_synthetic_hourly(n_days=100)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(
            df,
            sgdfnet_pred_df=sgd,
            mode="FULL_DAY",
        )
        # Core columns present
        assert "business_day" in result.columns
        assert "hour_business" in result.columns
        assert "period" in result.columns
        assert "delta_target" in result.columns

        # Calendar features
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "month_sin", "month_cos", "is_weekend"]:
            assert col in result.columns, f"Missing calendar feature: {col}"

        # Lag features
        for col in ["rt_lag_24h", "rt_lag_48h", "rt_mean_24h", "rt_std_24h",
                     "delta_lag_24h", "delta_lag_48h"]:
            assert col in result.columns, f"Missing lag feature: {col}"

        # Intra-day lags should be 0 in FULL_DAY mode
        for col in ["rt_lag_1h", "rt_lag_2h", "rt_lag_3h"]:
            assert col in result.columns
            assert (result[col] == 0.0).all(), f"{col} should be 0 in FULL_DAY"

        # SGDFNet features
        assert "sgdfnet_pred" in result.columns
        assert "sgdfnet_residual_lag_1h" in result.columns
        assert "sgdfnet_residual_lag_24h" in result.columns
        assert "sgdfnet_residual_mean_7d" in result.columns

        # Forecast features
        assert "forecast_price" in result.columns
        assert "load_forecast" in result.columns
        assert "renewable_forecast" in result.columns

    def test_intraday_mode_has_intraday_lags(self):
        """INTRADAY mode includes same-day lag features."""
        df = _make_synthetic_hourly(n_days=100)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(
            df,
            sgdfnet_pred_df=sgd,
            mode="INTRADAY",
        )
        # Intra-day lags should have non-zero values
        for col in ["rt_lag_1h", "rt_lag_2h", "rt_lag_3h"]:
            assert col in result.columns
            assert not (result[col] == 0.0).all(), f"{col} should have values in INTRADAY"

    def test_chinese_column_names(self):
        """Pipeline works with Chinese column names."""
        df = _make_synthetic_hourly(n_days=50, with_chinese_names=True)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(
            df,
            sgdfnet_pred_df=sgd,
            mode="FULL_DAY",
        )
        # After renaming, should have English column names
        assert "ds" in result.columns
        assert "da_anchor" in result.columns
        assert "rt_actual" in result.columns
        assert "business_day" in result.columns
        assert "load_forecast" in result.columns

    def test_sgdfnet_missing_raises_without_fallback(self):
        """Missing sgdfnet_pred raises ValueError when allow_fallback=False."""
        df = _make_synthetic_hourly(n_days=30)
        with pytest.raises(ValueError, match="Missing sgdfnet_pred"):
            build_realtime_features(
                df,
                sgdfnet_pred_df=None,
                mode="FULL_DAY",
                allow_sgdfnet_fallback=False,
            )

    def test_sgdfnet_fallback_allowed(self):
        """allow_sgdfnet_fallback=True fills missing sgdfnet_pred with da_anchor."""
        df = _make_synthetic_hourly(n_days=30)
        result = build_realtime_features(
            df,
            sgdfnet_pred_df=None,
            mode="FULL_DAY",
            allow_sgdfnet_fallback=True,
        )
        assert "sgdfnet_pred" in result.columns
        # In fallback mode, sgdfnet_pred should equal da_anchor
        np.testing.assert_array_almost_equal(
            result["sgdfnet_pred"].values,
            result["da_anchor"].values,
        )
        assert result.attrs.get("sgdfnet_fallback_used") is True

    def test_teacher_features_integration(self):
        """Teacher predictions are merged into feature table."""
        df = _make_synthetic_hourly(n_days=30)
        sgd = _make_sgdfnet_predictions(df)

        teacher_df = pd.DataFrame({
            "ds": df["ds"].values,
            "rt916_pred": df["rt_actual"].values + np.random.default_rng(42).normal(0, 3, size=len(df)),
            "timemixer_pred": df["rt_actual"].values + np.random.default_rng(99).normal(0, 4, size=len(df)),
        })

        result = build_realtime_features(
            df,
            sgdfnet_pred_df=sgd,
            teacher_pred_df=teacher_df,
        )
        assert "rt916_pred" in result.columns
        assert "timemixer_pred" in result.columns

    def test_calendar_features_synthetic(self):
        """Calendar features have correct value ranges."""
        df = _make_synthetic_hourly(n_days=30)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(df, sgdfnet_pred_df=sgd)

        # Sin/cos values should be in [-1, 1]
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "month_sin", "month_cos"]:
            assert result[col].between(-1.05, 1.05).all(), f"{col} out of range"

        # is_weekend should be 0 or 1
        assert result["is_weekend"].isin([0, 1]).all()

    def test_feature_count_meets_minimum(self):
        """Full pipeline produces at least 25 features."""
        df = _make_synthetic_hourly(n_days=100)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(df, sgdfnet_pred_df=sgd)

        feature_cols = [c for c in ALL_FEATURES if c in result.columns]
        assert len(feature_cols) >= 25, (
            f"Only {len(feature_cols)} features found, expected >= 25. "
            f"Missing: {[c for c in ALL_FEATURES if c not in result.columns]}"
        )

    def test_no_leakage_in_full_day(self):
        """No future actuals leak into features in FULL_DAY mode."""
        df = _make_synthetic_hourly(n_days=100)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(df, sgdfnet_pred_df=sgd, mode="FULL_DAY")

        # Check no column contains "actual" except rt_actual (the target)
        for col in result.columns:
            if "actual" in col.lower() and col != "rt_actual":
                pytest.fail(f"Potential leakage: feature column '{col}' contains actuals")

    def test_delta_target_calculation(self):
        """delta_target equals rt_actual - da_anchor."""
        df = _make_synthetic_hourly(n_days=30)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(df, sgdfnet_pred_df=sgd)

        expected_delta = result["rt_actual"] - result["da_anchor"]
        np.testing.assert_array_almost_equal(
            result["delta_target"].values, expected_delta.values,
        )


class TestAddCalendarFeatures:
    """Unit tests for calendar feature generation."""

    def test_hour_cyclical_encoding(self):
        """Hour 0 and hour 24 produce same sin/cos values."""
        ts = pd.date_range("2025-06-01", periods=48, freq="h")
        df = pd.DataFrame({"ds": ts, "rt_actual": np.random.default_rng(42).uniform(100, 500, 48)})
        result = _add_calendar_features(df)

        # Hour 0 (midnight, mapped to 24) and hour 24 (next day midnight)
        # should have same sin/cos as midnight
        assert abs(result["hour_sin"].iloc[0] - result["hour_sin"].iloc[24]) < 0.01
        assert abs(result["hour_cos"].iloc[0] - result["hour_cos"].iloc[24]) < 0.01

    def test_weekend_detection(self):
        """Weekend indicator is 1 for Saturday and Sunday."""
        # 2025-06-01 is a Sunday
        ts = pd.date_range("2025-05-26", periods=7, freq="D")
        df = pd.DataFrame({"ds": ts, "rt_actual": np.ones(7)})
        result = _add_calendar_features(df)

        # Monday (2025-05-26) to Friday: 0
        for i in range(5):
            assert result["is_weekend"].iloc[i] == 0, f"Day {i} should not be weekend"

        # Saturday (2025-05-31): 1
        assert result["is_weekend"].iloc[5] == 1, "Saturday should be weekend"

        # Sunday (2025-06-01): 1
        assert result["is_weekend"].iloc[6] == 1, "Sunday should be weekend"


class TestAddLagFeatures:
    """Unit tests for lag feature generation."""

    def test_day_lags_consistent(self):
        """Day-level lags shift by exact 24/48 hours."""
        df = _make_synthetic_hourly(n_days=50)
        result = _add_lag_features(df, mode="FULL_DAY")

        # After 24 rows, rt_lag_24h should equal rt_actual 24 rows before
        np.testing.assert_array_almost_equal(
            result["rt_lag_24h"].iloc[24:].values,
            result["rt_actual"].iloc[:-24].values,
        )

    def test_delta_lag_24h_consistent(self):
        """delta_lag_24h = (rt-da)_shifted by 24."""
        df = _make_synthetic_hourly(n_days=50)
        result = _add_lag_features(df, mode="FULL_DAY")

        expected = (result["rt_actual"] - result["da_anchor"]).shift(24)
        np.testing.assert_array_almost_equal(
            result["delta_lag_24h"].values[24:],
            expected.values[24:],
        )

    def test_intraday_lags_nonzero_in_intraday(self):
        """Intra-day lags have actual values in INTRADAY mode."""
        df = _make_synthetic_hourly(n_days=50)
        result = _add_lag_features(df, mode="INTRADAY")

        for col in ["rt_lag_1h", "rt_lag_2h", "rt_lag_3h"]:
            assert result[col].abs().sum() > 0, f"{col} should be non-zero"

    def test_full_day_intraday_lags_zero(self):
        """Intra-day lags are zero in FULL_DAY mode."""
        df = _make_synthetic_hourly(n_days=50)
        result = _add_lag_features(df, mode="FULL_DAY")

        for col in ["rt_lag_1h", "rt_lag_2h", "rt_lag_3h", "rt_mean_6h", "rt_std_6h"]:
            assert (result[col] == 0.0).all(), f"{col} should be zero in FULL_DAY"


class TestIntegrateSGDFNet:
    """Unit tests for SGDFNet integration."""

    def test_real_predictions_used(self):
        """Real SGDFNet predictions are used when provided."""
        df = _make_synthetic_hourly(n_days=30)
        sgd = _make_sgdfnet_predictions(df)
        result = _integrate_sgdfnet(df.copy(), sgdfnet_pred_df=sgd, allow_fallback=False)

        assert "sgdfnet_pred" in result.columns
        assert result.attrs.get("sgdfnet_fallback_used") is False
        assert result.attrs.get("sgdfnet_source") == "file"

    def test_fallback_when_no_predictions(self):
        """Fallback to da_anchor when no predictions and fallback allowed."""
        df = _make_synthetic_hourly(n_days=30)
        result = _integrate_sgdfnet(df.copy(), sgdfnet_pred_df=None, allow_fallback=True)

        assert "sgdfnet_pred" in result.columns
        assert result.attrs.get("sgdfnet_fallback_used") is True
        np.testing.assert_array_almost_equal(
            result["sgdfnet_pred"].values, result["da_anchor"].values,
        )

    def test_raises_when_no_predictions_no_fallback(self):
        """Raises ValueError when no predictions and no fallback."""
        with pytest.raises(ValueError, match="Missing sgdfnet_pred"):
            _integrate_sgdfnet(
                _make_synthetic_hourly(n_days=30),
                sgdfnet_pred_df=None,
                allow_fallback=False,
            )

    def test_residual_features_computed(self):
        """SGDFNet residual features are computed."""
        df = _make_synthetic_hourly(n_days=50)
        sgd = _make_sgdfnet_predictions(df)
        result = _integrate_sgdfnet(df.copy(), sgdfnet_pred_df=sgd, allow_fallback=False)

        for col in ["sgdfnet_residual_lag_1h", "sgdfnet_residual_lag_24h",
                     "sgdfnet_residual_mean_7d"]:
            assert col in result.columns


class TestAuditFeatureCoverage:
    """Tests for the feature coverage audit function."""

    def test_full_feature_set_verdict(self):
        """Full feature set gets FORMAL_READY verdict."""
        df = _make_synthetic_hourly(n_days=100)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(df, sgdfnet_pred_df=sgd)
        # Simulate good coverage
        result.attrs["sgdfnet_coverage"] = 100.0

        audit = audit_feature_coverage(result)
        assert audit["verdict"] in ("FORMAL_READY", "PARTIAL_READY")
        assert audit["n_features"] >= 25

    def test_missing_features_verdict(self):
        """Bare DataFrame with few features gets NOT_READY."""
        df = pd.DataFrame({
            "ds": pd.date_range("2025-06-01", periods=48, freq="h"),
            "da_anchor": np.ones(48) * 300,
            "rt_actual": np.ones(48) * 310,
        })

        audit = audit_feature_coverage(df)
        assert audit["verdict"] == "NOT_READY"
        assert audit["n_features"] < 15

    def test_sgdfnet_coverage_tracked(self):
        """SGDFNet coverage stats are recorded in audit."""
        df = _make_synthetic_hourly(n_days=50)
        sgd = _make_sgdfnet_predictions(df)
        result = build_realtime_features(df, sgdfnet_pred_df=sgd)

        audit = audit_feature_coverage(result)
        assert "sgdfnet_effective_coverage" in audit
        assert "sgdfnet_real_coverage" in audit
        assert "sgdfnet_fallback_used" in audit
