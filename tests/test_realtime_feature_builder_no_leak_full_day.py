"""Leakage-safety tests for FULL_DAY mode feature builder.

Covers:
1. FULL_DAY rt_mean_24h not affected by D-day earlier actuals
2. FULL_DAY sgdfnet_residual_mean_7d not affected by D-day earlier actuals
3. FULL_DAY sgdfnet_residual_lag_1h is 0
4. FULL_DAY rt_lag_24h is D-1 same-hour actual
5. INTRADAY mode allows same-day earlier actuals with explicit flag
6. check_leakage on residual-derived features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.realtime_feature_builder import (
    _add_lag_features,
    _integrate_sgdfnet,
    build_realtime_features,
)
from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.realtime_feature_contract import check_leakage


# ── Helpers ────────────────────────────────────────────────────────────

def _make_hourly_df(n_days: int = 30, start_date: str = "2025-06-01",
                    seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly data with stable per-day patterns."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start_date, periods=n_days * 24, freq="h")
    df = pd.DataFrame({"ds": ts})
    df = add_business_time_columns(df)

    # Per-day pattern: assign a daily baseline so we can test D-day independence
    daily_baseline = {}
    for i, bd in enumerate(df["business_day"].unique()):
        daily_baseline[bd] = 200.0 + rng.uniform(-50, 50)

    df["da_anchor"] = 300.0
    df["rt_actual"] = df["business_day"].map(daily_baseline).values + rng.normal(0, 10, len(df))
    df["forecast_price"] = df["da_anchor"]
    df["sgdfnet_pred"] = df["rt_actual"] + rng.normal(0, 5, len(df))

    return df


# ── Tests ──────────────────────────────────────────────────────────────


class TestFullDayNoLeak:
    """FULL_DAY mode lag features must not leak same-day actuals."""

    def test_rt_mean_24h_not_affected_by_current_day_change(self):
        """
        Changing D-day early hour actuals should NOT affect rt_mean_24h
        for later hours on the same day.
        """
        df = _make_hourly_df(n_days=30)

        # Compute baseline FULL_DAY features
        baseline = _add_lag_features(df.copy(), mode="FULL_DAY")

        # Get a specific day D, hour 16's rt_mean_24h
        d_day = df["business_day"].unique()[15]  # pick a day in the middle
        mask_d16 = (df["business_day"] == d_day) & (df["hour_business"] == 16)
        idx_d16 = df.index[mask_d16][0]

        # Now modify D-day hour 1's rt_actual (same day, earlier hour)
        df_modified = df.copy()
        mask_d1 = (df_modified["business_day"] == d_day) & (df_modified["hour_business"] == 1)
        df_modified.loc[mask_d1, "rt_actual"] = 9999.0  # extreme value

        modified = _add_lag_features(df_modified, mode="FULL_DAY")

        # rt_mean_24h for hour 16 should be identical (based on D-1, not D-day)
        bd_16 = baseline.loc[idx_d16, "rt_mean_24h"]
        mod_16 = modified.loc[idx_d16, "rt_mean_24h"]
        assert abs(bd_16 - mod_16) < 0.01, (
            f"rt_mean_24h changed from {bd_16:.2f} to {mod_16:.2f} "
            f"after modifying same-day hour 1 — LEAKAGE!"
        )

    def test_sgdfnet_residual_mean_7d_not_affected_by_current_day(self):
        """Changing D-day actuals must not affect sgdfnet_residual_mean_7d."""
        df = _make_hourly_df(n_days=60)

        baseline = _integrate_sgdfnet(df.copy(), allow_fallback=True, mode="FULL_DAY")

        d_day = df["business_day"].unique()[30]
        mask_d16 = (df["business_day"] == d_day) & (df["hour_business"] == 16)
        idx = df.index[mask_d16][0]

        # Modify D-day hour 1's rt_actual
        df_modified = df.copy()
        mask_d1 = (df_modified["business_day"] == d_day) & (df_modified["hour_business"] == 1)
        df_modified.loc[mask_d1, "rt_actual"] = 9999.0

        modified = _integrate_sgdfnet(df_modified.copy(), allow_fallback=True, mode="FULL_DAY")

        bd_val = baseline.loc[idx, "sgdfnet_residual_mean_7d"]
        mod_val = modified.loc[idx, "sgdfnet_residual_mean_7d"]
        assert abs(bd_val - mod_val) < 0.01, (
            f"sgdfnet_residual_mean_7d changed from {bd_val:.2f} to {mod_val:.2f} "
            f"after modifying same-day hour 1 — LEAKAGE!"
        )

    def test_sgdfnet_residual_lag_1h_is_zero_in_full_day(self):
        """sgdfnet_residual_lag_1h must be 0 in FULL_DAY mode."""
        df = _make_hourly_df(n_days=30)
        result = _integrate_sgdfnet(df.copy(), allow_fallback=True, mode="FULL_DAY")
        assert (result["sgdfnet_residual_lag_1h"] == 0.0).all(), (
            "sgdfnet_residual_lag_1h should be 0 in FULL_DAY mode"
        )

    def test_rt_lag_24h_is_d_minus_1_same_hour(self):
        """rt_lag_24h equals D-1's same-hour rt_actual."""
        df = _make_hourly_df(n_days=30)
        result = _add_lag_features(df.copy(), mode="FULL_DAY")

        # For any hour after the first day, rt_lag_24h == rt_actual from 24 rows ago
        # (which is D-1 same hour_business)
        np.testing.assert_array_almost_equal(
            result["rt_lag_24h"].iloc[24:].values,
            result["rt_actual"].iloc[:-24].values,
        )

    def test_check_leakage_on_residual_features(self):
        """check_leakage should handle residual-derived features."""
        df = _make_hourly_df(n_days=30)
        result = _integrate_sgdfnet(df.copy(), allow_fallback=True, mode="FULL_DAY")
        # The check should pass (residual features use D-1 data only in FULL_DAY)
        ok = check_leakage(result)
        assert ok, "Leakage check failed on residual features"


class TestFullDayDayLagAccuracy:
    """FULL_DAY day-level lag accuracy."""

    def test_previous_day_delta_mean_24h_is_d_minus_1_daily_mean(self):
        """previous_day_delta_mean_24h should be D-1's delta mean."""
        df = _make_hourly_df(n_days=30)
        df["_delta"] = df["rt_actual"] - df["da_anchor"]

        # Compute expected D-1 daily delta mean manually
        daily_delta_mean = df.groupby("business_day")["_delta"].mean().shift(1)

        result = _add_lag_features(df.copy(), mode="FULL_DAY")
        df = df.drop(columns=["_delta"])

        # Check a specific day
        day_idx = df["business_day"].unique()[10]
        mask = result["business_day"] == day_idx
        expected = daily_delta_mean.loc[day_idx]
        actual = result.loc[mask, "previous_day_delta_mean_24h"].iloc[0]
        assert abs(actual - expected) < 0.1 or (pd.isna(actual) and pd.isna(expected)), (
            f"previous_day_delta_mean_24h mismatch: expected {expected}, got {actual}"
        )


class TestIntradayModeLeakage:
    """INTRADAY mode explicitly allows same-day earlier actuals."""

    def test_intraday_has_same_day_actuals(self):
        """INTRADAY mode should include same-day rt_lag features."""
        df = _make_hourly_df(n_days=30)
        result_fullday = _add_lag_features(df.copy(), mode="FULL_DAY")
        result_intraday = _add_lag_features(df.copy(), mode="INTRADAY")

        # INTRADAY should have different rt_lag_1h (non-zero) vs FULL_DAY (zero)
        assert not (result_intraday["rt_lag_1h"] == 0.0).all()
        assert (result_fullday["rt_lag_1h"] == 0.0).all()

    def test_sgdfnet_residual_lag_1h_nonzero_in_intraday(self):
        """sgdfnet_residual_lag_1h should be non-zero in INTRADAY mode."""
        df = _make_hourly_df(n_days=30)
        result = _integrate_sgdfnet(df.copy(), allow_fallback=True, mode="INTRADAY")
        assert not (result["sgdfnet_residual_lag_1h"] == 0.0).all(), (
            "sgdfnet_residual_lag_1h should be non-zero in INTRADAY mode"
        )

    def test_intraday_requires_explicit_mode(self):
        """Same-day features only available when mode=INTRADAY is set."""
        df = _make_hourly_df(n_days=30)
        result = build_realtime_features(
            df, sgdfnet_pred_df=None, mode="FULL_DAY", allow_sgdfnet_fallback=True,
        )
        # FULL_DAY should have sgdfnet_residual_lag_1h = 0
        assert (result["sgdfnet_residual_lag_1h"] == 0.0).all()
