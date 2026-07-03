"""Tests for models.deep_sgdf_delta.metrics — evaluation metrics.

Covers:
  - smape_floor50: perfect predictions, known input/output pairs
  - compute_period_mask: correct hour ranges for 1_8, 9_16, 17_24
  - compute_full_metrics: correct structure and key presence
  - compute_monthly_metrics: correct grouping by month
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from models.deep_sgdf_delta.metrics import (
    smape_floor50,
    compute_period_mask,
    compute_full_metrics,
    compute_monthly_metrics,
)


# ── smape_floor50 ─────────────────────────────────────────────────────

class TestSmapeFloor50:
    def test_smape_floor50_perfect(self):
        """Perfect predictions -> 0."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = y_true.copy()
        result = smape_floor50(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_smape_floor50_known_values(self):
        """Known input/output pairs."""
        # y_true = [100, 200], y_pred = [110, 190]
        # After floor-50 capping (all values >= 50, so no change):
        # sMAPE = mean(200 * |110-100| / (100+110+eps), 200 * |190-200| / (200+190+eps))
        #       = mean(200 * 10 / 210, 200 * 10 / 390)
        #       = mean(9.5238, 5.1282)
        #       = 7.326
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        result = smape_floor50(y_true, y_pred)

        expected_1 = 200.0 * 10.0 / (100.0 + 110.0 + 1e-6)
        expected_2 = 200.0 * 10.0 / (200.0 + 190.0 + 1e-6)
        expected = (expected_1 + expected_2) / 2.0

        assert result == pytest.approx(expected, rel=1e-4)

    def test_smape_floor50_floor_effect(self):
        """Values below floor=50 are capped to 50."""
        y_true = np.array([10.0])  # below floor -> capped to 50
        y_pred = np.array([10.0])  # below floor -> capped to 50
        result = smape_floor50(y_true, y_pred)
        # Both capped to 50, so perfect match -> 0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_smape_floor50_asymmetric(self):
        """Asymmetric predictions produce positive sMAPE."""
        y_true = np.array([300.0, 300.0, 300.0])
        y_pred = np.array([350.0, 250.0, 300.0])
        result = smape_floor50(y_true, y_pred)
        assert result > 0.0

    def test_smape_floor50_non_negative(self):
        """sMAPE is always non-negative."""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 300
        y_pred = np.random.randn(100) * 100 + 300
        result = smape_floor50(y_true, y_pred)
        assert result >= 0.0


# ── compute_period_mask ──────────────────────────────────────────────

class TestPeriodMasks:
    def test_period_masks_1_8(self):
        """Correct hour range for period '1_8'."""
        hours = np.arange(1, 25)
        mask = compute_period_mask(hours, "1_8")
        expected = np.array([True] * 8 + [False] * 16)
        np.testing.assert_array_equal(mask, expected)

    def test_period_masks_9_16(self):
        """Correct hour range for period '9_16'."""
        hours = np.arange(1, 25)
        mask = compute_period_mask(hours, "9_16")
        expected = np.array([False] * 8 + [True] * 8 + [False] * 8)
        np.testing.assert_array_equal(mask, expected)

    def test_period_masks_17_24(self):
        """Correct hour range for period '17_24'."""
        hours = np.arange(1, 25)
        mask = compute_period_mask(hours, "17_24")
        expected = np.array([False] * 16 + [True] * 8)
        np.testing.assert_array_equal(mask, expected)

    def test_period_masks_cover_all_hours(self):
        """All three masks together cover all 24 hours exactly once."""
        hours = np.arange(1, 25)
        m1 = compute_period_mask(hours, "1_8")
        m2 = compute_period_mask(hours, "9_16")
        m3 = compute_period_mask(hours, "17_24")
        combined = m1 | m2 | m3
        assert combined.all()
        # No overlap
        assert (m1.astype(int) + m2.astype(int) + m3.astype(int) == 1).all()

    def test_period_masks_invalid_period(self):
        """Unknown period raises ValueError."""
        hours = np.arange(1, 25)
        with pytest.raises(ValueError, match="Unknown period"):
            compute_period_mask(hours, "invalid")


# ── compute_full_metrics ──────────────────────────────────────────────

class TestComputeFullMetrics:
    def _make_metrics_df(self, n_hours: int = 48) -> pd.DataFrame:
        """Create a synthetic prediction DataFrame for metrics testing."""
        np.random.seed(42)
        hours = np.tile(np.arange(1, 25), n_hours // 24)
        rt_actual = np.random.randn(n_hours) * 50 + 300
        rt_pred = rt_actual + np.random.randn(n_hours) * 10
        delta_target = rt_actual - 300.0
        delta_pred = rt_pred - 300.0
        da_anchor = np.full(n_hours, 300.0)

        return pd.DataFrame({
            "rt_actual": rt_actual,
            "rt_pred": rt_pred,
            "delta_target": delta_target,
            "delta_pred": delta_pred,
            "hour": hours,
            "da_anchor": da_anchor,
        })

    def test_compute_full_metrics(self):
        """Correct structure with all expected keys."""
        df = self._make_metrics_df()
        result = compute_full_metrics(df)

        # Check all expected keys
        expected_keys = {
            "overall_sMAPE_floor50",
            "delta_mae",
            "rows_total",
            "rows_missing",
            "1_8_sMAPE_floor50",
            "9_16_sMAPE_floor50",
            "17_24_sMAPE_floor50",
            "normal_sMAPE_floor50",
            "spike_sMAPE_floor50",
            "spike_count",
            "negative_sMAPE_floor50",
            "negative_count",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check types
        assert isinstance(result["overall_sMAPE_floor50"], float)
        assert isinstance(result["delta_mae"], float)
        assert isinstance(result["rows_total"], int)
        assert isinstance(result["rows_missing"], int)

        # Check values are sensible
        assert result["overall_sMAPE_floor50"] >= 0.0
        assert result["delta_mae"] >= 0.0
        assert result["rows_total"] == 48

    def test_compute_full_metrics_empty(self):
        """Empty DataFrame returns NaN metrics."""
        df = pd.DataFrame(columns=["rt_actual", "rt_pred", "delta_target", "delta_pred", "hour"])
        result = compute_full_metrics(df)
        assert result["rows_total"] == 0
        assert np.isnan(result["overall_sMAPE_floor50"])


# ── compute_monthly_metrics ──────────────────────────────────────────

class TestMonthlyMetrics:
    def test_monthly_metrics(self):
        """Correct grouping by month."""
        np.random.seed(42)
        n_days = 60
        rows = []
        for day_offset in range(n_days):
            bd = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_offset)
            for hour in range(1, 25):
                ds = bd + pd.Timedelta(hours=hour)
                rt_actual = 300.0 + np.random.randn() * 50
                rt_pred = rt_actual + np.random.randn() * 10
                rows.append({
                    "ds": ds,
                    "rt_actual": rt_actual,
                    "rt_pred": rt_pred,
                })

        df = pd.DataFrame(rows)
        result = compute_monthly_metrics(df)

        # Should have rows for each month present
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "month" in result.columns
        assert "sMAPE_floor50" in result.columns
        assert "count" in result.columns

        # All sMAPE values should be non-negative
        assert (result["sMAPE_floor50"] >= 0).all()

        # Total count should match input rows
        assert result["count"].sum() == len(df)

    def test_monthly_metrics_with_target_month(self):
        """Works with explicit target_month column."""
        np.random.seed(42)
        df = pd.DataFrame({
            "rt_actual": [300.0, 310.0, 320.0, 290.0],
            "rt_pred": [305.0, 308.0, 315.0, 295.0],
            "target_month": ["2024-01", "2024-01", "2024-02", "2024-02"],
        })
        result = compute_monthly_metrics(df)
        assert len(result) == 2
        assert set(result["month"].tolist()) == {"2024-01", "2024-02"}

    def test_monthly_metrics_empty(self):
        """Empty DataFrame returns empty result."""
        df = pd.DataFrame(columns=["rt_actual", "rt_pred"])
        result = compute_monthly_metrics(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
