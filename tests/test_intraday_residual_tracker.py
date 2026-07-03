"""Tests for IntradayResidualTracker — Phase 9."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.intraday_residual_tracker import (
    IntradayResidualState,
    IntradayTrackerConfig,
    apply_intraday_correction,
    compute_intraday_residual_state,
    predict_intraday_correction,
)


def _make_observed_data(
    business_day: str = "2026-02-15",
    hours: list[int] | None = None,
    residual_pattern: str = "positive",
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic observed data for testing."""
    if hours is None:
        hours = [9, 10, 11]
    rng = np.random.RandomState(seed)
    n = len(hours)
    sgdf_pred = rng.uniform(100, 300, n)
    if residual_pattern == "positive":
        residuals = rng.uniform(20, 80, n)
    elif residual_pattern == "negative":
        residuals = rng.uniform(-80, -20, n)
    elif residual_pattern == "mixed":
        residuals = rng.uniform(-50, 50, n)
    else:
        residuals = np.zeros(n)
    rt_actual = sgdf_pred + residuals
    return pd.DataFrame({
        "business_day": pd.Timestamp(business_day),
        "hour_business": hours,
        "sgdfnet_pred": sgdf_pred,
        "rt_actual": rt_actual,
        "da_anchor": sgdf_pred + rng.uniform(-10, 10, n),
    })


def _make_future_data(
    business_day: str = "2026-02-15",
    hours: list[int] | None = None,
    seed: int = 43,
) -> pd.DataFrame:
    """Create synthetic future hours for testing."""
    if hours is None:
        hours = [12, 13, 14, 15, 16]
    rng = np.random.RandomState(seed)
    n = len(hours)
    sgdf_pred = rng.uniform(100, 300, n)
    return pd.DataFrame({
        "business_day": pd.Timestamp(business_day),
        "hour_business": hours,
        "sgdfnet_pred": sgdf_pred,
        "da_anchor": sgdf_pred + rng.uniform(-10, 10, n),
    })


class TestComputeIntradayResidualState:
    """Tests for compute_intraday_residual_state()."""

    def test_basic_state_fields(self):
        """State has all required fields."""
        obs = _make_observed_data()
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        assert state.n_observed == 3
        assert state.mean_residual_today != 0
        assert state.median_residual_today != 0
        assert state.ewm_residual_today != 0
        assert state.last_residual != 0
        assert state.residual_std_today >= 0
        assert state.bias_direction in ("positive", "negative", "mixed", "insufficient")
        assert 0.0 <= state.confidence <= 1.0

    def test_empty_observations(self):
        """Empty observations return zero state."""
        obs = pd.DataFrame(columns=["hour_business", "sgdfnet_pred", "rt_actual"])
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        assert state.n_observed == 0
        assert state.confidence == 0.0
        assert state.bias_direction == "insufficient"

    def test_positive_bias_detected(self):
        """Positive residuals → positive bias direction."""
        obs = _make_observed_data(residual_pattern="positive")
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        assert state.bias_direction == "positive"
        assert state.mean_residual_today > 0

    def test_negative_bias_detected(self):
        """Negative residuals → negative bias direction."""
        obs = _make_observed_data(residual_pattern="negative")
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        assert state.bias_direction == "negative"
        assert state.mean_residual_today < 0

    def test_cutoff_filters_hours(self):
        """Only hours <= cutoff_hour are used."""
        obs = _make_observed_data(hours=[9, 10, 11, 12, 13])
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        assert state.n_observed == 3
        assert state.observed_hours == [9, 10, 11]

    def test_confidence_increases_with_more_data(self):
        """More observed hours → higher confidence."""
        obs3 = _make_observed_data(hours=[9, 10, 11])
        obs6 = _make_observed_data(hours=[9, 10, 11, 12, 13, 14])
        state3 = compute_intraday_residual_state(obs3, "2026-02-15", cutoff_hour=11)
        state6 = compute_intraday_residual_state(obs6, "2026-02-15", cutoff_hour=14)
        assert state6.confidence >= state3.confidence


class TestPredictIntradayCorrection:
    """Tests for predict_intraday_correction()."""

    def test_output_columns(self):
        """Output has expected columns."""
        obs = _make_observed_data()
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        future = _make_future_data()
        result = predict_intraday_correction(future, state)
        for col in ["intraday_raw_correction", "intraday_correction_weight",
                     "intraday_correction", "intraday_corrected_pred"]:
            assert col in result.columns

    def test_insufficient_observations_no_correction(self):
        """With < min_observed_hours, no correction applied."""
        obs = _make_observed_data(hours=[9])  # only 1 hour
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=9)
        future = _make_future_data()
        config = IntradayTrackerConfig(min_observed_hours=2)
        result = predict_intraday_correction(future, state, config)
        assert (result["intraday_correction"] == 0.0).all()

    def test_positive_bias_gives_positive_correction(self):
        """Positive bias → positive correction (adding to prediction)."""
        obs = _make_observed_data(residual_pattern="positive")
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        future = _make_future_data()
        result = predict_intraday_correction(future, state)
        assert (result["intraday_correction"] > 0).all()

    def test_correction_clipped(self):
        """Correction is clipped to max_abs_correction."""
        obs = _make_observed_data(hours=[9, 10, 11], residual_pattern="positive", seed=1)
        # Make residuals very large
        obs["rt_actual"] = obs["sgdfnet_pred"] + 500
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        future = _make_future_data()
        config = IntradayTrackerConfig(max_abs_correction=50.0)
        result = predict_intraday_correction(future, state, config)
        assert (result["intraday_correction"].abs() <= 50.0).all()

    def test_corrected_pred_formula(self):
        """corrected_pred = sgdfnet_pred + correction."""
        obs = _make_observed_data()
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        future = _make_future_data()
        result = predict_intraday_correction(future, state)
        expected = result["sgdfnet_pred"].values + result["intraday_correction"].values
        np.testing.assert_allclose(result["intraday_corrected_pred"].values, expected)

    def test_distance_decay(self):
        """Further hours from cutoff get smaller corrections."""
        obs = _make_observed_data()
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        future = _make_future_data(hours=[12, 13, 14, 15, 16])
        result = predict_intraday_correction(future, state)
        weights = result["intraday_correction_weight"].values
        # Weight should decrease with distance
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]


class TestApplyIntradayCorrection:
    """Tests for apply_intraday_correction()."""

    def test_guardrail_reason_column(self):
        """Output includes guardrail_reason."""
        obs = _make_observed_data()
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        future = _make_future_data()
        result = apply_intraday_correction(future, state)
        assert "guardrail_reason" in result.columns

    def test_negative_price_guardrail(self):
        """Negative da_anchor reduces correction weight."""
        obs = _make_observed_data()
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        future = _make_future_data()
        future.loc[0, "da_anchor"] = -10.0  # negative price risk
        config = IntradayTrackerConfig(negative_price_guardrail=True, negative_price_weight=0.3)
        result = apply_intraday_correction(future, state, config)
        assert result.iloc[0]["guardrail_reason"] == "negative_price_risk"
        assert result.iloc[0]["intraday_correction_weight"] < 1.0

    def test_past_hours_not_corrected(self):
        """Hours <= cutoff_hour get zero correction."""
        obs = _make_observed_data()
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=11)
        # Include some past hours in future_rows
        mixed = pd.concat([
            _make_future_data(hours=[10, 11, 12, 13]),  # 10, 11 are past
        ], ignore_index=True)
        result = apply_intraday_correction(mixed, state)
        past_mask = result["hour_business"] <= 11
        assert (result.loc[past_mask, "intraday_correction"] == 0.0).all()

    def test_low_confidence_no_correction(self):
        """Very low confidence → no correction."""
        obs = _make_observed_data(hours=[9])  # only 1 hour → low confidence
        state = compute_intraday_residual_state(obs, "2026-02-15", cutoff_hour=9)
        future = _make_future_data()
        config = IntradayTrackerConfig(min_observed_hours=2, min_confidence=0.5)
        result = apply_intraday_correction(future, state, config)
        assert (result["intraday_correction"] == 0.0).all()
