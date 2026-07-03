"""Tests for IntradayTrackerPolicy — Phase 10."""
from __future__ import annotations

import pandas as pd
import pytest

from models.deep_sgdf_delta.intraday_residual_tracker import (
    IntradayResidualState,
    compute_intraday_residual_state,
)
from models.deep_sgdf_delta.intraday_tracker_policy import (
    PolicyConfig,
    PolicyDecision,
    PolicyResult,
    evaluate_policy,
)
from models.deep_sgdf_delta.prediction_modes import PredictionMode


def _make_state(
    n_observed: int = 5,
    cutoff_hour: int = 12,
    confidence: float = 0.6,
    residual_std: float = 50.0,
    bias_direction: str = "positive",
    mean_residual: float = 30.0,
) -> IntradayResidualState:
    """Create a synthetic IntradayResidualState for testing."""
    return IntradayResidualState(
        business_day=pd.Timestamp("2026-02-15"),
        cutoff_hour=cutoff_hour,
        n_observed=n_observed,
        mean_residual_today=mean_residual,
        median_residual_today=mean_residual * 0.9,
        ewm_residual_today=mean_residual * 1.1,
        last_residual=mean_residual * 0.8,
        residual_std_today=residual_std,
        bias_direction=bias_direction,
        confidence=confidence,
        observed_hours=list(range(9, 9 + n_observed)),
        observed_residuals=[mean_residual] * n_observed,
    )


class TestEvaluatePolicy:
    """Tests for evaluate_policy()."""

    def test_full_day_mode_disabled(self):
        """FULL_DAY mode → DISABLED."""
        state = _make_state()
        result = evaluate_policy(state, PredictionMode.FULL_DAY)
        assert result.policy_decision == PolicyDecision.DISABLED
        assert result.fusion_weight == 0.0
        assert result.reason == "mode_not_intraday"

    def test_insufficient_observations_disabled(self):
        """n_observed < 3 → DISABLED."""
        state = _make_state(n_observed=2)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.DISABLED
        assert "n_observed" in result.reason

    def test_cutoff_below_shadow_threshold_disabled(self):
        """cutoff < 10 → DISABLED."""
        state = _make_state(cutoff_hour=9)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.DISABLED
        assert "below_min_shadow" in result.reason

    def test_cutoff_below_correction_threshold_shadow(self):
        """cutoff 10-11 → SHADOW_ONLY."""
        state = _make_state(cutoff_hour=11)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.SHADOW_ONLY
        assert result.shadow_only_flag is True
        assert result.fusion_weight == 0.0

    def test_low_confidence_shadow(self):
        """confidence < 0.35 → SHADOW_ONLY."""
        state = _make_state(confidence=0.30)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.SHADOW_ONLY
        assert "confidence" in result.reason

    def test_high_residual_std_shadow(self):
        """residual_std > 180 → SHADOW_ONLY."""
        state = _make_state(residual_std=200.0)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.SHADOW_ONLY
        assert "residual_std" in result.reason

    def test_negative_risk_low_weight(self):
        """Negative price risk → LOW_WEIGHT with reduced weight."""
        state = _make_state()
        result = evaluate_policy(state, PredictionMode.INTRADAY, has_negative_risk=True)
        assert result.policy_decision == PolicyDecision.LOW_WEIGHT
        assert result.fusion_weight == 0.08  # negative_risk_weight
        assert result.reason == "negative_price_risk"

    def test_high_weight_conditions(self):
        """High confidence + cutoff >= 14 → HIGH_WEIGHT."""
        state = _make_state(confidence=0.60, cutoff_hour=14)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.HIGH_WEIGHT
        assert result.fusion_weight == 0.22

    def test_default_low_weight(self):
        """Default case → LOW_WEIGHT."""
        state = _make_state(confidence=0.45, cutoff_hour=12)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.LOW_WEIGHT
        assert result.fusion_weight == 0.12

    def test_cutoff_13_low_weight(self):
        """Cutoff 13 with decent confidence → LOW_WEIGHT (not high)."""
        state = _make_state(confidence=0.60, cutoff_hour=13)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.LOW_WEIGHT

    def test_custom_config(self):
        """Custom config overrides defaults."""
        state = _make_state(confidence=0.40, cutoff_hour=12)
        config = PolicyConfig(min_confidence_high=0.35, fusion_weight_high=0.30)
        result = evaluate_policy(state, PredictionMode.INTRADAY, config=config)
        # With custom min_confidence_high=0.35 and cutoff=12 (< 14), still LOW_WEIGHT
        assert result.policy_decision == PolicyDecision.LOW_WEIGHT

    def test_shadow_only_flag_false_when_disabled(self):
        """DISABLED decisions have shadow_only_flag=False."""
        state = _make_state(n_observed=1)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.shadow_only_flag is False

    def test_boundary_cutoff_12_with_good_confidence(self):
        """Cutoff exactly 12 with confidence > 0.35 → LOW_WEIGHT."""
        state = _make_state(cutoff_hour=12, confidence=0.50)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.LOW_WEIGHT
        assert result.fusion_weight > 0

    def test_boundary_cutoff_14_high_confidence(self):
        """Cutoff exactly 14 with confidence >= 0.55 → HIGH_WEIGHT."""
        state = _make_state(cutoff_hour=14, confidence=0.55)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.HIGH_WEIGHT

    def test_boundary_cutoff_14_low_confidence(self):
        """Cutoff 14 but confidence < 0.55 → LOW_WEIGHT."""
        state = _make_state(cutoff_hour=14, confidence=0.45)
        result = evaluate_policy(state, PredictionMode.INTRADAY)
        assert result.policy_decision == PolicyDecision.LOW_WEIGHT


class TestPolicyDecisionEnum:
    """Tests for PolicyDecision enum."""

    def test_enum_values(self):
        assert PolicyDecision.DISABLED.value == "DISABLED"
        assert PolicyDecision.SHADOW_ONLY.value == "SHADOW_ONLY"
        assert PolicyDecision.LOW_WEIGHT.value == "LOW_WEIGHT"
        assert PolicyDecision.HIGH_WEIGHT.value == "HIGH_WEIGHT"

    def test_enum_is_string(self):
        assert isinstance(PolicyDecision.DISABLED, str)
