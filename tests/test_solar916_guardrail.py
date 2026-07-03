"""Tests for Solar916 guardrail module — Phase 8."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.solar916_guardrail import (
    Solar916GuardrailConfig,
    apply_guardrail,
    compute_guarded_metrics,
)


def _make_predictions(n: int = 24, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic predictions DataFrame for testing."""
    rng = np.random.RandomState(seed)
    hours = list(range(9, 17)) * (n // 8)
    df = pd.DataFrame({
        "business_day": pd.Timestamp("2026-02-01"),
        "hour_business": hours[:n],
        "ds": pd.Timestamp("2026-02-01"),
        "period": "9_16",
        "rt_actual": rng.uniform(50, 200, n),
        "da_anchor": rng.uniform(50, 200, n),
        "sgdfnet_pred": rng.uniform(50, 200, n),
        "solar916_residual_pred": rng.uniform(-30, 30, n),
        "feature_missing_flag": False,
    })
    return df


class TestApplyGuardrail:
    """Tests for apply_guardrail()."""

    def test_basic_output_columns(self):
        """Guardrail adds expected columns."""
        df = _make_predictions()
        result = apply_guardrail(df)
        for col in [
            "solar916_raw_residual_pred",
            "solar916_guardrail_weight",
            "solar916_residual_pred_after_guardrail",
            "solar916_corrected_pred",
            "guardrail_reason",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_raw_pred_preserved(self):
        """Raw residual prediction is stored unchanged."""
        df = _make_predictions()
        result = apply_guardrail(df)
        np.testing.assert_array_equal(
            result["solar916_raw_residual_pred"].values,
            df["solar916_residual_pred"].values,
        )

    def test_no_guardrail_by_default(self):
        """With default config, weight=1.0 for all 9_16 rows."""
        df = _make_predictions()
        result = apply_guardrail(df)
        np.testing.assert_array_equal(
            result["solar916_guardrail_weight"].values,
            np.ones(len(df)),
        )

    def test_disabled_hours_zero_weight(self):
        """Disabled hours get weight=0."""
        df = _make_predictions()
        config = Solar916GuardrailConfig(disabled_hours=[9, 11])
        result = apply_guardrail(df, config)
        for hour in [9, 11]:
            mask = result["hour_business"] == hour
            assert (result.loc[mask, "solar916_guardrail_weight"] == 0.0).all()

    def test_reduced_weight_hours(self):
        """Reduced weight hours get the configured weight."""
        df = _make_predictions()
        config = Solar916GuardrailConfig(reduced_weight_hours={10: 0.3})
        result = apply_guardrail(df, config)
        mask = result["hour_business"] == 10
        np.testing.assert_allclose(
            result.loc[mask, "solar916_guardrail_weight"].values, 0.3
        )

    def test_negative_risk_weight(self):
        """Negative da_anchor triggers negative risk weight reduction."""
        df = _make_predictions()
        df.loc[0, "da_anchor"] = -10.0  # negative price risk
        config = Solar916GuardrailConfig(
            negative_risk_threshold=0.0,
            negative_risk_weight=0.3,
        )
        result = apply_guardrail(df, config)
        assert result.iloc[0]["solar916_guardrail_weight"] == pytest.approx(0.3)

    def test_missing_feature_weight(self):
        """Feature missing flag reduces weight."""
        df = _make_predictions()
        df.loc[0, "feature_missing_flag"] = True
        config = Solar916GuardrailConfig(missing_feature_weight=0.5)
        result = apply_guardrail(df, config)
        assert result.iloc[0]["solar916_guardrail_weight"] == pytest.approx(0.5)

    def test_max_abs_correction_clip(self):
        """Correction is clipped to max_abs_correction."""
        df = _make_predictions()
        df["solar916_residual_pred"] = 500.0  # very large correction
        config = Solar916GuardrailConfig(max_abs_correction=50.0)
        result = apply_guardrail(df, config)
        assert (result["solar916_residual_pred_after_guardrail"].abs() <= 50.0).all()

    def test_corrected_pred_formula(self):
        """Corrected pred = sgdfnet_pred + guarded residual."""
        df = _make_predictions()
        result = apply_guardrail(df)
        expected = result["sgdfnet_pred"].values + result["solar916_residual_pred_after_guardrail"].values
        np.testing.assert_allclose(result["solar916_corrected_pred"].values, expected)

    def test_non_916_period_zero_weight(self):
        """Non-9_16 period rows get weight=0."""
        df = _make_predictions()
        df.loc[0, "period"] = "1_8"
        result = apply_guardrail(df)
        assert result.iloc[0]["solar916_guardrail_weight"] == 0.0


class TestComputeGuardedMetrics:
    """Tests for compute_guarded_metrics()."""

    def test_metrics_structure(self):
        """Metrics dict has expected keys."""
        df = _make_predictions()
        result = apply_guardrail(df)
        metrics = compute_guarded_metrics(result)
        assert "overall" in metrics
        assert "hourly" in metrics
        assert "buckets" in metrics
        assert "baseline_smape" in metrics["overall"]
        assert "corrected_smape" in metrics["overall"]
        assert "improvement" in metrics["overall"]

    def test_hourly_metrics_have_all_hours(self):
        """Hourly metrics cover all present hours."""
        df = _make_predictions()
        result = apply_guardrail(df)
        metrics = compute_guarded_metrics(result)
        hours = [h["hour"] for h in metrics["hourly"]]
        assert set(hours) == {9, 10, 11, 12, 13, 14, 15, 16}

    def test_guardrail_improves_negative_bucket(self):
        """Guardrail with negative risk weight should limit negative bucket damage."""
        df = _make_predictions(n=48, seed=123)
        # Make some negative actuals
        df.loc[:5, "rt_actual"] = -20.0
        df.loc[:5, "da_anchor"] = -15.0
        df.loc[:5, "solar916_residual_pred"] = 50.0  # bad correction for negative

        # Without guardrail
        result_no_gr = apply_guardrail(df)
        metrics_no_gr = compute_guarded_metrics(result_no_gr)

        # With guardrail reducing weight for negative risk
        config = Solar916GuardrailConfig(
            negative_risk_threshold=0.0,
            negative_risk_weight=0.0,  # fully disable for negative risk
        )
        result_gr = apply_guardrail(df, config)
        metrics_gr = compute_guarded_metrics(result_gr)

        # Guarded negative bucket should be at least as good as unguarded
        neg_no_gr = next(b for b in metrics_no_gr["buckets"] if b["bucket"] == "negative")
        neg_gr = next(b for b in metrics_gr["buckets"] if b["bucket"] == "negative")
        # With weight=0, corrected = sgdfnet_pred (no correction), so no damage
        assert neg_gr["corrected_smape"] <= neg_no_gr["corrected_smape"] + 1.0
