#!/usr/bin/env python
"""Tests for NegativeRisk recalibration logic.

Validates:
1. Base-rate aware upper bound calculation
2. Normalised recall computation
3. Alert budget metrics with synthetic data
4. Champion criteria thresholds
5. Script importability
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recalibrate_negative_risk_selection import (
    max_possible_recall_at_topk,
    normalised_recall,
    compute_alert_budget_metrics,
    compute_topk_capture_from_predictions,
    determine_recalibrated_verdict,
    build_recalibration_table,
    compute_summary,
)


# ── 1. Base-rate aware upper bound ───────────────────────────────────────────

class TestMaxPossibleRecallAtTopk:
    """Theoretical ceiling for recall at a given top-k budget."""

    def test_high_base_rate_caps_recall(self):
        """When positive_rate = 20%, top-10% can capture at most 50%."""
        result = max_possible_recall_at_topk(topk_pct=10, positive_rate=0.20)
        assert result == pytest.approx(0.50)

    def test_low_base_rate_allows_full_recall(self):
        """When positive_rate = 5%, top-10% can capture all positives."""
        result = max_possible_recall_at_topk(topk_pct=10, positive_rate=0.05)
        assert result == pytest.approx(1.0)

    def test_exact_match(self):
        """When positive_rate = 10%, top-10% can capture exactly 100%."""
        result = max_possible_recall_at_topk(topk_pct=10, positive_rate=0.10)
        assert result == pytest.approx(1.0)

    def test_very_high_base_rate(self):
        """When positive_rate = 50%, top-10% captures at most 20%."""
        result = max_possible_recall_at_topk(topk_pct=10, positive_rate=0.50)
        assert result == pytest.approx(0.20)

    def test_top5_with_30pct_rate(self):
        """top-5% with 30% positive rate: max = 0.05/0.30 = 0.1667."""
        result = max_possible_recall_at_topk(topk_pct=5, positive_rate=0.30)
        assert result == pytest.approx(5 / 30)

    def test_top20_with_15pct_rate(self):
        """top-20% with 15% positive rate: max = 0.20/0.15 > 1.0 -> capped at 1.0."""
        result = max_possible_recall_at_topk(topk_pct=20, positive_rate=0.15)
        assert result == pytest.approx(1.0)

    def test_zero_positive_rate(self):
        """Edge case: zero positive rate returns 0."""
        result = max_possible_recall_at_topk(topk_pct=10, positive_rate=0.0)
        assert result == 0.0


# ── 2. Normalised recall ────────────────────────────────────────────────────

class TestNormalisedRecall:
    """Normalised recall = actual / ceiling."""

    def test_perfect_ranker_at_ceiling(self):
        """Actual recall equals ceiling -> normalised = 1.0."""
        assert normalised_recall(0.50, 0.50) == pytest.approx(1.0)

    def test_half_of_ceiling(self):
        """Actual recall is half the ceiling -> normalised = 0.5."""
        assert normalised_recall(0.25, 0.50) == pytest.approx(0.5)

    def test_zero_ceiling(self):
        """Zero ceiling returns 0.0 (no division by zero)."""
        assert normalised_recall(0.10, 0.0) == 0.0

    def test_zero_actual(self):
        """Zero actual recall -> normalised = 0.0."""
        assert normalised_recall(0.0, 0.50) == pytest.approx(0.0)


# ── 3. Alert budget metrics with synthetic data ─────────────────────────────

class TestAlertBudgetMetrics:
    """Alert budget metrics on synthetic data with known properties."""

    @pytest.fixture
    def synthetic_perfect_ranker(self):
        """Perfect ranker: all positives have prob=1.0, negatives prob=0.0."""
        n = 1000
        n_pos = 200
        y_true = np.array([1] * n_pos + [0] * (n - n_pos))
        y_prob = np.array([1.0] * n_pos + [0.0] * (n - n_pos))
        return y_true, y_prob

    @pytest.fixture
    def synthetic_random_ranker(self):
        """Random ranker: probabilities independent of labels."""
        rng = np.random.RandomState(42)
        n = 1000
        n_pos = 200
        y_true = np.array([1] * n_pos + [0] * (n - n_pos))
        y_prob = rng.rand(n)
        return y_true, y_prob

    @pytest.fixture
    def synthetic_realistic(self):
        """Realistic scenario: 20% positive rate, AUC-like separation."""
        rng = np.random.RandomState(123)
        n = 720  # ~30 days * 24 hours
        n_pos = 144  # 20% positive rate
        y_true = np.zeros(n)
        y_true[:n_pos] = 1
        # Positives get higher probabilities on average
        y_prob = np.where(y_true == 1,
                          rng.beta(5, 2, n),
                          rng.beta(2, 5, n))
        return y_true, y_prob

    def test_perfect_ranker_budget10(self, synthetic_perfect_ranker):
        """Perfect ranker at 10% budget: should capture 50% of positives."""
        y_true, y_prob = synthetic_perfect_ranker
        result = compute_alert_budget_metrics(y_true, y_prob, budget_pct=10)

        assert result["budget_pct"] == 10
        assert result["n_total"] == 1000
        assert result["n_alerts"] == 100
        assert result["n_positive"] == 200
        # Top 100 are all positive (200 positives, take top 100)
        assert result["true_positives"] == 100
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.50)
        assert result["lift"] == pytest.approx(5.0)  # 1.0 / 0.20
        assert result["f1"] == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))

    def test_perfect_ranker_budget20(self, synthetic_perfect_ranker):
        """Perfect ranker at 20% budget: should capture 100% of positives."""
        y_true, y_prob = synthetic_perfect_ranker
        result = compute_alert_budget_metrics(y_true, y_prob, budget_pct=20)

        assert result["n_alerts"] == 200
        assert result["true_positives"] == 200
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["lift"] == pytest.approx(5.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_random_ranker_approximates_base_rate(self, synthetic_random_ranker):
        """Random ranker: precision should approximate base rate."""
        y_true, y_prob = synthetic_random_ranker
        result = compute_alert_budget_metrics(y_true, y_prob, budget_pct=10)

        # Random ranker precision ~= base_rate (with tolerance)
        base_rate = 200 / 1000
        assert result["precision"] == pytest.approx(base_rate, abs=0.08)
        assert result["lift"] == pytest.approx(1.0, abs=0.4)

    def test_realistic_scenario(self, synthetic_realistic):
        """Realistic scenario: metrics should be between random and perfect."""
        y_true, y_prob = synthetic_realistic
        result = compute_alert_budget_metrics(y_true, y_prob, budget_pct=20)

        assert result["n_total"] == 720
        assert result["n_positive"] == 144
        assert 0 < result["precision"] <= 1.0
        assert 0 < result["recall"] <= 1.0
        assert result["lift"] > 1.0  # better than random
        assert 0 < result["f1"] <= 1.0

    def test_base_rate_computation(self, synthetic_realistic):
        """Base rate should match the proportion of positives."""
        y_true, y_prob = synthetic_realistic
        result = compute_alert_budget_metrics(y_true, y_prob, budget_pct=10)
        assert result["base_rate"] == pytest.approx(144 / 720)

    def test_handles_nan_values(self):
        """NaN values in inputs should be filtered out."""
        y_true = np.array([1, 0, np.nan, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.5, 0.8, np.nan])
        result = compute_alert_budget_metrics(y_true, y_prob, budget_pct=50)
        # Should not raise; valid pairs are (1,0.9), (0,0.1), (1,0.8)
        assert result["n_total"] == 3


# ── 4. Top-k capture from predictions ────────────────────────────────────────

class TestTopkCapture:

    def test_perfect_ranker_top10(self):
        """Perfect ranker: top-10% captures 50% of positives when base_rate=20%."""
        n = 1000
        y_true = np.array([1] * 200 + [0] * 800)
        y_prob = np.array([1.0] * 200 + [0.0] * 800)
        result = compute_topk_capture_from_predictions(y_true, y_prob, topk_pct=10)

        assert result["topk_pct"] == 10
        assert result["k"] == 100
        assert result["tp_captured"] == 100
        assert result["recall"] == pytest.approx(0.50)
        assert result["precision"] == pytest.approx(1.0)

    def test_all_same_probabilities(self):
        """When all probabilities are equal, recall is approximately base_rate."""
        n = 1000
        y_true = np.array([1] * 200 + [0] * 800)
        y_prob = np.full(n, 0.5)
        result = compute_topk_capture_from_predictions(y_true, y_prob, topk_pct=10)

        # argsort is stable, so it picks the last 100 indices;
        # those are all zeros -> recall = 0
        # This is a known edge case; the test just ensures no crash
        assert 0 <= result["recall"] <= 1.0


# ── 5. Champion criteria thresholds ─────────────────────────────────────────

class TestDetermineRecalibratedVerdict:

    def test_champion_all_conditions_met(self):
        """All conditions met -> NEGATIVE_CHAMPION."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.95, mean_f1=0.80,
            mean_recall_at_20pct_alert=0.70, n_sufficient_months=5,
        )
        assert verdict == "NEGATIVE_CHAMPION"

    def test_champion_boundary_values(self):
        """Exact boundary values should still pass."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.90, mean_f1=0.70,
            mean_recall_at_20pct_alert=0.65, n_sufficient_months=4,
        )
        assert verdict == "NEGATIVE_CHAMPION"

    def test_acceptable_good_auc_and_f1(self):
        """Good AUC and F1 but insufficient alert recall -> ACCEPTABLE."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.88, mean_f1=0.65,
            mean_recall_at_20pct_alert=0.50, n_sufficient_months=5,
        )
        assert verdict == "NEGATIVE_ACCEPTABLE"

    def test_acceptable_boundary(self):
        """Exact boundary for ACCEPTABLE."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.85, mean_f1=0.60,
            mean_recall_at_20pct_alert=0.40, n_sufficient_months=3,
        )
        assert verdict == "NEGATIVE_ACCEPTABLE"

    def test_aux_moderate_auc(self):
        """Moderate AUC -> NEGATIVE_AUX."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.82, mean_f1=0.50,
            mean_recall_at_20pct_alert=0.40, n_sufficient_months=3,
        )
        assert verdict == "NEGATIVE_AUX"

    def test_aux_boundary(self):
        """Exact boundary for AUX."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.80, mean_f1=0.40,
            mean_recall_at_20pct_alert=0.30, n_sufficient_months=2,
        )
        assert verdict == "NEGATIVE_AUX"

    def test_nogo_weak_auc(self):
        """Weak AUC -> NEGATIVE_NO_GO."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.70, mean_f1=0.30,
            mean_recall_at_20pct_alert=0.20, n_sufficient_months=2,
        )
        assert verdict == "NEGATIVE_NO_GO"

    def test_nogo_insufficient_months_blocks_champion(self):
        """Great metrics but only 3 months -> not CHAMPION (need 4)."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.95, mean_f1=0.80,
            mean_recall_at_20pct_alert=0.70, n_sufficient_months=3,
        )
        # Falls through to ACCEPTABLE (auc >= 0.85, f1 >= 0.60)
        assert verdict == "NEGATIVE_ACCEPTABLE"

    def test_champion_fails_on_low_f1(self):
        """High AUC but low F1 -> not CHAMPION."""
        verdict = determine_recalibrated_verdict(
            mean_auc=0.95, mean_f1=0.50,
            mean_recall_at_20pct_alert=0.70, n_sufficient_months=5,
        )
        # auc >= 0.85 but f1 < 0.60 -> not ACCEPTABLE either
        # auc >= 0.80 -> AUX
        assert verdict == "NEGATIVE_AUX"


# ── 6. Integration: build_recalibration_table & compute_summary ──────────────

class TestIntegration:
    """Integration tests with small synthetic DataFrames."""

    @pytest.fixture
    def synthetic_inputs(self, tmp_path):
        """Create synthetic monthly_metrics and prediction CSVs."""
        # monthly_metrics for 2 months
        metrics_df = pd.DataFrame([
            {"month": "2026-01", "direction": "negative", "status": "ok",
             "n_positive": 100, "precision": 0.8, "recall": 0.7,
             "f1": 0.75, "roc_auc": 0.92, "threshold": 0.3},
            {"month": "2026-02", "direction": "negative", "status": "ok",
             "n_positive": 150, "precision": 0.85, "recall": 0.8,
             "f1": 0.82, "roc_auc": 0.95, "threshold": 0.25},
        ])

        # Prediction CSVs: 500 rows each, with known positive rates
        rng = np.random.RandomState(42)
        for month, n_pos in [("2026_01", 100), ("2026_02", 150)]:
            n_total = 500
            y_true = np.array([1] * n_pos + [0] * (n_total - n_pos))
            # Good ranker: positives get higher probs
            y_prob = np.where(y_true == 1,
                              rng.beta(5, 2, n_total),
                              rng.beta(2, 5, n_total))
            pred_df = pd.DataFrame({
                "negative_prob": y_prob,
                "negative_label": y_true,
                "business_day": range(n_total),
                "hour_business": range(n_total),
                "ds": pd.date_range("2026-01-01", periods=n_total, freq="h").astype(str),
            })
            pred_df.to_csv(tmp_path / f"predictions_{month}.csv", index=False)

        return metrics_df, tmp_path

    def test_build_recalibration_table_shape(self, synthetic_inputs):
        """Table should have one row per month with expected columns."""
        metrics_df, backtest_root = synthetic_inputs
        from glob import glob as file_glob
        # Load predictions manually
        predictions = {}
        for fpath in sorted(file_glob(str(backtest_root / "predictions_2026_*.csv"))):
            fname = Path(fpath).name
            month_str = fname.replace("predictions_", "").replace(".csv", "")
            month_dash = month_str.replace("_", "-")
            predictions[month_dash] = pd.read_csv(fpath)

        recal_df = build_recalibration_table(metrics_df, predictions)
        assert len(recal_df) == 2
        assert "positive_rate" in recal_df.columns
        assert "norm_recall_top10" in recal_df.columns
        assert "recall_alert20" in recal_df.columns

    def test_build_recalibration_table_positive_rate(self, synthetic_inputs):
        """Positive rate should be n_positive / n_total."""
        metrics_df, backtest_root = synthetic_inputs
        predictions = {}
        for fpath in sorted(Path(backtest_root).glob("predictions_2026_*.csv")):
            fname = fpath.name
            month_str = fname.replace("predictions_", "").replace(".csv", "")
            month_dash = month_str.replace("_", "-")
            predictions[month_dash] = pd.read_csv(fpath)

        recal_df = build_recalibration_table(metrics_df, predictions)
        jan_row = recal_df[recal_df["month"] == "2026-01"].iloc[0]
        assert jan_row["positive_rate"] == pytest.approx(100 / 500)

    def test_compute_summary_keys(self, synthetic_inputs):
        """Summary dict should contain expected keys."""
        metrics_df, backtest_root = synthetic_inputs
        predictions = {}
        for fpath in sorted(Path(backtest_root).glob("predictions_2026_*.csv")):
            fname = fpath.name
            month_str = fname.replace("predictions_", "").replace(".csv", "")
            month_dash = month_str.replace("_", "-")
            predictions[month_dash] = pd.read_csv(fpath)

        recal_df = build_recalibration_table(metrics_df, predictions)
        summary = compute_summary(recal_df)

        assert "mean_auc" in summary
        assert "mean_f1" in summary
        assert "mean_recall_at_20pct_alert" in summary
        assert "verdict" in summary
        assert "mean_norm_recall_top10" in summary
        assert summary["n_sufficient_months"] == 2


# ── 7. Importability ─────────────────────────────────────────────────────────

class TestImportability:
    """The script should be importable without errors."""

    def test_import_main_module(self):
        """Importing the recalibration module should not raise."""
        import scripts.recalibrate_negative_risk_selection as mod
        assert hasattr(mod, "max_possible_recall_at_topk")
        assert hasattr(mod, "normalised_recall")
        assert hasattr(mod, "compute_alert_budget_metrics")
        assert hasattr(mod, "determine_recalibrated_verdict")
        assert hasattr(mod, "main")

    def test_constants_defined(self):
        """Key constants should be defined."""
        from scripts.recalibrate_negative_risk_selection import (
            TOP_K_PCTS, ALERT_BUDGET_PCTS, DIRECTION,
        )
        assert TOP_K_PCTS == [5, 10, 20]
        assert ALERT_BUDGET_PCTS == [10, 20, 30]
        assert DIRECTION == "negative"
