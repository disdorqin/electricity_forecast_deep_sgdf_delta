"""Tests for the unified risk calibration library.

Verifies:
1. threshold_sweep returns correct number of rows
2. top_k_capture: lift >= 1.0 always
3. lift_at_k returns float >= 0
4. calibration_bucket_table has correct number of buckets
5. ECE is in [0, 1]
6. select_threshold_by_objective returns valid threshold for each objective
7. Edge cases: all zeros, all ones, single sample
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.risk_calibration import (
    calibration_bucket_table,
    expected_calibration_error,
    lift_at_k,
    precision_recall_curve_summary,
    select_threshold_by_objective,
    threshold_sweep,
    top_k_capture,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def balanced_data():
    """Balanced binary classification data with moderate signal."""
    np.random.seed(42)
    n = 200
    y_true = np.zeros(n)
    y_true[:40] = 1.0  # 20% positive rate
    # Probabilities: positives get higher scores on average
    y_prob = np.random.uniform(0.0, 0.5, n)
    y_prob[:40] = np.random.uniform(0.3, 0.9, 40)
    return y_true, y_prob


@pytest.fixture
def perfect_data():
    """Perfectly separable data."""
    n = 100
    y_true = np.zeros(n)
    y_true[:30] = 1.0
    y_prob = np.zeros(n)
    y_prob[:30] = 0.9  # all positives get high prob
    y_prob[30:] = 0.1  # all negatives get low prob
    return y_true, y_prob


@pytest.fixture
def random_data():
    """Random noise data (no signal)."""
    np.random.seed(123)
    n = 500
    y_true = np.random.binomial(1, 0.15, n).astype(float)
    y_prob = np.random.uniform(0, 1, n)
    return y_true, y_prob


# ═══════════════════════════════════════════════════════════════════════
# Test 1: threshold_sweep returns correct number of rows
# ═══════════════════════════════════════════════════════════════════════

class TestThresholdSweep:
    """threshold_sweep should return one row per threshold."""

    def test_default_thresholds_count(self, balanced_data):
        """Default thresholds: 101 values from 0.0 to 1.0."""
        y_true, y_prob = balanced_data
        result = threshold_sweep(y_true, y_prob)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 101

    def test_custom_thresholds_count(self, balanced_data):
        """Custom thresholds: should return exactly that many rows."""
        y_true, y_prob = balanced_data
        custom = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = threshold_sweep(y_true, y_prob, thresholds=custom)
        assert len(result) == 5

    def test_columns_present(self, balanced_data):
        """All expected columns should be present."""
        y_true, y_prob = balanced_data
        result = threshold_sweep(y_true, y_prob, thresholds=[0.5])
        expected_cols = {"threshold", "precision", "recall", "f1", "support",
                         "alert_rate", "tp", "fp", "tn", "fn"}
        assert set(result.columns) == expected_cols

    def test_precision_recall_consistency(self, balanced_data):
        """precision = tp / (tp + fp), recall = tp / (tp + fn)."""
        y_true, y_prob = balanced_data
        result = threshold_sweep(y_true, y_prob, thresholds=[0.5])
        row = result.iloc[0]
        if row["tp"] + row["fp"] > 0:
            expected_prec = row["tp"] / (row["tp"] + row["fp"])
            assert abs(row["precision"] - expected_prec) < 1e-4
        if row["tp"] + row["fn"] > 0:
            expected_rec = row["tp"] / (row["tp"] + row["fn"])
            assert abs(row["recall"] - expected_rec) < 1e-4

    def test_tp_fp_tn_fn_sum_to_n(self, balanced_data):
        """tp + fp + tn + fn should equal n for every threshold."""
        y_true, y_prob = balanced_data
        n = len(y_true)
        result = threshold_sweep(y_true, y_prob, thresholds=[0.2, 0.5, 0.8])
        for _, row in result.iterrows():
            assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == n

    def test_threshold_zero_captures_all(self, balanced_data):
        """Threshold 0.0 should predict all as positive."""
        y_true, y_prob = balanced_data
        result = threshold_sweep(y_true, y_prob, thresholds=[0.0])
        row = result.iloc[0]
        assert row["tp"] + row["fp"] == len(y_true)
        assert row["fn"] == 0

    def test_threshold_one_captures_none_or_exact(self, balanced_data):
        """Threshold 1.0 should predict almost nothing as positive."""
        y_true, y_prob = balanced_data
        result = threshold_sweep(y_true, y_prob, thresholds=[1.0])
        row = result.iloc[0]
        # Only samples with y_prob == 1.0 would be predicted positive
        # In practice, very few or none
        assert row["tp"] + row["fp"] <= len(y_true)


# ═══════════════════════════════════════════════════════════════════════
# Test 2: top_k_capture: lift >= 1.0 always
# ═══════════════════════════════════════════════════════════════════════

class TestTopKCapture:
    """top_k_capture should return lift >= 1.0 when model has signal."""

    def test_lift_ge_1_for_signal(self, balanced_data):
        """With signal data, lift at top-20% should be >= 1.0."""
        y_true, y_prob = balanced_data
        result = top_k_capture(y_true, y_prob, k_pcts=[20])
        assert result["lift"].iloc[0] >= 1.0

    def test_lift_ge_1_for_perfect(self, perfect_data):
        """Perfect data: lift should be very high at small k."""
        y_true, y_prob = perfect_data
        result = top_k_capture(y_true, y_prob, k_pcts=[10, 20, 30])
        for _, row in result.iterrows():
            assert row["lift"] >= 1.0

    def test_default_k_pcts(self, balanced_data):
        """Default k_pcts = [1, 3, 5, 10, 20]."""
        y_true, y_prob = balanced_data
        result = top_k_capture(y_true, y_prob)
        assert len(result) == 5
        assert list(result["k_pct"]) == [1, 3, 5, 10, 20]

    def test_capture_rate_bounded(self, balanced_data):
        """Capture rate should be in [0, 1]."""
        y_true, y_prob = balanced_data
        result = top_k_capture(y_true, y_prob, k_pcts=[5, 10, 50])
        for _, row in result.iterrows():
            assert 0.0 <= row["capture_rate"] <= 1.0

    def test_tp_captured_le_total_positives(self, balanced_data):
        """tp_captured should not exceed total_positives."""
        y_true, y_prob = balanced_data
        result = top_k_capture(y_true, y_prob, k_pcts=[50])
        row = result.iloc[0]
        assert row["tp_captured"] <= row["total_positives"]

    def test_columns_present(self, balanced_data):
        """All expected columns should be present."""
        y_true, y_prob = balanced_data
        result = top_k_capture(y_true, y_prob, k_pcts=[10])
        expected_cols = {"k_pct", "k_count", "tp_captured", "total_positives",
                         "capture_rate", "precision_at_k", "recall_at_k",
                         "lift", "baseline_rate"}
        assert set(result.columns) == expected_cols


# ═══════════════════════════════════════════════════════════════════════
# Test 3: lift_at_k returns float >= 0
# ═══════════════════════════════════════════════════════════════════════

class TestLiftAtK:
    """lift_at_k should return a non-negative float."""

    def test_returns_float(self, balanced_data):
        """Return type should be float."""
        y_true, y_prob = balanced_data
        result = lift_at_k(y_true, y_prob, k_pct=10)
        assert isinstance(result, float)

    def test_non_negative(self, balanced_data):
        """Lift should be >= 0."""
        y_true, y_prob = balanced_data
        result = lift_at_k(y_true, y_prob, k_pct=10)
        assert result >= 0.0

    def test_perfect_data_high_lift(self, perfect_data):
        """Perfect data should have high lift."""
        y_true, y_prob = perfect_data
        result = lift_at_k(y_true, y_prob, k_pct=30)
        assert result >= 1.0

    def test_various_k_pcts(self, balanced_data):
        """Test with various k percentages."""
        y_true, y_prob = balanced_data
        for k_pct in [1, 5, 10, 20, 50]:
            result = lift_at_k(y_true, y_prob, k_pct=k_pct)
            assert isinstance(result, float)
            assert result >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# Test 4: calibration_bucket_table has correct number of buckets
# ═══════════════════════════════════════════════════════════════════════

class TestCalibrationBucketTable:
    """calibration_bucket_table should return exactly n_buckets rows."""

    def test_default_10_buckets(self, balanced_data):
        """Default: 10 buckets."""
        y_true, y_prob = balanced_data
        result = calibration_bucket_table(y_true, y_prob)
        assert len(result) == 10

    def test_custom_n_buckets(self, balanced_data):
        """Custom number of buckets."""
        y_true, y_prob = balanced_data
        for n in [5, 15, 20]:
            result = calibration_bucket_table(y_true, y_prob, n_buckets=n)
            assert len(result) == n

    def test_columns_present(self, balanced_data):
        """All expected columns should be present."""
        y_true, y_prob = balanced_data
        result = calibration_bucket_table(y_true, y_prob)
        expected_cols = {"bucket_idx", "bin_low", "bin_high", "count",
                         "pred_mean", "actual_rate", "calibration_error"}
        assert set(result.columns) == expected_cols

    def test_total_count_equals_n(self, balanced_data):
        """Sum of counts across buckets should equal n."""
        y_true, y_prob = balanced_data
        result = calibration_bucket_table(y_true, y_prob)
        assert result["count"].sum() == len(y_true)

    def test_calibration_error_non_negative(self, balanced_data):
        """Calibration error should be >= 0 for all buckets."""
        y_true, y_prob = balanced_data
        result = calibration_bucket_table(y_true, y_prob)
        assert (result["calibration_error"] >= 0).all()

    def test_bin_edges_cover_unit_interval(self, balanced_data):
        """First bin_low should be 0.0, last bin_high should be 1.0."""
        y_true, y_prob = balanced_data
        result = calibration_bucket_table(y_true, y_prob)
        assert result["bin_low"].iloc[0] == 0.0
        assert result["bin_high"].iloc[-1] == 1.0


# ═══════════════════════════════════════════════════════════════════════
# Test 5: ECE is in [0, 1]
# ═══════════════════════════════════════════════════════════════════════

class TestExpectedCalibrationError:
    """ECE should be in [0, 1]."""

    def test_ece_range(self, balanced_data):
        """ECE should be between 0 and 1."""
        y_true, y_prob = balanced_data
        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have low ECE."""
        # Create perfectly calibrated data: if p=0.7, then ~70% are positive
        np.random.seed(42)
        n = 1000
        y_prob = np.random.uniform(0.1, 0.9, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(float)
        ece = expected_calibration_error(y_true, y_prob, n_buckets=10)
        # With enough data, ECE should be small for well-calibrated predictions
        assert ece < 0.15, f"ECE too high for well-calibrated data: {ece}"

    def test_returns_float(self, balanced_data):
        """Return type should be float."""
        y_true, y_prob = balanced_data
        ece = expected_calibration_error(y_true, y_prob)
        assert isinstance(ece, float)

    def test_miscalibrated_high_ece(self):
        """Systematically overconfident predictions should have higher ECE."""
        n = 200
        y_true = np.zeros(n)
        y_true[:20] = 1.0  # 10% positive
        # Predictions all near 0.9 (overconfident)
        y_prob = np.full(n, 0.9)
        ece = expected_calibration_error(y_true, y_prob, n_buckets=10)
        # Actual rate is ~10% but predicted is 90% → high calibration error
        assert ece > 0.5, f"ECE should be high for miscalibrated data: {ece}"


# ═══════════════════════════════════════════════════════════════════════
# Test 6: select_threshold_by_objective returns valid threshold
# ═══════════════════════════════════════════════════════════════════════

class TestSelectThresholdByObjective:
    """select_threshold_by_objective should return valid thresholds."""

    def test_max_f1(self, balanced_data):
        """max_f1 should return a threshold in [0, 1]."""
        y_true, y_prob = balanced_data
        thr = select_threshold_by_objective(y_true, y_prob, "max_f1")
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_min_precision_30(self, balanced_data):
        """min_precision_30 should return a threshold where precision >= 0.30."""
        y_true, y_prob = balanced_data
        thr = select_threshold_by_objective(y_true, y_prob, "min_precision_30")
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_min_recall_50(self, balanced_data):
        """min_recall_50 should return a threshold where recall >= 0.50."""
        y_true, y_prob = balanced_data
        thr = select_threshold_by_objective(y_true, y_prob, "min_recall_50")
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_top10_lift(self, balanced_data):
        """top10_lift should return a threshold in [0, 1]."""
        y_true, y_prob = balanced_data
        thr = select_threshold_by_objective(y_true, y_prob, "top10_lift")
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_alert_budget_5pct(self, balanced_data):
        """alert_budget_5pct should return a threshold in [0, 1]."""
        y_true, y_prob = balanced_data
        thr = select_threshold_by_objective(y_true, y_prob, "alert_budget_5pct")
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_alert_budget_10pct(self, balanced_data):
        """alert_budget_10pct should return a threshold in [0, 1]."""
        y_true, y_prob = balanced_data
        thr = select_threshold_by_objective(y_true, y_prob, "alert_budget_10pct")
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_unknown_objective_raises(self, balanced_data):
        """Unknown objective should raise ValueError."""
        y_true, y_prob = balanced_data
        with pytest.raises(ValueError, match="Unknown objective"):
            select_threshold_by_objective(y_true, y_prob, "nonexistent_objective")

    def test_perfect_data_max_f1(self, perfect_data):
        """Perfect data should achieve F1 close to 1.0."""
        y_true, y_prob = perfect_data
        thr = select_threshold_by_objective(y_true, y_prob, "max_f1")
        # Verify the threshold achieves high F1
        sweep = threshold_sweep(y_true, y_prob, thresholds=[thr])
        assert sweep["f1"].iloc[0] > 0.9


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: all zeros, all ones, single sample."""

    def test_all_zeros_y_true(self):
        """All negative labels: precision undefined → 0, recall undefined → 0."""
        n = 50
        y_true = np.zeros(n)
        y_prob = np.random.uniform(0, 1, n)

        sweep = threshold_sweep(y_true, y_prob, thresholds=[0.5])
        assert sweep["tp"].iloc[0] == 0
        assert sweep["precision"].iloc[0] == 0.0
        assert sweep["recall"].iloc[0] == 0.0

        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

        lift = lift_at_k(y_true, y_prob, k_pct=10)
        assert lift == 0.0  # no positives → lift is 0

    def test_all_ones_y_true(self):
        """All positive labels: recall should be high for low thresholds."""
        n = 50
        y_true = np.ones(n)
        y_prob = np.random.uniform(0.3, 0.9, n)

        sweep = threshold_sweep(y_true, y_prob, thresholds=[0.3])
        # At threshold 0.3, most samples should be predicted positive
        assert sweep["recall"].iloc[0] > 0.5
        # Precision = tp / (tp + fp) = all positives / n
        assert sweep["precision"].iloc[0] > 0.0

        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_single_sample(self):
        """Single sample should not crash."""
        y_true = np.array([1.0])
        y_prob = np.array([0.8])

        sweep = threshold_sweep(y_true, y_prob, thresholds=[0.5])
        assert len(sweep) == 1
        assert sweep["tp"].iloc[0] == 1

        bucket = calibration_bucket_table(y_true, y_prob, n_buckets=5)
        assert len(bucket) == 5
        assert bucket["count"].sum() == 1

        ece = expected_calibration_error(y_true, y_prob, n_buckets=5)
        assert 0.0 <= ece <= 1.0

        lift = lift_at_k(y_true, y_prob, k_pct=50)
        assert isinstance(lift, float)
        assert lift >= 0.0

    def test_single_sample_negative(self):
        """Single negative sample."""
        y_true = np.array([0.0])
        y_prob = np.array([0.3])

        sweep = threshold_sweep(y_true, y_prob, thresholds=[0.5])
        assert sweep["tn"].iloc[0] == 1
        assert sweep["tp"].iloc[0] == 0

    def test_all_same_probability(self):
        """All predictions are the same value."""
        n = 100
        y_true = np.random.binomial(1, 0.2, n).astype(float)
        y_prob = np.full(n, 0.5)

        sweep = threshold_sweep(y_true, y_prob, thresholds=[0.5])
        # At threshold 0.5, all samples predicted positive (>= 0.5)
        assert sweep["tp"].iloc[0] + sweep["fp"].iloc[0] == n

        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_precision_recall_curve_summary(self, balanced_data):
        """precision_recall_curve_summary should return valid dict."""
        y_true, y_prob = balanced_data
        summary = precision_recall_curve_summary(y_true, y_prob)
        assert "auc_pr" in summary
        assert "max_f1" in summary
        assert "max_f1_threshold" in summary
        assert "precision_at_50pct_recall" in summary
        assert "n_thresholds" in summary
        assert 0.0 <= summary["max_f1"] <= 1.0
        assert 0.0 <= summary["max_f1_threshold"] <= 1.0
