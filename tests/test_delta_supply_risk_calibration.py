"""Tests for DeltaSupply risk calibration script.

Covers:
  1. Threshold sweep produces 19 rows (thresholds 0.05..0.95)
  2. Top-k capture rates are in [0, 1]
  3. Lift >= 1.0 always (by definition)
  4. Bucket calibration has 10 buckets
  5. Verdict is one of RISK_FEATURE_GO / RISK_FEATURE_LOW_VALUE / RISK_FEATURE_NO_GO
  6. Script runs end-to-end with fixture data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_delta_supply_risk import (
    N_BUCKETS,
    THRESHOLDS,
    TOPK_PCTS,
    bucket_calibration,
    decide_verdict,
    main,
    pick_best_thresholds,
    run_calibration,
    threshold_sweep,
    topk_capture,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def fixture_pred_df():
    """Build a synthetic predictions DataFrame with known structure.

    Creates 1000 rows with correlated probabilities and labels so that
    metrics are non-trivial (not all zeros / all ones).
    """
    rng = np.random.RandomState(42)
    n = 1000

    # Generate probabilities from a beta distribution (realistic shape)
    upward_prob = rng.beta(2, 5, size=n)
    downward_prob = rng.beta(2, 5, size=n)
    large_abs_prob = rng.beta(2, 5, size=n)

    # Labels correlated with probabilities (higher prob → more likely positive)
    upward_label = (rng.rand(n) < upward_prob).astype(int)
    downward_label = (rng.rand(n) < downward_prob).astype(int)
    large_abs_label = (rng.rand(n) < large_abs_prob).astype(int)

    return pd.DataFrame({
        "upward_deviation_prob": upward_prob,
        "downward_deviation_prob": downward_prob,
        "large_abs_deviation_prob": large_abs_prob,
        "upward_deviation_label": upward_label,
        "downward_deviation_label": downward_label,
        "large_abs_deviation_label": large_abs_label,
        "deviation_risk_score": (upward_prob + downward_prob + large_abs_prob) / 3,
    })


@pytest.fixture()
def fixture_y_arrays():
    """Return simple (y_true, y_prob) arrays for unit-level tests."""
    rng = np.random.RandomState(7)
    n = 500
    y_prob = rng.rand(n)
    y_true = (rng.rand(n) < y_prob * 0.6).astype(int)  # moderate positive rate
    return y_true, y_prob


# ── 1. Threshold sweep ────────────────────────────────────────────────────

class TestThresholdSweep:
    def test_correct_number_of_rows(self, fixture_y_arrays):
        """Sweep must produce exactly 19 rows for thresholds 0.05..0.95."""
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        assert len(sweep_df) == 19

    def test_threshold_values(self, fixture_y_arrays):
        """Thresholds should be exactly 0.05, 0.10, ..., 0.95."""
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        expected = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
        assert sweep_df["threshold"].tolist() == expected

    def test_required_columns(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        for col in ["threshold", "precision", "recall", "f1", "support", "alert_rate"]:
            assert col in sweep_df.columns

    def test_precision_recall_in_range(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        assert (sweep_df["precision"] >= 0).all() and (sweep_df["precision"] <= 1).all()
        assert (sweep_df["recall"] >= 0).all() and (sweep_df["recall"] <= 1).all()
        assert (sweep_df["f1"] >= 0).all() and (sweep_df["f1"] <= 1).all()
        assert (sweep_df["alert_rate"] >= 0).all() and (sweep_df["alert_rate"] <= 1).all()

    def test_high_threshold_low_recall(self, fixture_y_arrays):
        """At threshold 0.95, recall should be low (few positives flagged)."""
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        last_row = sweep_df.iloc[-1]
        assert last_row["threshold"] == 0.95
        assert last_row["recall"] <= 0.5  # very few positives captured

    def test_low_threshold_high_alert_rate(self, fixture_y_arrays):
        """At threshold 0.05, alert_rate should be high (most flagged)."""
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        first_row = sweep_df.iloc[0]
        assert first_row["threshold"] == 0.05
        assert first_row["alert_rate"] >= 0.5  # most samples flagged


class TestPickBestThresholds:
    def test_returns_dict_with_expected_keys(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        best = pick_best_thresholds(sweep_df)
        assert "best_f1_threshold" in best
        assert "best_precision_at_recall_30" in best
        assert "best_recall_at_precision_30" in best

    def test_best_f1_threshold_is_valid(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        sweep_df = threshold_sweep(y_true, y_prob)
        best = pick_best_thresholds(sweep_df)
        assert best["best_f1_threshold"] in THRESHOLDS

    def test_empty_sweep_returns_nones(self):
        best = pick_best_thresholds(pd.DataFrame())
        assert best["best_f1_threshold"] is None


# ── 2. Top-k capture ─────────────────────────────────────────────────────

class TestTopkCapture:
    def test_capture_rates_in_range(self, fixture_y_arrays):
        """All capture rates must be in [0, 1]."""
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        assert (topk_df["capture_rate"] >= 0).all()
        assert (topk_df["capture_rate"] <= 1).all()

    def test_lift_at_least_one(self, fixture_y_arrays):
        """Lift vs random must be >= 1.0 by definition (top-k by predicted prob
        cannot be worse than random selection when k < n)."""
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        assert (topk_df["lift_vs_random"] >= 1.0 - 1e-9).all(), (
            f"Lift below 1.0:\n{topk_df[['topk_pct', 'lift_vs_random']]}"
        )

    def test_correct_number_of_rows(self, fixture_y_arrays):
        """Should have one row per top-k percentage."""
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        assert len(topk_df) == len(TOPK_PCTS)

    def test_required_columns(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        for col in ["topk_pct", "k", "capture_rate", "lift_vs_random",
                     "precision_at_k", "recall_at_k"]:
            assert col in topk_df.columns

    def test_recall_increases_with_k(self, fixture_y_arrays):
        """Recall at k should be non-decreasing as k grows."""
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        recalls = topk_df["recall_at_k"].values
        for i in range(len(recalls) - 1):
            assert recalls[i + 1] >= recalls[i] - 1e-9

    def test_precision_at_k_in_range(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        assert (topk_df["precision_at_k"] >= 0).all()
        assert (topk_df["precision_at_k"] <= 1).all()


# ── 3. Bucket calibration ────────────────────────────────────────────────

class TestBucketCalibration:
    def test_ten_buckets(self, fixture_y_arrays):
        """Must produce exactly 10 buckets."""
        y_true, y_prob = fixture_y_arrays
        bucket_df = bucket_calibration(y_true, y_prob)
        assert len(bucket_df) == N_BUCKETS
        assert len(bucket_df) == 10

    def test_required_columns(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        bucket_df = bucket_calibration(y_true, y_prob)
        for col in ["bucket_lo", "bucket_hi", "bucket_pred_mean",
                     "bucket_actual_rate", "bucket_count", "calibration_error"]:
            assert col in bucket_df.columns

    def test_bucket_edges_cover_unit_interval(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        bucket_df = bucket_calibration(y_true, y_prob)
        assert bucket_df["bucket_lo"].iloc[0] == 0.0
        assert bucket_df["bucket_hi"].iloc[-1] == 1.0

    def test_total_count_matches_input(self, fixture_y_arrays):
        """Sum of bucket counts should equal total valid samples."""
        y_true, y_prob = fixture_y_arrays
        bucket_df = bucket_calibration(y_true, y_prob)
        assert bucket_df["bucket_count"].sum() == len(y_true)

    def test_calibration_error_non_negative(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        bucket_df = bucket_calibration(y_true, y_prob)
        assert (bucket_df["calibration_error"] >= 0).all()

    def test_actual_rate_in_range(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        bucket_df = bucket_calibration(y_true, y_prob)
        assert (bucket_df["bucket_actual_rate"] >= 0).all()
        assert (bucket_df["bucket_actual_rate"] <= 1).all()


# ── 4. Verdict ────────────────────────────────────────────────────────────

class TestVerdict:
    VALID_VERDICTS = {"RISK_FEATURE_GO", "RISK_FEATURE_LOW_VALUE", "RISK_FEATURE_NO_GO"}

    def test_verdict_is_valid_label(self, fixture_y_arrays):
        """Verdict must be one of the three allowed labels."""
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        bucket_df = bucket_calibration(y_true, y_prob)
        result = decide_verdict(topk_df, bucket_df)
        assert result["verdict"] in self.VALID_VERDICTS

    def test_verdict_contains_criteria(self, fixture_y_arrays):
        y_true, y_prob = fixture_y_arrays
        topk_df = topk_capture(y_true, y_prob)
        bucket_df = bucket_calibration(y_true, y_prob)
        result = decide_verdict(topk_df, bucket_df)
        assert "criteria" in result
        assert "go_lift_threshold" in result["criteria"]
        assert "low_value_lift_threshold" in result["criteria"]

    def test_nogo_when_lift_low(self):
        """If lift < 1.3, verdict must be NO_GO."""
        # Construct a topk_df with lift_top10 < 1.3
        topk_df = pd.DataFrame({
            "topk_pct": TOPK_PCTS,
            "k": [10, 30, 50, 100, 200],
            "capture_rate": [0.05, 0.10, 0.15, 0.25, 0.40],
            "lift_vs_random": [0.5, 0.33, 0.30, 0.25, 0.20],  # all < 1.3
            "precision_at_k": [0.05, 0.05, 0.05, 0.05, 0.04],
            "recall_at_k": [0.05, 0.10, 0.15, 0.25, 0.40],
        })
        bucket_df = pd.DataFrame({
            "bucket_lo": np.linspace(0, 0.9, 10),
            "bucket_hi": np.linspace(0.1, 1.0, 10),
            "bucket_pred_mean": np.linspace(0.05, 0.95, 10),
            "bucket_actual_rate": np.linspace(0.05, 0.95, 10),
            "bucket_count": [100] * 10,
            "calibration_error": [0.0] * 10,
        })
        result = decide_verdict(topk_df, bucket_df)
        assert result["verdict"] == "RISK_FEATURE_NO_GO"

    def test_go_when_strong_signal(self):
        """If lift >= 2.0, recall >= 0.4, and calibration monotonic → GO."""
        topk_df = pd.DataFrame({
            "topk_pct": TOPK_PCTS,
            "k": [10, 30, 50, 100, 200],
            "capture_rate": [0.40, 0.50, 0.60, 0.80, 0.95],
            "lift_vs_random": [4.0, 1.67, 1.20, 2.5, 0.475],  # topk_pct=10 → lift=2.5
            "precision_at_k": [0.40, 0.25, 0.20, 0.16, 0.095],
            "recall_at_k": [0.40, 0.50, 0.60, 0.80, 0.95],  # topk_pct=20 → recall=0.95
        })
        bucket_df = pd.DataFrame({
            "bucket_lo": np.linspace(0, 0.9, 10),
            "bucket_hi": np.linspace(0.1, 1.0, 10),
            "bucket_pred_mean": np.linspace(0.05, 0.95, 10),
            "bucket_actual_rate": np.linspace(0.02, 0.80, 10),  # monotonically increasing
            "bucket_count": [100] * 10,
            "calibration_error": [0.03] * 10,
        })
        result = decide_verdict(topk_df, bucket_df)
        assert result["verdict"] == "RISK_FEATURE_GO"

    def test_low_value_when_moderate_lift(self):
        """If 1.3 <= lift < 2.0 → LOW_VALUE."""
        topk_df = pd.DataFrame({
            "topk_pct": TOPK_PCTS,
            "k": [10, 30, 50, 100, 200],
            "capture_rate": [0.20, 0.35, 0.45, 0.65, 0.85],
            "lift_vs_random": [2.0, 1.17, 0.90, 1.5, 0.425],  # topk_pct=10 → lift=1.5
            "precision_at_k": [0.20, 0.17, 0.15, 0.13, 0.085],
            "recall_at_k": [0.20, 0.35, 0.45, 0.65, 0.85],
        })
        bucket_df = pd.DataFrame({
            "bucket_lo": np.linspace(0, 0.9, 10),
            "bucket_hi": np.linspace(0.1, 1.0, 10),
            "bucket_pred_mean": np.linspace(0.05, 0.95, 10),
            "bucket_actual_rate": np.linspace(0.05, 0.50, 10),
            "bucket_count": [100] * 10,
            "calibration_error": [0.05] * 10,
        })
        result = decide_verdict(topk_df, bucket_df)
        assert result["verdict"] == "RISK_FEATURE_LOW_VALUE"


# ── 5. run_calibration integration ────────────────────────────────────────

class TestRunCalibration:
    def test_all_three_directions_present(self, fixture_pred_df):
        """run_calibration should produce results for all three directions."""
        results = run_calibration(fixture_pred_df)
        assert "upward" in results
        assert "downward" in results
        assert "large_abs" in results

    def test_each_direction_has_all_outputs(self, fixture_pred_df):
        results = run_calibration(fixture_pred_df)
        for direction, res in results.items():
            assert isinstance(res["threshold_sweep"], pd.DataFrame)
            assert isinstance(res["topk_capture"], pd.DataFrame)
            assert isinstance(res["bucket_calibration"], pd.DataFrame)
            assert isinstance(res["verdict"], dict)
            assert isinstance(res["best_thresholds"], dict)

    def test_missing_columns_gracefully_skipped(self):
        """If a direction's columns are missing, it should be skipped."""
        df = pd.DataFrame({
            "upward_deviation_prob": [0.5, 0.8],
            "upward_deviation_label": [0, 1],
        })
        results = run_calibration(df)
        assert "upward" in results
        assert "downward" not in results
        assert "large_abs" not in results


# ── 6. End-to-end CLI ────────────────────────────────────────────────────

class TestEndToEnd:
    def test_script_runs_end_to_end(self, fixture_pred_df, tmp_path):
        """Full CLI invocation should produce all expected output files."""
        pred_path = tmp_path / "predictions.csv"
        fixture_pred_df.to_csv(pred_path, index=False)
        out_dir = tmp_path / "risk_output"

        old_argv = sys.argv
        sys.argv = [
            "calibrate_delta_supply_risk.py",
            "--predictions", str(pred_path),
            "--out-dir", str(out_dir),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        # Check all output files exist
        assert (out_dir / "threshold_sweep.csv").exists()
        assert (out_dir / "topk_capture.csv").exists()
        assert (out_dir / "bucket_calibration.csv").exists()
        assert (out_dir / "risk_verdict.json").exists()

    def test_output_csvs_parseable(self, fixture_pred_df, tmp_path):
        """Output CSVs should be valid and parseable."""
        pred_path = tmp_path / "predictions.csv"
        fixture_pred_df.to_csv(pred_path, index=False)
        out_dir = tmp_path / "risk_output2"

        old_argv = sys.argv
        sys.argv = [
            "calibrate_delta_supply_risk.py",
            "--predictions", str(pred_path),
            "--out-dir", str(out_dir),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        sweep = pd.read_csv(out_dir / "threshold_sweep.csv")
        topk = pd.read_csv(out_dir / "topk_capture.csv")
        bucket = pd.read_csv(out_dir / "bucket_calibration.csv")

        # 3 directions × 19 thresholds = 57 rows
        assert len(sweep) == 3 * 19
        # 3 directions × 5 top-k levels = 15 rows
        assert len(topk) == 3 * len(TOPK_PCTS)
        # 3 directions × 10 buckets = 30 rows
        assert len(bucket) == 3 * N_BUCKETS

    def test_verdict_json_structure(self, fixture_pred_df, tmp_path):
        """risk_verdict.json should have the expected structure."""
        pred_path = tmp_path / "predictions.csv"
        fixture_pred_df.to_csv(pred_path, index=False)
        out_dir = tmp_path / "risk_output3"

        old_argv = sys.argv
        sys.argv = [
            "calibrate_delta_supply_risk.py",
            "--predictions", str(pred_path),
            "--out-dir", str(out_dir),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        with open(out_dir / "risk_verdict.json", "r", encoding="utf-8") as f:
            verdict_data = json.load(f)

        assert "directions" in verdict_data
        assert "overall_verdict" in verdict_data
        assert verdict_data["overall_verdict"] in {
            "RISK_FEATURE_GO", "RISK_FEATURE_LOW_VALUE", "RISK_FEATURE_NO_GO"
        }

        for direction_name, vdict in verdict_data["directions"].items():
            assert vdict["verdict"] in {
                "RISK_FEATURE_GO", "RISK_FEATURE_LOW_VALUE", "RISK_FEATURE_NO_GO"
            }
            assert "lift_top10pct" in vdict
            assert "recall_top20pct" in vdict
            assert "calibration_monotonic" in vdict
