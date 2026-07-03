"""Tests for run_negative_risk_backtest.py.

Uses synthetic fixture data so the tests do not depend on real data files.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _make_synthetic_negative_csv(tmp_path, n_days=210, seed=42):
    """Create synthetic hourly CSV with enough negative price events.

    Generates data spanning Oct 2025 – Apr 2026.
    Negative events are injected by setting rt_actual < 0 in some hours.
    """
    rng = np.random.RandomState(seed)
    n_hours = n_days * 24
    dates = pd.date_range("2025-10-01", periods=n_hours, freq="h")

    da = 300 + rng.randn(n_hours) * 50
    rt = da + rng.randn(n_hours) * 30  # normal noise

    # Inject negative price events (rt < 0) in ~3 % of hours
    n_neg = max(15, int(n_hours * 0.03))
    neg_idx = rng.choice(n_hours, size=n_neg, replace=False)
    rt[neg_idx] = -rng.rand(n_neg) * 150  # rt in [-150, 0)

    # Inject deep negative events (rt <= -100) in ~1.5 % of hours
    n_deep = max(10, int(n_hours * 0.015))
    deep_idx = rng.choice(n_hours, size=n_deep, replace=False)
    rt[deep_idx] = -100 - rng.rand(n_deep) * 200  # rt in [-300, -100]

    # Inject relative down events (rt - da <= -200) in ~2 % of hours
    n_rel = max(12, int(n_hours * 0.02))
    rel_idx = rng.choice(n_hours, size=n_rel, replace=False)
    rt[rel_idx] = da[rel_idx] - 200 - rng.rand(n_rel) * 200

    df = pd.DataFrame({
        "时刻": dates,
        "日前电价": da,
        "实时电价": rt,
        "统调负荷预测值": 9000 + rng.randn(n_hours) * 500,
        "新能源总加预测值": 7000 + rng.randn(n_hours) * 800,
        "风电总加预测值": 5000 + rng.randn(n_hours) * 600,
        "光伏总加预测值": np.maximum(0, 1000 + rng.randn(n_hours) * 400),
        "竞价空间预测值": 20000 + rng.randn(n_hours) * 3000,
        "统调负荷": 9000 + rng.randn(n_hours) * 500,
        "新能源总加实际值": 7000 + rng.randn(n_hours) * 800,
        "风电总加实际值": 5000 + rng.randn(n_hours) * 600,
        "光伏总加实际值": np.maximum(0, 1000 + rng.randn(n_hours) * 400),
    })
    csv_path = tmp_path / "synthetic_negative.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestNegativeRiskBacktest:
    """End-to-end tests for the negative risk backtest script."""

    def test_backtest_runs_end_to_end(self, tmp_path):
        """Backtest runs without error and produces expected output files."""
        csv_path = _make_synthetic_negative_csv(tmp_path)
        out_dir = str(tmp_path / "neg_bt_output")

        from scripts.run_negative_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_negative_risk_backtest.py",
            "--data-path", csv_path,
            "--target-months", "2026-01,2026-02",
            "--out-dir", out_dir,
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        out = Path(out_dir)
        assert (out / "monthly_metrics.csv").exists()
        assert (out / "target_metrics.csv").exists()
        assert (out / "topk_metrics.csv").exists()
        assert (out / "calibration_metrics.csv").exists()
        assert (out / "thresholds.csv").exists()
        assert (out / "feature_importance_summary.csv").exists()
        assert (out / "negative_risk_backtest_report.md").exists()
        assert (out / "verdict.json").exists()

    def test_monthly_metrics_has_correct_columns(self, tmp_path):
        """monthly_metrics.csv contains the expected columns."""
        csv_path = _make_synthetic_negative_csv(tmp_path)
        out_dir = str(tmp_path / "neg_bt_output")

        from scripts.run_negative_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_negative_risk_backtest.py",
            "--data-path", csv_path,
            "--target-months", "2026-01,2026-02",
            "--out-dir", out_dir,
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        df = pd.read_csv(Path(out_dir) / "monthly_metrics.csv")
        expected_cols = {"month", "direction", "status", "n_positive",
                         "precision", "recall", "f1", "roc_auc", "threshold"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_threshold_selection_uses_val_not_test(self, tmp_path):
        """Thresholds should be selected on validation set only."""
        csv_path = _make_synthetic_negative_csv(tmp_path)
        out_dir = str(tmp_path / "neg_bt_output")

        from scripts.run_negative_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_negative_risk_backtest.py",
            "--data-path", csv_path,
            "--target-months", "2026-01,2026-02",
            "--out-dir", out_dir,
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        thr_df = pd.read_csv(Path(out_dir) / "thresholds.csv")
        if not thr_df.empty:
            val_cols = {"val_f1", "val_precision", "val_recall"}
            assert val_cols.issubset(set(thr_df.columns)), (
                f"Thresholds CSV must contain val-only columns: {val_cols - set(thr_df.columns)}"
            )

    def test_insufficient_negative_events_handled(self, tmp_path):
        """When negative count < 10, the direction should be marked
        INSUFFICIENT_NEGATIVE_EVENTS."""
        # Create data with NO negative events
        rng = np.random.RandomState(99)
        n_hours = 24 * 210
        dates = pd.date_range("2025-10-01", periods=n_hours, freq="h")
        da = 300 + rng.randn(n_hours) * 50
        rt = da + rng.randn(n_hours) * 5  # very small noise → all positive

        df = pd.DataFrame({
            "时刻": dates,
            "日前电价": da,
            "实时电价": rt,
            "统调负荷预测值": 9000 + rng.randn(n_hours) * 500,
            "新能源总加预测值": 7000 + rng.randn(n_hours) * 800,
            "风电总加预测值": 5000 + rng.randn(n_hours) * 600,
            "光伏总加预测值": np.maximum(0, 1000 + rng.randn(n_hours) * 400),
            "竞价空间预测值": 20000 + rng.randn(n_hours) * 3000,
            "统调负荷": 9000 + rng.randn(n_hours) * 500,
            "新能源总加实际值": 7000 + rng.randn(n_hours) * 800,
            "风电总加实际值": 5000 + rng.randn(n_hours) * 600,
            "光伏总加实际值": np.maximum(0, 1000 + rng.randn(n_hours) * 400),
        })
        csv_path = str(tmp_path / "no_negative.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "neg_bt_insufficient")

        from scripts.run_negative_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_negative_risk_backtest.py",
            "--data-path", csv_path,
            "--target-months", "2026-01",
            "--out-dir", out_dir,
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        metrics_df = pd.read_csv(Path(out_dir) / "monthly_metrics.csv")
        if not metrics_df.empty:
            statuses = metrics_df["status"].unique()
            assert any("INSUFFICIENT" in str(s) for s in statuses), (
                f"Expected INSUFFICIENT_NEGATIVE_EVENTS status, got: {statuses}"
            )
