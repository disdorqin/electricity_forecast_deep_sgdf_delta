"""Tests for run_delta_supply_risk_backtest.py.

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

def _make_synthetic_csv(tmp_path, n_days=210, seed=42):
    """Create synthetic hourly CSV with enough deviation events.

    Generates data spanning Oct 2025 – Apr 2026 so that target months
    2026-01 .. 2026-03 have sufficient train/val/test data.

    Deviation events are injected by making rt_actual differ from da_anchor
    by >= 100 in some hours.
    """
    rng = np.random.RandomState(seed)
    n_hours = n_days * 24
    dates = pd.date_range("2025-10-01", periods=n_hours, freq="h")

    da = 300 + rng.randn(n_hours) * 50
    rt = da + rng.randn(n_hours) * 30  # small noise → most hours no deviation

    # Inject upward deviation events (price_delta >= 100) in ~5 % of hours
    n_up = max(15, int(n_hours * 0.05))
    up_idx = rng.choice(n_hours, size=n_up, replace=False)
    rt[up_idx] = da[up_idx] + 100 + rng.rand(n_up) * 200

    # Inject downward deviation events (price_delta <= -100) in ~5 % of hours
    n_down = max(15, int(n_hours * 0.05))
    down_idx = rng.choice(n_hours, size=n_down, replace=False)
    rt[down_idx] = da[down_idx] - 100 - rng.rand(n_down) * 200

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
    csv_path = tmp_path / "synthetic_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestDeltaSupplyRiskBacktest:
    """End-to-end tests for the delta supply risk backtest script."""

    def test_backtest_runs_end_to_end(self, tmp_path):
        """Backtest runs without error and produces expected output files."""
        csv_path = _make_synthetic_csv(tmp_path)
        out_dir = str(tmp_path / "risk_bt_output")

        from scripts.run_delta_supply_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_delta_supply_risk_backtest.py",
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
        assert (out / "delta_supply_risk_backtest_report.md").exists()
        assert (out / "verdict.json").exists()

    def test_monthly_metrics_has_correct_columns(self, tmp_path):
        """monthly_metrics.csv contains the expected columns."""
        csv_path = _make_synthetic_csv(tmp_path)
        out_dir = str(tmp_path / "risk_bt_output")

        from scripts.run_delta_supply_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_delta_supply_risk_backtest.py",
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
        """Thresholds in thresholds.csv should come from validation set,
        not from the test set.  We verify this by checking that the threshold
        selection record contains val_f1/val_precision/val_recall columns."""
        csv_path = _make_synthetic_csv(tmp_path)
        out_dir = str(tmp_path / "risk_bt_output")

        from scripts.run_delta_supply_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_delta_supply_risk_backtest.py",
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

    def test_insufficient_events_handled(self, tmp_path):
        """When positive count < 10, the month/direction should be marked
        INSUFFICIENT_EVENTS in monthly_metrics.csv."""
        # Create data with NO deviation events in a specific month
        rng = np.random.RandomState(99)
        n_hours = 24 * 210
        dates = pd.date_range("2025-10-01", periods=n_hours, freq="h")
        da = 300 + rng.randn(n_hours) * 50
        # Very small noise → no deviations
        rt = da + rng.randn(n_hours) * 5

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
        csv_path = str(tmp_path / "no_events.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "risk_bt_insufficient")

        from scripts.run_delta_supply_risk_backtest import main

        old_argv = sys.argv
        sys.argv = [
            "run_delta_supply_risk_backtest.py",
            "--data-path", csv_path,
            "--target-months", "2026-01",
            "--out-dir", out_dir,
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        metrics_df = pd.read_csv(Path(out_dir) / "monthly_metrics.csv")
        # All directions should be INSUFFICIENT_EVENTS (no deviation events)
        if not metrics_df.empty:
            statuses = metrics_df["status"].unique()
            assert any("INSUFFICIENT" in str(s) for s in statuses), (
                f"Expected INSUFFICIENT_EVENTS status, got: {statuses}"
            )
