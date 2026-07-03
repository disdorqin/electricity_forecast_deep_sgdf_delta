"""Tests for DeltaSupply training and evaluation scripts (smoke tests)."""
import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_test_csv(tmp_path, n_days=120):
    """Create a minimal test CSV with Chinese column names."""
    n_hours = n_days * 24
    dates = pd.date_range("2025-10-01", periods=n_hours, freq="h")
    rng = np.random.RandomState(42)

    df = pd.DataFrame({
        "时刻": dates,
        "日前电价": 300 + rng.randn(n_hours) * 50,
        "实时电价": 300 + rng.randn(n_hours) * 80,
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
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestTrainScript:
    def test_train_runs_end_to_end(self, tmp_path):
        """Smoke test: training script runs without error."""
        csv_path = _make_test_csv(tmp_path, n_days=90)
        out_dir = str(tmp_path / "train_output")

        from scripts.train_delta_supply_module import main
        import sys
        old_argv = sys.argv
        sys.argv = [
            "train_delta_supply_module.py",
            "--data-path", csv_path,
            "--target-month", "2025-12",
            "--out-dir", out_dir,
            "--fast-dev-run",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        # Check outputs exist
        out = Path(out_dir)
        assert (out / "model.pkl").exists()
        assert (out / "config.yaml").exists()
        assert (out / "feature_manifest.json").exists()
        assert (out / "train_manifest.json").exists()
        assert (out / "predictions.csv").exists()
        assert (out / "feature_importance.csv").exists()
        assert (out / "metrics_summary.json").exists()

    def test_train_manifest_has_required_fields(self, tmp_path):
        csv_path = _make_test_csv(tmp_path, n_days=90)
        out_dir = str(tmp_path / "train_output")

        from scripts.train_delta_supply_module import main
        import sys
        old_argv = sys.argv
        sys.argv = [
            "train_delta_supply_module.py",
            "--data-path", csv_path,
            "--target-month", "2025-12",
            "--out-dir", out_dir,
            "--fast-dev-run",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        with open(Path(out_dir) / "train_manifest.json") as f:
            manifest = json.load(f)

        required = ["target_month", "n_train", "n_test", "feature_columns",
                     "n_features", "feature_audit_verdict", "thresholds", "created_at"]
        for field in required:
            assert field in manifest, f"Missing field: {field}"


class TestEvalScript:
    def test_eval_runs_end_to_end(self, tmp_path):
        """Smoke test: evaluation script runs without error."""
        # First train
        csv_path = _make_test_csv(tmp_path, n_days=90)
        train_dir = str(tmp_path / "train_output")

        from scripts.train_delta_supply_module import main as train_main
        import sys
        old_argv = sys.argv
        sys.argv = [
            "train_delta_supply_module.py",
            "--data-path", csv_path,
            "--target-month", "2025-12",
            "--out-dir", train_dir,
            "--fast-dev-run",
        ]
        try:
            train_main()
        finally:
            sys.argv = old_argv

        # Then evaluate
        eval_dir = str(tmp_path / "eval_output")
        pred_path = str(Path(train_dir) / "predictions.csv")

        from scripts.evaluate_delta_supply_module import main as eval_main
        sys.argv = [
            "evaluate_delta_supply_module.py",
            "--predictions", pred_path,
            "--out-dir", eval_dir,
        ]
        try:
            eval_main()
        finally:
            sys.argv = old_argv

        # Check outputs
        out = Path(eval_dir)
        assert (out / "metrics_summary.json").exists()
        assert (out / "classification_metrics.csv").exists()
        assert (out / "regression_metrics.csv").exists()
        assert (out / "correction_simulation.csv").exists()
        assert (out / "go_nogo.md").exists()


class TestCorrectionSimulation:
    def test_correction_uses_predicted_magnitude_only(self):
        """Correction simulation must use model prediction, not test actual."""
        from scripts.evaluate_delta_supply_module import run_correction_simulation

        pred_df = pd.DataFrame({
            "da_anchor": [300.0, 300.0, 300.0, 300.0],
            "rt_actual": [400.0, 200.0, 350.0, 250.0],
            "deviation_magnitude_pred": [50.0, -50.0, 30.0, -30.0],
        })

        result = run_correction_simulation(pred_df, [0.0, 0.5, 1.0])
        assert len(result) == 3
        assert "correction_weight" in result.columns
        assert "improvement_pp" in result.columns

        # weight=0.0 should give same as DA anchor
        row_w0 = result[result["correction_weight"] == 0.0].iloc[0]
        assert abs(row_w0["corrected_smape"] - row_w0["da_anchor_smape"]) < 1e-10
