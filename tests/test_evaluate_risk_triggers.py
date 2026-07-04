"""Tests for RiskTriggerEvaluator."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

from scripts.evaluate_risk_triggers import (
    RiskTriggerEvaluator,
    RiskTriggerEvalConfig,
    evaluate_risk_triggers,
)


def _create_test_data_for_trigger_eval(
    out_dir: Path,
    n_hours: int = 100,
    start_date: str = "2026-01-01",
) -> tuple[Path, Path]:
    """Create test data for trigger evaluation.
    
    Returns:
        Tuple of (risk_pack_file, data_file)
    """
    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
    
    # Create risk pack
    risk_df = pd.DataFrame({
        "ds": timestamps,
    })
    
    # Add business time
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    risk_df = add_business_time_columns(risk_df, timestamp_col="ds")
    risk_df["target_month"] = risk_df["business_day"].dt.strftime("%Y-%m")
    
    # Add risk scores
    risk_df["negative_prob"] = np.random.uniform(0, 1, n_hours)
    risk_df["negative_risk_score"] = risk_df["negative_prob"]
    risk_df["spike_prob"] = np.random.uniform(0, 1, n_hours)
    risk_df["spike_risk_score"] = risk_df["spike_prob"]
    risk_df["deviation_down_prob"] = np.random.uniform(0, 1, n_hours)
    risk_df["deviation_up_prob"] = np.random.uniform(0, 1, n_hours)
    
    # Add y_true (actual prices)
    # Some low prices (negative events)
    y_true = np.random.uniform(200, 800, n_hours)
    y_true[:10] = np.random.uniform(50, 150, 10)  # Low prices
    y_true[10:20] = np.random.uniform(1000, 1500, 10)  # High prices (spikes)
    risk_df["y_true"] = y_true
    
    # Save risk pack
    risk_pack_file = out_dir / "test_risk_pack.csv"
    risk_df.to_csv(risk_pack_file, index=False)
    
    # Create data file (with price column)
    data_df = risk_df[["ds", "y_true"]].copy()
    data_df = data_df.rename(columns={"y_true": "price"})
    data_df["ds"] = pd.to_datetime(data_df["ds"])
    
    # Save data file
    data_file = out_dir / "test_data.csv"
    data_df.to_csv(data_file, index=False)
    
    return risk_pack_file, data_file


class TestRiskTriggerEvalConfig:
    """Tests for RiskTriggerEvalConfig."""

    def test_default_config(self):
        """Default config should be valid."""
        config = RiskTriggerEvalConfig(
            risk_pack_path="test.csv",
            target_months=["2026-01"],
        )
        
        assert config.negative_threshold == 0.6
        assert config.spike_threshold == 0.7
        assert config.delta_supply_threshold == 0.6


class TestRiskTriggerEvaluator:
    """Tests for RiskTriggerEvaluator."""

    @pytest.fixture
    def test_data(self, tmp_path):
        """Create test data."""
        return _create_test_data_for_trigger_eval(tmp_path, n_hours=100)

    def test_evaluate_with_y_true(self, test_data, tmp_path):
        """Should evaluate triggers with y_true."""
        risk_pack_file, data_file = test_data
        
        config = RiskTriggerEvalConfig(
            risk_pack_path=risk_pack_file,
            data_path=data_file,
            target_months=["2026-01"],
            out_dir=str(tmp_path / "output"),
        )
        
        evaluator = RiskTriggerEvaluator()
        result = evaluator.evaluate(config)
        
        assert result is not None
        assert result.config == config
        assert result.summary is not None
        assert result.monthly is not None
        assert result.threshold_sweep is not None

    def test_negative_evaluation(self, test_data, tmp_path):
        """Should evaluate negative alerts."""
        risk_pack_file, data_file = test_data
        
        config = RiskTriggerEvalConfig(
            risk_pack_path=risk_pack_file,
            data_path=data_file,
            target_months=["2026-01"],
            out_dir=str(tmp_path / "output"),
        )
        
        evaluator = RiskTriggerEvaluator()
        result = evaluator.evaluate(config)
        
        # Check negative evaluation
        neg_eval = result.summary.get("negative", {})
        assert "precision" in neg_eval
        assert "recall" in neg_eval
        assert "f1" in neg_eval

    def test_spike_evaluation(self, test_data, tmp_path):
        """Should evaluate spike alerts."""
        risk_pack_file, data_file = test_data
        
        config = RiskTriggerEvalConfig(
            risk_pack_path=risk_pack_file,
            data_path=data_file,
            target_months=["2026-01"],
            out_dir=str(tmp_path / "output"),
        )
        
        evaluator = RiskTriggerEvaluator()
        result = evaluator.evaluate(config)
        
        # Check spike evaluation
        spike_eval = result.summary.get("spike", {})
        assert "precision" in spike_eval
        assert "recall" in spike_eval
        assert "f1" in spike_eval

    def test_export_results(self, test_data, tmp_path):
        """Should export evaluation results."""
        risk_pack_file, data_file = test_data
        
        out_dir = tmp_path / "output"
        
        config = RiskTriggerEvalConfig(
            risk_pack_path=risk_pack_file,
            data_path=data_file,
            target_months=["2026-01"],
            out_dir=str(out_dir),
        )
        
        evaluator = RiskTriggerEvaluator()
        result = evaluator.evaluate(config)
        
        # Export results
        evaluator.export_results(result, out_dir=out_dir)
        
        # Check files
        assert (out_dir / "trigger_eval_summary.json").exists()
        assert (out_dir / "trigger_eval_monthly.csv").exists()
        assert (out_dir / "trigger_eval_threshold_sweep.csv").exists()
        assert (out_dir / "trigger_eval_report.md").exists()

    def test_no_data_path(self, test_data, tmp_path):
        """Should handle missing data_path (no y_true)."""
        risk_pack_file, _ = test_data
        
        config = RiskTriggerEvalConfig(
            risk_pack_path=risk_pack_file,
            target_months=["2026-01"],
            out_dir=str(tmp_path / "output"),
        )
        
        evaluator = RiskTriggerEvaluator()
        result = evaluator.evaluate(config)
        
        # Should still work, but summary will have errors
        assert result is not None
        assert result.summary.get("negative", {}).get("error") is not None
