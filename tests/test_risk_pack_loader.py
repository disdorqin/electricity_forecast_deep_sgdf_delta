"""Tests for RiskPackLoader."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json
import os

from models.deep_sgdf_delta.risk_pack_loader import (
    RiskPackLoader,
    RiskPackLoadResult,
    load_risk_pack,
    PROBABILITY_COLUMNS,
    MODULE_STATUS_COLUMNS,
)


def _create_test_risk_pack(
    path: str | Path,
    n_hours: int = 100,
    start_date: str = "2026-01-01",
    include_y_true: bool = False,
    prob_out_of_range: bool = False,
    all_unknown_status: bool = False,
) -> Path:
    """Create a synthetic risk feature pack for testing."""
    path = Path(path)
    
    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
    
    # Create business time columns
    df = pd.DataFrame({
        "ds": timestamps,
    })
    
    # Add business_day and hour_business
    df["business_day"] = pd.to_datetime(df["ds"].dt.date)
    df["hour_business"] = df["ds"].dt.hour
    df.loc[df["hour_business"] == 0, "hour_business"] = 24
    df.loc[df["hour_business"] == 0, "business_day"] = df["business_day"] - pd.Timedelta(days=1)
    
    # Add target_month
    df["target_month"] = df["business_day"].dt.strftime("%Y-%m")
    
    # Add risk scores (probabilities in [0, 1])
    if prob_out_of_range:
        # Some values outside [0, 1]
        df["negative_prob"] = np.random.uniform(-0.1, 1.2, n_hours)
        df["spike_prob"] = np.random.uniform(-0.1, 1.2, n_hours)
    else:
        df["negative_prob"] = np.random.uniform(0, 1, n_hours)
        df["spike_prob"] = np.random.uniform(0, 1, n_hours)
    
    df["negative_risk_score"] = df["negative_prob"]
    df["spike_risk_score"] = df["spike_prob"]
    df["deviation_down_prob"] = np.random.uniform(0, 1, n_hours)
    df["deviation_up_prob"] = np.random.uniform(0, 1, n_hours)
    df["deviation_risk_score"] = (df["deviation_down_prob"] + df["deviation_up_prob"]) / 2
    
    # Add module status
    if all_unknown_status:
        df["negative_module_status"] = "UNKNOWN"
        df["spike_module_status"] = "UNKNOWN"
        df["delta_supply_module_status"] = "UNKNOWN"
    else:
        df["negative_module_status"] = np.random.choice(["CHAMPION", "ACCEPTABLE", "AUX"], n_hours)
        df["spike_module_status"] = np.random.choice(["CHAMPION", "ACCEPTABLE", "AUX"], n_hours)
        df["delta_supply_module_status"] = np.random.choice(["CHAMPION", "ACCEPTABLE", "AUX"], n_hours)
    
    # Add y_true if requested
    if include_y_true:
        df["y_true"] = np.random.uniform(200, 800, n_hours)
    
    # Select output columns
    output_cols = [
        "business_day", "hour_business", "target_month", "ds",
        "negative_prob", "negative_risk_score",
        "spike_prob", "spike_risk_score",
        "deviation_down_prob", "deviation_up_prob", "deviation_risk_score",
        "negative_module_status", "spike_module_status", "delta_supply_module_status",
    ]
    
    if include_y_true:
        output_cols.append("y_true")
    
    df[output_cols].to_csv(path, index=False)
    
    return path


def _create_test_manifest(
    path: str | Path,
    risk_feature_version: str = "v1.1.0",
    metric_alignment_status: str = "PASS",
    quality_gate_passed: bool = True,
) -> Path:
    """Create a synthetic manifest JSON for testing."""
    path = Path(path)
    
    manifest = {
        "risk_feature_version": risk_feature_version,
        "metric_alignment_status": metric_alignment_status,
        "quality_gate_passed": quality_gate_passed,
        "n_samples": 100,
        "columns": [
            "business_day", "hour_business", "target_month", "ds",
            "negative_prob", "negative_risk_score",
            "spike_prob", "spike_risk_score",
        ],
    }
    
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return path


class TestLoadRiskPack:
    """Tests for load_risk_pack()."""

    @pytest.fixture
    def risk_pack_file(self, tmp_path):
        """Create a temporary risk pack file."""
        file_path = tmp_path / "risk_feature_pack.csv"
        _create_test_risk_pack(file_path, n_hours=100)
        return file_path

    @pytest.fixture
    def manifest_file(self, tmp_path):
        """Create a temporary manifest file."""
        file_path = tmp_path / "manifest.json"
        _create_test_manifest(file_path)
        return file_path

    def test_loads_successfully(self, risk_pack_file, manifest_file):
        """Should load risk pack successfully."""
        result = load_risk_pack(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "SUCCESS"
        assert isinstance(result.df, pd.DataFrame)
        assert isinstance(result.manifest, dict)

    def test_version_starts_with_v1(self, risk_pack_file, manifest_file):
        """risk_feature_version should start with 'v1.'."""
        result = load_risk_pack(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        version = result.manifest.get("risk_feature_version", "")
        assert version.startswith("v1.")

    def test_allows_pass_metric_status(self, risk_pack_file, tmp_path):
        """Should allow metric_alignment_status = PASS."""
        manifest_file = tmp_path / "manifest.json"
        _create_test_manifest(
            path=manifest_file,
            metric_alignment_status="PASS",
        )
        
        result = load_risk_pack(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "SUCCESS"

    def test_allows_warn_metric_status(self, risk_pack_file, tmp_path):
        """Should allow metric_alignment_status = WARN."""
        manifest_file = tmp_path / "manifest.json"
        _create_test_manifest(
            path=manifest_file,
            metric_alignment_status="WARN",
        )
        
        result = load_risk_pack(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "SUCCESS"
        assert any("WARN" in w for w in result.warnings)

    def test_rejects_fail_metric_status(self, risk_pack_file, tmp_path):
        """Should reject metric_alignment_status = FAIL."""
        manifest_file = tmp_path / "manifest.json"
        _create_test_manifest(
            path=manifest_file,
            metric_alignment_status="FAIL",
        )
        
        result = load_risk_pack(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "ERROR"
        assert "FAIL" in result.error_message

    def test_online_mode_rejects_y_true(self, risk_pack_file, manifest_file, tmp_path):
        """Online mode should reject y_true in risk pack."""
        # Create risk pack with y_true
        file_path = tmp_path / "risk_feature_pack_with_y_true.csv"
        _create_test_risk_pack(file_path, n_hours=100, include_y_true=True)
        
        result = load_risk_pack(
            risk_pack_path=file_path,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "ERROR"
        assert "y_true" in result.error_message

    def test_eval_mode_allows_y_true(self, risk_pack_file, manifest_file, tmp_path):
        """Eval mode should allow y_true in risk pack."""
        # Create risk pack with y_true
        file_path = tmp_path / "risk_feature_pack_with_y_true.csv"
        _create_test_risk_pack(file_path, n_hours=100, include_y_true=True)
        
        result = load_risk_pack(
            risk_pack_path=file_path,
            manifest_path=manifest_file,
            online_mode=False,  # Eval mode
        )
        
        assert result.status == "SUCCESS"

    def test_probability_columns_in_range(self, risk_pack_file, manifest_file):
        """Probability columns should be in [0, 1] or NaN."""
        result = load_risk_pack(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        for col in PROBABILITY_COLUMNS:
            if col in result.df.columns:
                invalid_mask = ~result.df[col].between(0, 1) & result.df[col].notna()
                assert not invalid_mask.any(), f"Column '{col}' has values outside [0, 1]"

    def test_warns_probability_out_of_range(self, risk_pack_file, manifest_file, tmp_path):
        """Should warn if probability columns out of range."""
        # Create risk pack with out-of-range probabilities
        file_path = tmp_path / "risk_feature_pack_out_of_range.csv"
        _create_test_risk_pack(file_path, n_hours=100, prob_out_of_range=True)
        
        result = load_risk_pack(
            risk_pack_path=file_path,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "SUCCESS"
        assert any("outside [0, 1]" in w for w in result.warnings)

    def test_warns_all_unknown_status(self, risk_pack_file, manifest_file, tmp_path):
        """Should warn if module_status columns are all UNKNOWN."""
        # Create risk pack with all UNKNOWN status
        file_path = tmp_path / "risk_feature_pack_all_unknown.csv"
        _create_test_risk_pack(file_path, n_hours=100, all_unknown_status=True)
        
        result = load_risk_pack(
            risk_pack_path=file_path,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "SUCCESS"
        assert any("UNKNOWN" in w for w in result.warnings)

    def test_key_uniqueness(self, risk_pack_file, manifest_file):
        """business_day + hour_business + target_month must be unique."""
        result = load_risk_pack(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        key_cols = ["business_day", "hour_business", "target_month"]
        assert not result.df[key_cols].duplicated().any()

    def test_missing_file_returns_error(self, manifest_file):
        """Should return error for missing risk pack file."""
        result = load_risk_pack(
            risk_pack_path="nonexistent_file.csv",
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "ERROR"
        assert "not found" in result.error_message


class TestRiskPackLoader:
    """Tests for RiskPackLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a RiskPackLoader instance."""
        return RiskPackLoader()

    @pytest.fixture
    def risk_pack_file(self, tmp_path):
        """Create a temporary risk pack file."""
        file_path = tmp_path / "risk_feature_pack.csv"
        _create_test_risk_pack(file_path, n_hours=100)
        return file_path

    @pytest.fixture
    def manifest_file(self, tmp_path):
        """Create a temporary manifest file."""
        file_path = tmp_path / "manifest.json"
        _create_test_manifest(file_path)
        return file_path

    def test_load_success(self, loader, risk_pack_file, manifest_file):
        """Should load successfully."""
        result = loader.load(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        assert result.status == "SUCCESS"

    def test_validate_no_errors(self, loader, risk_pack_file, manifest_file):
        """Validation should pass with no errors."""
        loader.load(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        errors = loader.validate()
        assert len(errors) == 0

    def test_validate_raises_if_not_loaded(self, loader):
        """Validation should return errors if not loaded."""
        errors = loader.validate()
        assert len(errors) > 0
        assert "No risk pack loaded" in errors[0]

    def test_get_risk_scores(self, loader, risk_pack_file, manifest_file):
        """Should extract risk scores."""
        loader.load(
            risk_pack_path=risk_pack_file,
            manifest_path=manifest_file,
            online_mode=True,
        )
        
        risk_df = loader.get_risk_scores()
        
        assert isinstance(risk_df, pd.DataFrame)
        assert "negative_prob" in risk_df.columns
        assert "spike_prob" in risk_df.columns

    def test_get_risk_scores_before_load_raises(self, loader):
        """Should raise if not loaded."""
        with pytest.raises(RuntimeError, match="No risk pack loaded"):
            loader.get_risk_scores()
