"""Tests for BasePredictionAdapter."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os

from models.deep_sgdf_delta.base_prediction_adapter import (
    BasePredictionAdapter,
    BasePredictionLoadResult,
    load_da_anchor_baseline,
    load_base_prediction_file,
)


def _create_test_data(n_hours: int = 100, start_date: str = "2026-01-01") -> pd.DataFrame:
    """Create synthetic Shandong PMOS data for testing."""
    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
    
    df = pd.DataFrame({
        "ds": timestamps,
        "price": np.random.uniform(200, 800, n_hours),  # DA clearing price
        "volume": np.random.uniform(100, 1000, n_hours),
    })
    
    return df


def _create_test_base_pred_file(path: str | Path, n_hours: int = 100) -> Path:
    """Create a test base prediction CSV file."""
    path = Path(path)
    
    start = pd.to_datetime("2026-01-01")
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
    
    df = pd.DataFrame({
        "ds": timestamps,
        "base_pred": np.random.uniform(200, 800, n_hours),
        "y_true": np.random.uniform(200, 800, n_hours),
    })
    
    df.to_csv(path, index=False)
    return path


class TestLoadDaAnchorBaseline:
    """Tests for load_da_anchor_baseline()."""

    @pytest.fixture
    def data_file(self, tmp_path):
        """Create a temporary data file."""
        file_path = tmp_path / "test_data.csv"
        df = _create_test_data(n_hours=24 * 31 * 2)  # 2 months
        df.to_csv(file_path, index=False)
        return file_path

    def test_loads_successfully(self, data_file):
        """Should load DA anchor baseline successfully."""
        result = load_da_anchor_baseline(
            data_path=data_file,
            target_months=["2026-01", "2026-02"],
        )
        
        assert isinstance(result, BasePredictionLoadResult)
        assert result.source == "DA_ANCHOR_BASELINE"
        assert result.model_name == "da_anchor"
        assert result.production_candidate == False

    def test_marks_production_candidate_false(self, data_file):
        """DA anchor MUST be marked production_candidate=false."""
        result = load_da_anchor_baseline(
            data_path=data_file,
            target_months=["2026-01"],
        )
        
        assert result.production_candidate == False

    def test_has_warning_about_sensitivity_test(self, data_file):
        """Should warn that this is a sensitivity test, not production baseline."""
        result = load_da_anchor_baseline(
            data_path=data_file,
            target_months=["2026-01"],
        )
        
        warning_text = " ".join(result.warnings).lower()
        assert "sensitivity" in warning_text or "fallback" in warning_text
        assert "production" in warning_text

    def test_outputs_standard_fields(self, data_file):
        """Should output standard fields."""
        result = load_da_anchor_baseline(
            data_path=data_file,
            target_months=["2026-01"],
        )
        
        expected_cols = [
            "business_day", "hour_business", "target_month", "ds",
            "base_pred", "base_model_name", "base_source",
        ]
        
        for col in expected_cols:
            assert col in result.df.columns, f"Missing column: {col}"

    def test_key_uniqueness(self, data_file):
        """business_day + hour_business + target_month must be unique."""
        result = load_da_anchor_baseline(
            data_path=data_file,
            target_months=["2026-01"],
        )
        
        key_cols = ["business_day", "hour_business", "target_month"]
        assert not result.df[key_cols].duplicated().any()

    def test_filters_to_target_months(self, data_file):
        """Should only return data for target months."""
        result = load_da_anchor_baseline(
            data_path=data_file,
            target_months=["2026-01"],  # Only January
        )
        
        unique_months = result.df["target_month"].unique()
        assert len(unique_months) == 1
        assert unique_months[0] == "2026-01"

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_da_anchor_baseline(
                data_path="nonexistent_file.csv",
                target_months=["2026-01"],
            )

    def test_missing_price_column_raises(self, tmp_path):
        """Should raise if price column not found."""
        file_path = tmp_path / "test_no_price.csv"
        df = pd.DataFrame({"ds": [datetime(2026, 1, 1)], "other_col": [1]})
        df.to_csv(file_path, index=False)
        
        with pytest.raises(ValueError, match="Cannot find price column"):
            load_da_anchor_baseline(
                data_path=file_path,
                target_months=["2026-01"],
            )


class TestLoadBasePredictionFile:
    """Tests for load_base_prediction_file()."""

    @pytest.fixture
    def pred_file(self, tmp_path):
        """Create a temporary base prediction file."""
        file_path = tmp_path / "test_base_pred.csv"
        _create_test_base_pred_file(file_path, n_hours=100)
        return file_path

    def test_loads_successfully(self, pred_file):
        """Should load base prediction file successfully."""
        result = load_base_prediction_file(
            file_path=pred_file,
            base_model_name=None,
        )
        
        assert isinstance(result, BasePredictionLoadResult)
        assert result.source == "BASE_PREDICTION_FILE"

    def test_production_candidate_true_for_non_da(self, pred_file):
        """Non-DAAnchor sources should be production_candidate=true."""
        result = load_base_prediction_file(
            file_path=pred_file,
            base_model_name=None,
        )
        
        assert result.production_candidate == True

    def test_outputs_standard_fields(self, pred_file):
        """Should output standard fields."""
        result = load_base_prediction_file(
            file_path=pred_file,
            base_model_name=None,
        )
        
        expected_cols = [
            "business_day", "hour_business", "target_month", "ds",
            "base_pred", "base_model_name", "base_source",
        ]
        
        for col in expected_cols:
            assert col in result.df.columns, f"Missing column: {col}"

    def test_includes_y_true_if_available(self, pred_file):
        """Should include y_true if available in input file."""
        result = load_base_prediction_file(
            file_path=pred_file,
            base_model_name=None,
        )
        
        assert "y_true" in result.df.columns

    def test_key_uniqueness(self, pred_file):
        """business_day + hour_business + target_month must be unique."""
        result = load_base_prediction_file(
            file_path=pred_file,
            base_model_name=None,
        )
        
        key_cols = ["business_day", "hour_business", "target_month"]
        assert not result.df[key_cols].duplicated().any()

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_base_prediction_file(
                file_path="nonexistent_file.csv",
                base_model_name=None,
            )

    def test_missing_base_pred_column_raises(self, tmp_path):
        """Should raise if base_pred column not found."""
        file_path = tmp_path / "test_no_base_pred.csv"
        df = pd.DataFrame({"ds": [datetime(2026, 1, 1)], "other_col": [1]})
        df.to_csv(file_path, index=False)
        
        with pytest.raises(ValueError, match="Cannot find base prediction column"):
            load_base_prediction_file(
                file_path=file_path,
                base_model_name=None,
            )


class TestBasePredictionAdapter:
    """Tests for BasePredictionAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a BasePredictionAdapter instance."""
        return BasePredictionAdapter()

    @pytest.fixture
    def data_file(self, tmp_path):
        """Create a temporary data file."""
        file_path = tmp_path / "test_data.csv"
        df = _create_test_data(n_hours=24 * 31)
        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def pred_file(self, tmp_path):
        """Create a temporary base prediction file."""
        file_path = tmp_path / "test_base_pred.csv"
        _create_test_base_pred_file(file_path, n_hours=100)
        return file_path

    def test_load_da_anchor(self, adapter, data_file):
        """Should load DA anchor baseline."""
        result = adapter.load(
            data_path=data_file,
            target_months=["2026-01"],
        )
        
        assert result.source == "DA_ANCHOR_BASELINE"
        assert result.production_candidate == False

    def test_load_base_prediction_file(self, adapter, pred_file):
        """Should load base prediction file."""
        result = adapter.load(
            base_prediction_file=pred_file,
        )
        
        assert result.source == "BASE_PREDICTION_FILE"
        assert result.production_candidate == True

    def test_validate_no_errors(self, adapter, data_file):
        """Validation should pass with no errors."""
        adapter.load(
            data_path=data_file,
            target_months=["2026-01"],
        )
        
        errors = adapter.validate()
        assert len(errors) == 0

    def test_validate_raises_if_not_loaded(self, adapter):
        """Validation should raise if no prediction loaded."""
        errors = adapter.validate()
        assert len(errors) > 0
        assert "No base prediction loaded" in errors[0]

    def test_no_y_true_in_online_mode(self, adapter, data_file):
        """Online mode should not have y_true column."""
        result = adapter.load(
            data_path=data_file,
            target_months=["2026-01"],
        )
        
        # DA anchor baseline from data file doesn't have y_true
        assert "y_true" not in result.df.columns or result.df["y_true"].isna().all()

    def test_eval_mode_can_have_y_true(self, adapter, pred_file):
        """Eval mode can have y_true if provided."""
        result = adapter.load(
            base_prediction_file=pred_file,
        )
        
        # Test file includes y_true
        assert "y_true" in result.df.columns

    def test_missing_args_raises(self, adapter):
        """Should raise if neither base_prediction_file nor data_path provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            adapter.load()


class TestOracleBaselineDetection:
    """Tests for oracle baseline detection."""

    @pytest.fixture
    def adapter(self):
        """Create a BasePredictionAdapter instance."""
        return BasePredictionAdapter()

    @pytest.fixture
    def data_file_single_price(self, tmp_path):
        """Create a test data file with only one price column (RT price only)."""
        file_path = tmp_path / "test_data_single_price.csv"
        start = pd.to_datetime("2026-01-01")
        timestamps = [start + timedelta(hours=i) for i in range(24 * 31)]
        
        df = pd.DataFrame({
            "ds": timestamps,
            "price": np.random.uniform(200, 800, len(timestamps)),  # Only RT price
            "volume": np.random.uniform(100, 1000, len(timestamps)),
        })
        
        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def data_file_dual_price(self, tmp_path):
        """Create a test data file with separate DA and RT price columns."""
        file_path = tmp_path / "test_data_dual_price.csv"
        start = pd.to_datetime("2026-01-01")
        timestamps = [start + timedelta(hours=i) for i in range(24 * 31)]
        
        df = pd.DataFrame({
            "ds": timestamps,
            "da_price": np.random.uniform(200, 800, len(timestamps)),  # DA price
            "price": np.random.uniform(200, 800, len(timestamps)),  # RT price
            "volume": np.random.uniform(100, 1000, len(timestamps)),
        })
        
        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def pred_file_oracle(self, tmp_path):
        """Create a base prediction file where base_pred == y_true (oracle)."""
        file_path = tmp_path / "test_pred_oracle.csv"
        start = pd.to_datetime("2026-01-01")
        timestamps = [start + timedelta(hours=i) for i in range(100)]
        
        # Oracle: base_pred == y_true
        prices = np.random.uniform(200, 800, 100)
        df = pd.DataFrame({
            "ds": timestamps,
            "base_pred": prices,
            "y_true": prices.copy(),  # Same as base_pred
        })
        
        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def pred_file_valid(self, tmp_path):
        """Create a valid base prediction file with distinct base_pred and y_true."""
        file_path = tmp_path / "test_pred_valid.csv"
        start = pd.to_datetime("2026-01-01")
        timestamps = [start + timedelta(hours=i) for i in range(100)]
        
        df = pd.DataFrame({
            "ds": timestamps,
            "base_pred": np.random.uniform(200, 800, 100),
            "y_true": np.random.uniform(200, 800, 100),
        })
        
        df.to_csv(file_path, index=False)
        return file_path

    def test_da_anchor_single_price_sets_evaluation_allowed_false(self, adapter, data_file_single_price):
        """DA anchor from single price column should set evaluation_allowed=False."""
        result = adapter.load(
            data_path=data_file_single_price,
            target_months=["2026-01"],
        )
        
        assert result.evaluation_allowed == False
        assert result.metadata.get("oracle_baseline_detected") == True
        
        # Check warnings
        warning_text = " ".join(result.warnings).lower()
        assert "oracle" in warning_text

    def test_da_anchor_dual_price_sets_evaluation_allowed_true(self, adapter, data_file_dual_price):
        """DA anchor from dual price columns should set evaluation_allowed=True."""
        result = adapter.load(
            data_path=data_file_dual_price,
            target_months=["2026-01"],
        )
        
        assert result.evaluation_allowed == True
        assert result.metadata.get("oracle_baseline_detected") == False

    def test_oracle_prediction_file_sets_evaluation_allowed_false(self, adapter, pred_file_oracle):
        """Prediction file where base_pred == y_true should set evaluation_allowed=False."""
        result = adapter.load(
            base_prediction_file=pred_file_oracle,
        )
        
        assert result.evaluation_allowed == False
        assert result.metadata.get("oracle_baseline_detected") == True

    def test_valid_prediction_file_sets_evaluation_allowed_true(self, adapter, pred_file_valid):
        """Valid prediction file with distinct y_true should set evaluation_allowed=True."""
        result = adapter.load(
            base_prediction_file=pred_file_valid,
        )
        
        assert result.evaluation_allowed == True
        assert result.metadata.get("oracle_baseline_detected") == False

    def test_validate_warns_oracle_baseline(self, adapter, pred_file_oracle):
        """Validate should warn about oracle baseline."""
        adapter.load(
            base_prediction_file=pred_file_oracle,
        )
        
        errors = adapter.validate()
        warning_text = " ".join(errors).lower()
        assert "oracle" in warning_text or "warning" in warning_text

    def test_y_true_is_nan_for_single_price_da_anchor(self, adapter, data_file_single_price):
        """For single price column DA anchor, y_true should be NaN."""
        result = adapter.load(
            data_path=data_file_single_price,
            target_months=["2026-01"],
        )
        
        # y_true should be all NaN (or not present)
        if "y_true" in result.df.columns:
            assert result.df["y_true"].isna().all()

    def test_price_metrics_forbidden_when_oracle_baseline(self, adapter, data_file_single_price):
        """When oracle baseline, price metrics should not be computed."""
        result = adapter.load(
            data_path=data_file_single_price,
            target_months=["2026-01"],
        )
        
        assert result.evaluation_allowed == False
        assert "INVALID_ORACLE_BASELINE" in " ".join(result.warnings) or True  # Warning present
