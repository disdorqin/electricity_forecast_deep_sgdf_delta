"""Tests for the SGDFNet bridge module.

These tests verify that sgdfnet_bridge correctly locates, imports, and
re-exports the required SGDFNet symbols.  They rely on the real SGDFNet
source tree sitting in the sibling directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# ── Helpers ────────────────────────────────────────────────────────────

# The well-known sibling SGDFNet path used across the project.
_SIBLING_SGDFNET = (
    Path(__file__).resolve().parent.parent.parent
    / "electricity_forecast_model2.0_exp"
    / "SGDFNet"
)


def _sibling_exists() -> bool:
    return (_SIBLING_SGDFNET / "src" / "sgdfnet").is_dir()


# ── Test 1: find_sgdfnet_root with explicit path ──────────────────────

class TestFindSgdfnetRootExplicit:
    """find_sgdfnet_root should resolve an explicit path argument."""

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_explicit_path(self):
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root

        root = find_sgdfnet_root(_SIBLING_SGDFNET)
        assert root.is_dir()
        assert (root / "src" / "sgdfnet").is_dir()
        assert root == _SIBLING_SGDFNET.resolve()

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_explicit_path_as_string(self):
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root

        root = find_sgdfnet_root(str(_SIBLING_SGDFNET))
        assert root.is_dir()
        assert (root / "src" / "sgdfnet").is_dir()

    def test_explicit_bad_path_raises(self):
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root

        with pytest.raises(FileNotFoundError, match="does not contain src/sgdfnet"):
            find_sgdfnet_root("/nonexistent/path/to/nowhere")

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_explicit_path_without_src_raises(self):
        """A path that exists but lacks src/sgdfnet should raise."""
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root

        # Pass the src directory itself (which contains sgdfnet/ but not src/sgdfnet/)
        src_dir = _SIBLING_SGDFNET / "src"
        with pytest.raises(FileNotFoundError, match="does not contain src/sgdfnet"):
            find_sgdfnet_root(src_dir)


# ── Test 2: find_sgdfnet_root with env variable ──────────────────────

class TestFindSgdfnetRootEnv:
    """find_sgdfnet_root should honour the SGDFNET_ROOT environment variable."""

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_env_variable(self, monkeypatch):
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root

        monkeypatch.setenv("SGDFNET_ROOT", str(_SIBLING_SGDFNET))
        # Also strip --sgdfnet-root from argv if present so the env var is used
        monkeypatch.setattr(sys, "argv", [sys.argv[0]])

        root = find_sgdfnet_root()
        assert root.is_dir()
        assert (root / "src" / "sgdfnet").is_dir()

    def test_env_variable_bad_path_raises(self, monkeypatch):
        from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root

        monkeypatch.setenv("SGDFNET_ROOT", "/nonexistent/sgdfnet")
        monkeypatch.setattr(sys, "argv", [sys.argv[0]])

        with pytest.raises(FileNotFoundError, match="SGDFNET_ROOT"):
            find_sgdfnet_root()


# ── Test 3: bridge imports data_contract symbols ──────────────────────

class TestBridgeImportsDataContract:
    """The bridge should re-export all required data_contract symbols."""

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_bridge_imports_data_contract(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        # Core column constants
        assert hasattr(bridge, "TIMESTAMP_COL")
        assert isinstance(bridge.TIMESTAMP_COL, str)
        assert len(bridge.TIMESTAMP_COL) > 0

        assert hasattr(bridge, "DA_COL")
        assert isinstance(bridge.DA_COL, str)

        assert hasattr(bridge, "RT_COL")
        assert isinstance(bridge.RT_COL, str)

        assert hasattr(bridge, "FORECAST_COLS")
        assert isinstance(bridge.FORECAST_COLS, list)
        assert len(bridge.FORECAST_COLS) > 0

        assert hasattr(bridge, "ACTUAL_COLS")
        assert isinstance(bridge.ACTUAL_COLS, list)

        assert hasattr(bridge, "ACTUAL_TO_FORECAST_MAP")
        assert isinstance(bridge.ACTUAL_TO_FORECAST_MAP, dict)

        assert hasattr(bridge, "REQUIRED_COLUMNS")
        assert isinstance(bridge.REQUIRED_COLUMNS, list)

        # Functions
        assert callable(bridge.load_dataset)
        assert callable(bridge.preprocess_dataframe)
        assert callable(bridge.add_business_time_columns)
        assert callable(bridge.build_feature_manifest)

        # FeatureConfig dataclass
        fc = bridge.FeatureConfig()
        assert hasattr(fc, "include_forecast_columns")

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_data_contract_feature_config_instantiation(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        fc = bridge.FeatureConfig(
            include_forecast_columns=True,
            include_actual_history_columns=False,
            use_visible_actual_history=True,
            include_delta_history_features=True,
            include_tf_moving_average_features=False,
            include_static_group_graph_features=False,
            include_weekly_history_features=False,
            include_forecast_residual_history_features=False,
            include_segment_local_stats=False,
            include_forecast_pressure_interactions=False,
            include_calendar_features=True,
            include_engineered_forecast_features=True,
        )
        assert fc.include_forecast_columns is True
        assert fc.include_actual_history_columns is False

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_data_contract_add_business_time_columns(self):
        import pandas as pd
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-03-15 00:00:00",
                "2026-03-15 01:00:00",
                "2026-03-15 12:00:00",
            ]),
        })
        result = bridge.add_business_time_columns(df, "timestamp")
        # 00:00 on Mar 15 -> business_day = Mar 14, target_hour = 24
        assert result.iloc[0]["business_day"] == pd.Timestamp("2026-03-14")
        assert result.iloc[0]["target_hour"] == 24
        # 01:00 on Mar 15 -> business_day = Mar 15, target_hour = 1
        assert result.iloc[1]["business_day"] == pd.Timestamp("2026-03-15")
        assert result.iloc[1]["target_hour"] == 1


# ── Test 4: bridge imports protocol_b_cutoff symbols ──────────────────

class TestBridgeImportsProtocolB:
    """The bridge should re-export protocol_b_cutoff symbols."""

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_bridge_imports_protocol_b(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        assert callable(bridge.run_protocol_b_cutoff_experiment)
        assert callable(bridge._build_protocol_b_visible_frame)
        assert callable(bridge._build_inference_frame)
        assert callable(bridge.load_protocol_b_cutoff_config)

        # ProtocolBCutoffConfig dataclass
        assert hasattr(bridge, "ProtocolBCutoffConfig")
        assert "experiment_name" in bridge.ProtocolBCutoffConfig.__dataclass_fields__

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_protocol_b_config_dataclass(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        cfg = bridge.ProtocolBCutoffConfig(
            experiment_name="test",
            data_path="/tmp/data.csv",
            output_root="/tmp/out",
            start_day="2026-01-01",
            end_day="2026-01-31",
        )
        assert cfg.experiment_name == "test"
        assert cfg.decision_hour == 15  # default


# ── Test 5: bridge imports metrics symbols ────────────────────────────

class TestBridgeImportsMetrics:
    """The bridge should re-export metrics symbols."""

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_bridge_imports_metrics(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        assert callable(bridge.build_metrics_frame)
        assert callable(bridge.capped_smape)
        assert callable(bridge.smape)
        assert callable(bridge.mae)
        assert callable(bridge.rmse)
        assert callable(bridge.direction_accuracy)
        assert callable(bridge.positive_direction_recall)
        assert callable(bridge.build_segment_metrics)
        assert callable(bridge.build_tail_metrics)

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_metrics_smape_values(self):
        import numpy as np
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        y_true = np.array([100.0, 200.0, 300.0])
        # Perfect prediction
        assert bridge.smape(y_true, y_true) == pytest.approx(0.0, abs=1e-6)
        # Known value: 200 * |150-100| / (100+150) = 40.0
        assert bridge.smape(
            np.array([100.0]), np.array([150.0])
        ) == pytest.approx(40.0, abs=1e-4)

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_metrics_capped_smape_floor(self):
        import numpy as np
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        # Both below floor=50 -> capped to 50, perfect match
        y_true = np.array([10.0])
        y_pred = np.array([20.0])
        result = bridge.capped_smape(y_true, y_pred, floor=50.0)
        # After capping: both = 50 -> |50-50| / (50+50) = 0
        assert result == pytest.approx(0.0, abs=1e-4)


# ── Test 6: bridge imports models symbols ─────────────────────────────

class TestBridgeImportsModels:
    """The bridge should re-export models symbols."""

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_bridge_imports_models(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        assert hasattr(bridge, "DeltaRegressor")
        assert hasattr(bridge, "HGBModelConfig")
        assert hasattr(bridge, "SegmentConditionedDeltaRegressor")

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_hgb_model_config_defaults(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        cfg = bridge.HGBModelConfig()
        assert cfg.loss == "absolute_error"
        assert cfg.learning_rate == 0.05
        assert cfg.max_depth == 6
        assert cfg.random_state == 42


# ── Test 7: bridge __all__ completeness ───────────────────────────────

class TestBridgeAllExports:
    """Verify __all__ lists match actual attributes."""

    @pytest.mark.skipif(not _sibling_exists(), reason="Sibling SGDFNet not found")
    def test_all_symbols_present(self):
        from models.deep_sgdf_delta import sgdfnet_bridge as bridge

        for name in bridge.__all__:
            assert hasattr(bridge, name), f"Symbol {name!r} in __all__ but not on module"
