"""Tests for models.deep_sgdf_delta.realtime_feature_contract.

Covers:
  - validate_features: required feature presence / missing detection
  - check_leakage: safe vs unsafe DataFrames
  - build_feature_manifest: structure and field correctness
  - get_period: hour-to-period mapping and invalid input handling
"""
from __future__ import annotations

import pytest
import pandas as pd

from models.deep_sgdf_delta.realtime_feature_contract import (
    REQUIRED_FEATURES,
    OPTIONAL_FEATURES,
    ALL_FEATURES,
    FEATURE_VERSION,
    validate_features,
    check_leakage,
    build_feature_manifest,
    get_period,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_complete_df() -> pd.DataFrame:
    """Return a DataFrame that contains ALL required features."""
    data = {feat: [0.0] for feat in REQUIRED_FEATURES}
    return pd.DataFrame(data)


def _make_partial_df() -> pd.DataFrame:
    """Return a DataFrame missing several required features."""
    # Keep only the first 3 required features
    present = REQUIRED_FEATURES[:3]
    data = {feat: [0.0] for feat in present}
    return pd.DataFrame(data)


# ── validate_features ─────────────────────────────────────────────────

class TestValidateFeatures:
    def test_validate_features_all_present(self):
        """All required features present -> empty missing list."""
        df = _make_complete_df()
        missing = validate_features(df)
        assert missing == [], f"Expected no missing features, got {missing}"

    def test_validate_features_missing(self):
        """Some features missing -> correct missing list returned."""
        df = _make_partial_df()
        missing = validate_features(df)
        expected_missing = REQUIRED_FEATURES[3:]
        assert missing == expected_missing
        assert len(missing) == len(REQUIRED_FEATURES) - 3

    def test_validate_features_optional_ignored(self):
        """Optional features are NOT flagged as missing."""
        # DataFrame with only required features (no optional)
        df = _make_complete_df()
        missing = validate_features(df)
        assert missing == []
        # Optional features should not appear in missing even if absent
        for opt in OPTIONAL_FEATURES:
            assert opt not in missing


# ── check_leakage ─────────────────────────────────────────────────────

class TestCheckLeakage:
    def test_check_leakage_safe(self):
        """No leakage columns -> True."""
        df = pd.DataFrame({
            "forecast_price": [300.0],
            "hour_sin": [0.1],
            "rt_lag_1h": [290.0],
        })
        assert check_leakage(df, cutoff_hour=15) is True

    def test_check_leakage_unsafe(self):
        """Column with 'actual' for future hours (> cutoff) -> False."""
        df = pd.DataFrame({
            "forecast_price": [300.0],
            "rt_actual_h16": [310.0],  # hour 16 > cutoff 15
            "rt_actual_h20": [320.0],  # hour 20 > cutoff 15
        })
        assert check_leakage(df, cutoff_hour=15) is False

    def test_check_leakage_safe_with_past_actual(self):
        """Column with 'actual' for hours <= cutoff -> True."""
        df = pd.DataFrame({
            "forecast_price": [300.0],
            "rt_actual_h10": [295.0],  # hour 10 <= cutoff 15
            "rt_actual_h15": [298.0],  # hour 15 == cutoff (allowed)
        })
        assert check_leakage(df, cutoff_hour=15) is True

    def test_check_leakage_no_hour_suffix(self):
        """Column with 'actual' but no hour suffix -> skipped (safe)."""
        df = pd.DataFrame({
            "rt_actual": [300.0],  # generic actual, no hour suffix
            "forecast_price": [300.0],
        })
        # No explicit hour encoded, so it passes
        assert check_leakage(df, cutoff_hour=15) is True

    def test_check_leakage_target_hour_violation(self):
        """Column with actual_hX where X >= target_hour -> False."""
        df = pd.DataFrame({
            "rt_actual_h10": [295.0],
            "target_hour": [10],  # col_hour (10) >= target_hour (10)
        })
        assert check_leakage(df, cutoff_hour=15) is False


# ── build_feature_manifest ────────────────────────────────────────────

class TestBuildFeatureManifest:
    def test_build_feature_manifest(self):
        """Correct structure and fields."""
        feature_cols = ["forecast_price", "hour_sin", "hour_cos"]
        target_cols = ["delta_target", "residual_target"]

        manifest = build_feature_manifest(
            feature_columns=feature_cols,
            target_columns=target_cols,
            date_range=("2024-01-01", "2024-06-30"),
            n_days=180,
        )

        # Check all expected keys exist
        assert "feature_version" in manifest
        assert "feature_columns" in manifest
        assert "target_columns" in manifest
        assert "n_features" in manifest
        assert "n_targets" in manifest
        assert "date_range" in manifest
        assert "n_days" in manifest
        assert "leakage_checks_passed" in manifest

        # Check values
        assert manifest["feature_version"] == FEATURE_VERSION
        assert manifest["feature_columns"] == feature_cols
        assert manifest["target_columns"] == target_cols
        assert manifest["n_features"] == 3
        assert manifest["n_targets"] == 2
        assert manifest["n_days"] == 180
        assert manifest["leakage_checks_passed"] is True

        # Check date_range structure
        assert manifest["date_range"] is not None
        assert "start" in manifest["date_range"]
        assert "end" in manifest["date_range"]
        assert manifest["date_range"]["start"] == "2024-01-01"
        assert manifest["date_range"]["end"] == "2024-06-30"

    def test_build_feature_manifest_no_date_range(self):
        """date_range=None -> date_range is None in manifest."""
        manifest = build_feature_manifest(
            feature_columns=["forecast_price"],
            target_columns=["delta_target"],
            date_range=None,
            n_days=None,
        )
        assert manifest["date_range"] is None
        assert manifest["n_days"] is None

    def test_build_feature_manifest_with_datetime_index(self):
        """DatetimeIndex date_range is converted to ISO strings."""
        idx = pd.date_range("2024-03-01", "2024-03-15")
        manifest = build_feature_manifest(
            feature_columns=["forecast_price"],
            target_columns=["delta_target"],
            date_range=idx,
            n_days=15,
        )
        assert manifest["date_range"]["start"] == "2024-03-01"
        assert manifest["date_range"]["end"] == "2024-03-15"

    def test_build_feature_manifest_diagnostic_breakdown(self):
        """Manifest includes required_present, optional_present, required_missing."""
        feature_cols = ["forecast_price", "hour_sin"]  # subset of required
        manifest = build_feature_manifest(
            feature_columns=feature_cols,
            target_columns=["delta_target"],
        )
        assert "forecast_price" in manifest["required_present"]
        assert "hour_sin" in manifest["required_present"]
        assert len(manifest["required_missing"]) == len(REQUIRED_FEATURES) - 2


# ── get_period ────────────────────────────────────────────────────────

class TestGetPeriod:
    def test_get_period_valley(self):
        """Hours 1-8 -> '1_8'."""
        for h in range(1, 9):
            assert get_period(h) == "1_8", f"Hour {h} should be '1_8'"

    def test_get_period_shoulder(self):
        """Hours 9-16 -> '9_16'."""
        for h in range(9, 17):
            assert get_period(h) == "9_16", f"Hour {h} should be '9_16'"

    def test_get_period_peak(self):
        """Hours 17-24 -> '17_24'."""
        for h in range(17, 25):
            assert get_period(h) == "17_24", f"Hour {h} should be '17_24'"

    def test_get_period_invalid_zero(self):
        """Hour 0 -> ValueError."""
        with pytest.raises(ValueError, match="1-24"):
            get_period(0)

    def test_get_period_invalid_twentyfive(self):
        """Hour 25 -> ValueError."""
        with pytest.raises(ValueError, match="1-24"):
            get_period(25)

    def test_get_period_invalid_negative(self):
        """Negative hour -> ValueError."""
        with pytest.raises(ValueError, match="1-24"):
            get_period(-1)

    def test_get_period_boundary_values(self):
        """Boundary hours 1, 8, 9, 16, 17, 24 map correctly."""
        assert get_period(1) == "1_8"
        assert get_period(8) == "1_8"
        assert get_period(9) == "9_16"
        assert get_period(16) == "9_16"
        assert get_period(17) == "17_24"
        assert get_period(24) == "17_24"
