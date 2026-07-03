"""Tests for SGDFNet fallback logic in realtime_dataset_final.py.

Covers:
- Formal training without sgdfnet_pred raises ValueError.
- fast-dev / predict mode allows fallback.
- Manifest records fallback_used and coverage.
- Real sgdfnet_pred present does not trigger fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.realtime_dataset_final import (
    _ensure_sgdfnet_pred,
    build_training_datasets_final,
    build_predict_dataset_final,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_frame(n_days: int = 50, has_sgdfnet: bool = True,
                start_date: str = "2025-06-01") -> pd.DataFrame:
    """Build a minimal hourly DataFrame for testing."""
    ts = pd.date_range(start=start_date, periods=n_days * 24, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "ds": ts,
        "da_anchor": np.round(200 + 50 * np.sin(2 * np.pi * ts.hour / 24), 2),
        "rt_actual": np.round(200 + 50 * np.sin(2 * np.pi * ts.hour / 24)
                                + rng.normal(0, 10, size=len(ts)), 2),
    })
    if has_sgdfnet:
        df["sgdfnet_pred"] = np.round(
            df["rt_actual"].values + rng.normal(0, 3, size=len(df)), 2
        )
    return df


# ── Tests ──────────────────────────────────────────────────────────────


class TestEnsureSGDFNetPred:
    """Unit tests for _ensure_sgdfnet_pred."""

    def test_formal_training_missing_raises(self):
        """Missing sgdfnet_pred raises ValueError when allow_fallback=False."""
        df = _make_frame(has_sgdfnet=False)
        with pytest.raises(ValueError, match="Missing sgdfnet_pred"):
            _ensure_sgdfnet_pred(df, allow_fallback=False)

    def test_predict_mode_allows_fallback(self):
        """Missing sgdfnet_pred with allow_fallback=True fills from da_anchor."""
        df = _make_frame(has_sgdfnet=False)
        result = _ensure_sgdfnet_pred(df, allow_fallback=True)
        assert "sgdfnet_pred" in result.columns
        np.testing.assert_array_almost_equal(
            result["sgdfnet_pred"].values, result["da_anchor"].values,
        )
        assert result.attrs.get("sgdfnet_fallback_used") is True

    def test_real_predictions_no_fallback(self):
        """Real sgdfnet_pred present — no fallback even with allow_fallback=True."""
        df = _make_frame(has_sgdfnet=True)
        result = _ensure_sgdfnet_pred(df, allow_fallback=False)
        assert "sgdfnet_pred" in result.columns
        assert result.attrs.get("sgdfnet_fallback_used") is False

    def test_nan_in_sgdfnet_raises_no_fallback(self):
        """NaN values in sgdfnet_pred raise error when allow_fallback=False."""
        df = _make_frame(has_sgdfnet=True)
        df.loc[df.index[:10], "sgdfnet_pred"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            _ensure_sgdfnet_pred(df, allow_fallback=False)

    def test_nan_in_sgdfnet_fallback_allowed(self):
        """NaN values in sgdfnet_pred are filled from da_anchor with fallback."""
        df = _make_frame(has_sgdfnet=True)
        df.loc[df.index[:10], "sgdfnet_pred"] = np.nan
        result = _ensure_sgdfnet_pred(df, allow_fallback=True)
        assert result.attrs.get("sgdfnet_fallback_used") is True
        assert result["sgdfnet_pred"].iloc[0] == result["da_anchor"].iloc[0]

    def test_coverage_tracked(self):
        """Coverage percentage is tracked in attrs."""
        df = _make_frame(has_sgdfnet=False)
        result = _ensure_sgdfnet_pred(df, allow_fallback=True)
        assert "sgdfnet_coverage" in result.attrs
        # All rows fell back, so coverage is 0
        assert result.attrs["sgdfnet_coverage"] == 0.0

        df2 = _make_frame(has_sgdfnet=True)
        result2 = _ensure_sgdfnet_pred(df2, allow_fallback=False)
        assert result2.attrs["sgdfnet_coverage"] == 100.0


class TestBuildTrainingDatasetsFallback:
    """Integration tests for build_training_datasets_final fallback."""

    def test_formal_training_missing_sgdfnet_raises(self):
        """build_training_datasets_final raises when sgdfnet_pred missing."""
        df = _make_frame(n_days=200, has_sgdfnet=False)
        with pytest.raises(ValueError, match="Missing sgdfnet_pred"):
            build_training_datasets_final(
                df,
                target_month="2025-11",
                train_min_days=10,
                val_days=5,
                allow_sgdfnet_fallback=False,
            )

    def test_fast_dev_allows_fallback(self):
        """build_training_datasets_final with allow_sgdfnet_fallback=True works."""
        df = _make_frame(n_days=200, has_sgdfnet=False)
        train_ds, val_ds, test_ds, manifest = build_training_datasets_final(
            df,
            target_month="2025-11",
            train_min_days=10,
            val_days=5,
            allow_sgdfnet_fallback=True,
        )
        assert manifest.get("sgdfnet_fallback_used") is True

    def test_manifest_records_coverage(self):
        """Manifest includes sgdfnet_coverage and sgdfnet_fallback_used."""
        df = _make_frame(n_days=200, has_sgdfnet=True)
        _, _, _, manifest = build_training_datasets_final(
            df,
            target_month="2025-11",
            train_min_days=10,
            val_days=5,
            allow_sgdfnet_fallback=False,
        )
        assert "sgdfnet_coverage" in manifest
        assert "sgdfnet_fallback_used" in manifest
        assert manifest["sgdfnet_fallback_used"] is False
        assert manifest["sgdfnet_coverage"] == 100.0
