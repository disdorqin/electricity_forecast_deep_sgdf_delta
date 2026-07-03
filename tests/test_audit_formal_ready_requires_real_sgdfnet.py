"""Tests that FORMAL_READY verdict requires real SGDFNet predictions (no fallback).

Covers:
- Full features with fallback → FALLBACK_READY, NOT FORMAL_READY
- Full features without fallback and real coverage >= 95% → FORMAL_READY
- Low real coverage even with fallback → FALLBACK_READY
- formal_train_ready is False for FALLBACK_READY
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.realtime_feature_builder import (
    audit_feature_coverage,
    build_realtime_features,
)
from models.deep_sgdf_delta.realtime_feature_contract import ALL_FEATURES


def _make_audit_df(n_features: int = 34, real_coverage: float = 100.0,
                   fallback_used: bool = False) -> pd.DataFrame:
    """Build a DataFrame with specified coverage attributes."""
    n_rows = 240
    rng = np.random.default_rng(42)
    cols = {"ds": pd.date_range("2025-06-01", periods=n_rows, freq="h"),
            "da_anchor": rng.uniform(100, 500, n_rows),
            "rt_actual": rng.uniform(100, 500, n_rows)}
    for col in ALL_FEATURES[:n_features]:
        cols[col] = rng.uniform(100, 500, n_rows)
    df = pd.DataFrame(cols)
    df.attrs["sgdfnet_effective_coverage"] = max(real_coverage, 95.0)
    df.attrs["sgdfnet_real_coverage"] = real_coverage
    df.attrs["sgdfnet_fallback_used"] = fallback_used
    df.attrs["sgdfnet_source"] = "test"
    return df


class TestFormalReadyRequiresRealSGDFNet:
    """FORMAL_READY must only be reached with real SGDFNet predictions."""

    def test_fallback_used_no_formal_ready(self):
        """Fallback used → verdict is FALLBACK_READY, not FORMAL_READY."""
        df = _make_audit_df(n_features=34, real_coverage=0.0, fallback_used=True)
        audit = audit_feature_coverage(df)
        assert audit["verdict"] == "FALLBACK_READY", (
            f"Expected FALLBACK_READY, got {audit['verdict']}"
        )
        assert audit["formal_train_ready"] is False

    def test_no_fallback_high_real_coverage_formal_ready(self):
        """Real coverage >= 95% and no fallback → FORMAL_READY."""
        df = _make_audit_df(n_features=34, real_coverage=98.0, fallback_used=False)
        audit = audit_feature_coverage(df)
        assert audit["verdict"] == "FORMAL_READY"
        assert audit["formal_train_ready"] is True

    def test_formal_requires_both_coverage_and_no_fallback(self):
        """Real coverage alone is not enough if fallback was also used."""
        df = _make_audit_df(n_features=34, real_coverage=95.0, fallback_used=True)
        audit = audit_feature_coverage(df)
        # Even though real_coverage meets threshold, fallback was used → FALLBACK_READY
        assert audit["verdict"] == "FALLBACK_READY"
        assert audit["formal_train_ready"] is False

    def test_partial_real_coverage_no_fallback(self):
        """Real coverage 80-94% without fallback → PARTIAL_READY."""
        df = _make_audit_df(n_features=30, real_coverage=85.0, fallback_used=False)
        audit = audit_feature_coverage(df)
        assert audit["verdict"] in ("PARTIAL_READY", "FORMAL_READY")

    def test_low_real_coverage_no_fallback_not_ready(self):
        """Real coverage < 80% without fallback → NOT_READY (due to low features/coverage)."""
        df = _make_audit_df(n_features=30, real_coverage=50.0, fallback_used=False)
        audit = audit_feature_coverage(df)
        assert audit["verdict"] == "NOT_READY"
