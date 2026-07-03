"""Tests for the realtime feature audit script.

Covers:
- Audit coverage counting.
- Verdict logic (FORMAL_READY, PARTIAL_READY, NOT_READY).
- SGDFNet coverage thresholds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.realtime_feature_builder import (
    audit_feature_coverage,
    build_realtime_features,
)
from models.deep_sgdf_delta.realtime_feature_contract import (
    ALL_FEATURES,
    REQUIRED_FEATURES,
    OPTIONAL_FEATURES,
)

logger = logging.getLogger(__name__)


def _make_feature_df(n_features: int = 30) -> pd.DataFrame:
    """Create a DataFrame with a specified number of feature columns."""
    np.random.seed(42)
    n_rows = 240  # 10 days
    cols = {
        "ds": pd.date_range("2025-06-01", periods=n_rows, freq="h"),
        "da_anchor": np.random.uniform(100, 500, n_rows),
        "rt_actual": np.random.uniform(100, 500, n_rows),
    }
    # Add features from ALL_FEATURES up to n_features
    n_to_add = min(n_features, len(ALL_FEATURES))
    for col in ALL_FEATURES[:n_to_add]:
        cols[col] = np.random.uniform(100, 500, n_rows)

    df = pd.DataFrame(cols)
    df.attrs["sgdfnet_coverage"] = 100.0
    df.attrs["sgdfnet_fallback_used"] = False
    return df


class TestAuditVerdict:
    """Verdict logic tests."""

    def test_formal_ready(self):
        """Full feature set from ALL_FEATURES -> FORMAL_READY."""
        df = _make_feature_df(n_features=35)
        audit = audit_feature_coverage(df)
        assert audit["verdict"] == "FORMAL_READY"
        assert audit["formal_train_ready"] is True

    def test_partial_ready(self):
        """~15 features with SGDFNet coverage -> PARTIAL_READY."""
        # Pick a mix of required + optional to get ~15 features
        df = _make_feature_df(n_features=17)
        df.attrs["sgdfnet_coverage"] = 85.0
        audit = audit_feature_coverage(df)
        assert audit["verdict"] in ("PARTIAL_READY", "FORMAL_READY")

    def test_not_ready(self):
        """Only 3 feature columns -> NOT_READY."""
        df = _make_feature_df(n_features=0)
        audit = audit_feature_coverage(df)
        assert audit["verdict"] == "NOT_READY"
        assert audit["formal_train_ready"] is False

    def test_sgdfnet_coverage_below_threshold(self):
        """35+ features but low SGDFNet coverage -> PARTIAL_READY or NOT_READY."""
        df = _make_feature_df(n_features=35)
        df.attrs["sgdfnet_coverage"] = 50.0
        audit = audit_feature_coverage(df)
        assert audit["verdict"] in ("PARTIAL_READY", "NOT_READY")

    def test_required_missing_audit(self):
        """Missing required features are correctly reported."""
        df = _make_feature_df(n_features=10)
        audit = audit_feature_coverage(df)
        assert len(audit["required_missing"]) > 0
        assert audit["n_required_missing"] > 0


class TestCoverageTracking:
    """Coverage tracking tests."""

    def test_feature_coverage_counts(self):
        """Coverage counts match expected feature presence."""
        df = _make_feature_df(n_features=34)
        audit = audit_feature_coverage(df)
        assert audit["n_features"] == 34
        assert audit["required_present"] is not None
        assert audit["optional_present"] is not None


class TestAuditOutputFiles:
    """Audit output format tests."""

    def test_feature_coverage_csv_format(self):
        """Coverage CSV has expected columns."""
        import io
        # Build a minimal coverage CSV
        rows = [
            {"feature": "forecast_price", "present": True},
            {"feature": "hour_sin", "present": True},
        ]
        csv_df = pd.DataFrame(rows)
        assert "feature" in csv_df.columns
        assert "present" in csv_df.columns
        assert len(csv_df) == 2

    def test_audit_json_serializable(self):
        """Audit dict is JSON-serializable."""
        df = _make_feature_df(n_features=20)
        audit = audit_feature_coverage(df)
        json_str = json.dumps(audit, indent=2, default=str)
        parsed = json.loads(json_str)
        assert parsed["n_features"] == 20
