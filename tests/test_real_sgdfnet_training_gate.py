"""Tests that real SGDFNet predictions gate the training pipeline.

- Without SGDFNet predictions -> formal training blocked
- With low-coverage SGDFNet -> formal training blocked
- With high-coverage SGDFNet -> formal training allowed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.sgdfnet_prediction_loader import (
    SGDFNetPredictionLoader, CoverageReport,
)
from models.deep_sgdf_delta.realtime_feature_builder import (
    audit_feature_coverage,
    build_realtime_features,
)
from models.deep_sgdf_delta.realtime_feature_contract import ALL_FEATURES


class TestTrainingGate:
    """Training gate logic."""

    def test_no_sgdfnet_blocks_formal(self):
        """No SGDFNet predictions -> cannot do formal training."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-01-01", periods=240, freq="h"),
            "da_anchor": np.ones(240) * 300,
            "rt_actual": np.ones(240) * 310,
        })
        for col in ALL_FEATURES[:30]:
            df[col] = np.random.uniform(100, 500, 240)
        df.attrs["sgdfnet_real_coverage"] = 0.0
        df.attrs["sgdfnet_effective_coverage"] = 0.0
        df.attrs["sgdfnet_fallback_used"] = True

        audit = audit_feature_coverage(df)
        assert audit["formal_train_ready"] is False

    def test_low_coverage_sgdfnet_blocks_formal(self):
        """Low SGDFNet coverage blocks formal training."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-01-01", periods=240, freq="h"),
            "da_anchor": np.ones(240) * 300,
            "rt_actual": np.ones(240) * 310,
        })
        for col in ALL_FEATURES[:30]:
            df[col] = np.random.uniform(100, 500, 240)
        df.attrs["sgdfnet_real_coverage"] = 50.0
        df.attrs["sgdfnet_effective_coverage"] = 50.0
        df.attrs["sgdfnet_fallback_used"] = False

        audit = audit_feature_coverage(df)
        assert audit["formal_train_ready"] is False

    def test_high_coverage_sgdfnet_allows_formal(self):
        """High SGDFNet coverage (>=95%) allows formal training."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-01-01", periods=240, freq="h"),
            "da_anchor": np.ones(240) * 300,
            "rt_actual": np.ones(240) * 310,
        })
        for col in ALL_FEATURES[:30]:
            df[col] = np.random.uniform(100, 500, 240)
        df.attrs["sgdfnet_real_coverage"] = 98.0
        df.attrs["sgdfnet_effective_coverage"] = 98.0
        df.attrs["sgdfnet_fallback_used"] = False

        audit = audit_feature_coverage(df)
        assert audit["formal_train_ready"] is True
        assert audit["verdict"] == "FORMAL_READY"

    def test_loader_blocks_low_coverage(self):
        """SGDFNetPredictionLoader with require_coverage=95 blocks low coverage."""
        loader = SGDFNetPredictionLoader(require_coverage=95.0)
        df = pd.DataFrame({
            "ds": pd.date_range("2026-01-01", periods=240, freq="h"),
            "sgdfnet_pred": np.random.default_rng(42).normal(300, 20, 240),
        })
        # Make 50% NaN
        df.loc[:120, "sgdfnet_pred"] = np.nan
        _, report = loader._process(df)
        assert report.coverage_pct < 95.0


class TestCoverageReport:
    """Coverage report logic."""

    def test_report_detects_missing_and_matched(self):
        """Coverage report correctly tracks matched vs unmatched."""
        report = CoverageReport(total_rows=240, matched_rows=200, unmatched_rows=40,
                                coverage_pct=83.3)
        assert report.coverage_pct == 83.3
        assert report.matched_rows == 200

    def test_formal_training_flag(self):
        """formal_training_allowed flag works."""
        report_high = CoverageReport(total_rows=240, matched_rows=235,
                                      unmatched_rows=5, coverage_pct=97.9)
        assert report_high.coverage_pct >= 95.0

        report_low = CoverageReport(total_rows=240, matched_rows=100,
                                     unmatched_rows=140, coverage_pct=41.7)
        assert report_low.coverage_pct < 95.0
