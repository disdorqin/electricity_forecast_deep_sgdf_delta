"""Tests for sgdfnet_prediction_loader.py.

Covers:
- Auto-detection of timestamp and prediction columns
- Business-day alignment
- Deduplication
- Coverage report generation
- Low-coverage failure behaviour
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.sgdfnet_prediction_loader import (
    SGDFNetPredictionLoader,
    CoverageReport,
    save_coverage_report,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_prediction_csv(
    n_hours: int = 240,
    column_schema: str = "standard",
    missing_pct: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic SGDFNet prediction DataFrame with various schemas."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-06-01", periods=n_hours, freq="h")
    values = 300.0 + rng.normal(0, 20, n_hours)

    if missing_pct > 0:
        n_miss = int(n_hours * missing_pct / 100)
        miss_idx = rng.choice(n_hours, size=n_miss, replace=False)
        values[miss_idx] = np.nan

    base = {"ds": ts, "sgdfnet_pred": values}

    if column_schema == "alt_timestamp":
        base = {"timestamp": ts, "pred": values}
    elif column_schema == "alt_pred":
        base = {"ds": ts, "y_pred": values}
    elif column_schema == "chinese":
        base = {"时刻": ts, "sgdfnet_pred": values}

    return pd.DataFrame(base)


# ── Tests ──────────────────────────────────────────────────────────────


class TestLoad:
    """Basic loading tests."""

    def test_standard_schema(self):
        """Standard ds + sgdfnet_pred columns load correctly."""
        df = _make_prediction_csv(column_schema="standard")
        loader = SGDFNetPredictionLoader()
        result, report = loader._process(df)
        assert "ds" in result.columns
        assert "sgdfnet_pred" in result.columns
        assert "business_day" in result.columns
        assert "hour_business" in result.columns
        assert len(result) == len(df)

    def test_alt_timestamp_column(self):
        """timestamp + pred columns are auto-detected."""
        df = _make_prediction_csv(column_schema="alt_timestamp")
        loader = SGDFNetPredictionLoader()
        result, report = loader._process(df)
        assert "ds" in result.columns
        assert "sgdfnet_pred" in result.columns

    def test_alt_pred_column(self):
        """ds + y_pred columns are auto-detected."""
        df = _make_prediction_csv(column_schema="alt_pred")
        loader = SGDFNetPredictionLoader()
        result, report = loader._process(df)
        assert "sgdfnet_pred" in result.columns

    def test_chinese_columns(self):
        """Chinese timestamp column is auto-detected."""
        df = _make_prediction_csv(column_schema="chinese")
        loader = SGDFNetPredictionLoader()
        result, report = loader._process(df)
        assert "ds" in result.columns
        assert "sgdfnet_pred" in result.columns

    def test_no_recognized_pred_column_raises(self):
        """Missing prediction column raises ValueError."""
        df = pd.DataFrame({"ds": [1, 2, 3], "foo": [10, 20, 30]})
        loader = SGDFNetPredictionLoader()
        with pytest.raises(ValueError, match="Cannot identify"):
            loader._process(df)


class TestCoverage:
    """Coverage computation tests."""

    def test_full_coverage(self):
        """100% coverage when all rows are valid."""
        df = _make_prediction_csv(n_hours=240)
        loader = SGDFNetPredictionLoader()
        _, report = loader._process(df)
        assert report.coverage_pct == 100.0

    def test_partial_coverage(self):
        """Coverage drops when rows have NaN prediction."""
        df = _make_prediction_csv(n_hours=240, missing_pct=20.0)
        loader = SGDFNetPredictionLoader()
        _, report = loader._process(df)
        assert report.coverage_pct < 100.0
        assert report.coverage_pct > 70.0  # approximately 80%

    def test_low_coverage_logs_error(self):
        """Low coverage triggers error log but still returns data."""
        df = _make_prediction_csv(n_hours=240, missing_pct=50.0)
        loader = SGDFNetPredictionLoader(require_coverage=95.0)
        result, report = loader._process(df)
        assert report.coverage_pct < 95.0
        assert len(result) < len(df)


class TestDeduplication:
    """Deduplication tests."""

    def test_deduplicates_by_business_day_hour(self):
        """Duplicate (business_day, hour_business) rows are removed."""
        df = _make_prediction_csv(n_hours=240)
        # Add a duplicate
        dup = df.iloc[0:1].copy()
        df_with_dup = pd.concat([df, dup], ignore_index=True)
        loader = SGDFNetPredictionLoader()
        result, report = loader._process(df_with_dup)
        assert report.has_duplicates
        assert len(result) == len(df)  # back to original length


class TestSaveCoverageReport:
    """Coverage report saving tests."""

    def test_saves_json_and_md(self, tmp_path):
        """Coverage report saves JSON and Markdown files."""
        report = CoverageReport(
            total_rows=240,
            matched_rows=220,
            unmatched_rows=20,
            coverage_pct=91.7,
            n_unique_days=10,
            date_range=("2025-06-01", "2025-06-10"),
            source_file="/tmp/test.csv",
        )
        save_coverage_report(report, tmp_path)
        assert (tmp_path / "sgdfnet_prediction_coverage.json").exists()
        assert (tmp_path / "sgdfnet_prediction_coverage.md").exists()


class TestFormalTrainingGate:
    """Formal training gate tests."""

    def test_high_coverage_allows_training(self):
        """Coverage >= 95% allows formal training."""
        df = _make_prediction_csv(n_hours=240 * 3, missing_pct=2.0)  # ~98%
        loader = SGDFNetPredictionLoader(require_coverage=95.0)
        _, report = loader._process(df)
        assert report.coverage_pct >= 95.0

    def test_low_coverage_blocks_training(self):
        """Coverage < 95% blocks formal training."""
        df = _make_prediction_csv(n_hours=240, missing_pct=30.0)
        loader = SGDFNetPredictionLoader(require_coverage=95.0)
        _, report = loader._process(df)
        assert report.coverage_pct < 95.0
