"""Tests for realtime prediction output contract.

Covers:
  - Prediction output produces 24 rows per day
  - All required columns are present
  - Online output has no y_true (NaN)
  - Hours are in business range 1-24
  - Period labels are correct for each hour
"""
from __future__ import annotations

import math

import pytest
import pandas as pd
import numpy as np

from models.deep_sgdf_delta.output_contract import (
    OUTPUT_COLUMNS,
    EVAL_ONLY_COLUMNS,
    build_prediction_row,
    build_predictions_dataframe,
    strip_eval_columns,
    validate_predictions,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _build_full_day_predictions(
    business_day: str = "2024-06-15",
    da_anchor_base: float = 300.0,
    delta_base: float = 10.0,
    sgdfnet_pred: float = 312.0,
) -> pd.DataFrame:
    """Build a full 24-hour prediction DataFrame for one business day."""
    rows = []
    for hour in range(1, 25):
        da = da_anchor_base + np.random.randn() * 5
        delta = delta_base + np.random.randn() * 3
        row = build_prediction_row(
            business_day=business_day,
            hour=hour,
            da_anchor=da,
            deep_delta_pred=delta,
            sgdfnet_pred=sgdfnet_pred + np.random.randn() * 3,
            blend_mode="sgdfnet_blend",
            blend_weight=0.4,
        )
        rows.append(row)
    return build_predictions_dataframe(rows)


# ── 24 rows per day ───────────────────────────────────────────────────

class TestOutputHas24Rows:
    def test_output_has_24_rows(self):
        """Prediction produces 24 rows per day."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        assert len(df) == 24, f"Expected 24 rows, got {len(df)}"

    def test_output_has_24_rows_multi_day(self):
        """Multiple days produce 24 rows each."""
        np.random.seed(42)
        dfs = []
        for day in ["2024-06-15", "2024-06-16", "2024-06-17"]:
            dfs.append(_build_full_day_predictions(business_day=day))
        combined = pd.concat(dfs, ignore_index=True)
        assert len(combined) == 72

        # Each day should have exactly 24 rows
        for bd in ["2024-06-15", "2024-06-16", "2024-06-17"]:
            day_df = combined[combined["business_day"] == pd.Timestamp(bd)]
            assert len(day_df) == 24


# ── Required columns ──────────────────────────────────────────────────

class TestOutputColumns:
    def test_output_columns(self):
        """All required columns present."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        missing = validate_predictions(df)
        assert missing == [], f"Missing columns: {missing}"

    def test_output_columns_exact_set(self):
        """Output has exactly the OUTPUT_COLUMNS."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        assert set(df.columns) == set(OUTPUT_COLUMNS)


# ── No y_true in online output ────────────────────────────────────────

class TestNoYTrueInOnline:
    def test_no_y_true_in_online(self):
        """Online output has y_true as NaN (not populated)."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        # y_true should be all NaN in prediction mode
        assert df["y_true"].isna().all(), "y_true should be NaN in prediction mode"

    def test_strip_eval_columns_removes_eval_only(self):
        """strip_eval_columns removes EVAL_ONLY_COLUMNS."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        stripped = strip_eval_columns(df)
        for col in EVAL_ONLY_COLUMNS:
            assert col not in stripped.columns, f"Column '{col}' should be stripped"


# ── Hour business range ───────────────────────────────────────────────

class TestHourBusinessRange:
    def test_hour_business_range(self):
        """Hours 1-24."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        hours = df["hour_business"].tolist()
        assert sorted(hours) == list(range(1, 25)), "Hours should be 1-24"

    def test_hour_business_min_max(self):
        """Min hour is 1, max hour is 24."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        assert df["hour_business"].min() == 1
        assert df["hour_business"].max() == 24


# ── Period labels ─────────────────────────────────────────────────────

class TestPeriodLabels:
    def test_period_labels(self):
        """Correct period for each hour."""
        np.random.seed(42)
        df = _build_full_day_predictions()

        # Check period mapping
        for _, row in df.iterrows():
            hour = int(row["hour_business"])
            period = row["period"]

            if 1 <= hour <= 8:
                assert period == "1_8", f"Hour {hour} should be period '1_8', got '{period}'"
            elif 9 <= hour <= 16:
                assert period == "9_16", f"Hour {hour} should be period '9_16', got '{period}'"
            elif 17 <= hour <= 24:
                assert period == "17_24", f"Hour {hour} should be period '17_24', got '{period}'"

    def test_period_labels_distribution(self):
        """Each period has the correct number of hours."""
        np.random.seed(42)
        df = _build_full_day_predictions()

        period_counts = df["period"].value_counts()
        assert period_counts.get("1_8", 0) == 8
        assert period_counts.get("9_16", 0) == 8
        assert period_counts.get("17_24", 0) == 8

    def test_period_labels_all_valid(self):
        """All period labels are in the valid set."""
        np.random.seed(42)
        df = _build_full_day_predictions()
        valid_periods = {"1_8", "9_16", "17_24"}
        actual_periods = set(df["period"].unique())
        assert actual_periods == valid_periods
