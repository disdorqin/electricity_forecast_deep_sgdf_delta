"""Tests for the output contract (predictions.csv schema).

Covers:
  1. OUTPUT_COLUMNS completeness
  2. validate_predictions detects missing columns
  3. build_prediction_row basic (deep_only)
  4. build_prediction_row with SGDFNet blend
  5. add_eval_columns correctness
  6. strip_eval_columns removes eval-only columns
  7. EVAL_ONLY_COLUMNS lists y_true-dependent fields
  8. compute_trend_confidence range [0.1, 0.95]
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.output_contract import (
    EVAL_ONLY_COLUMNS,
    OUTPUT_COLUMNS,
    add_eval_columns,
    build_prediction_row,
    build_predictions_dataframe,
    compute_trend_confidence,
    strip_eval_columns,
    validate_predictions,
)


# ── Test 1: OUTPUT_COLUMNS completeness ──────────────────────────────


class TestOutputColumnsDefined:
    """OUTPUT_COLUMNS must contain every column the contract requires."""

    REQUIRED_NAMES = [
        "business_day",
        "hour_business",
        "period",
        "ds",
        "da_anchor",
        "y_true",
        "deep_delta_pred",
        "deep_rt_pred",
        "sgdfnet_pred",
        "blend_pred",
        "trend_pred",
        "trend_model_name",
        "trend_confidence",
        "normal_trend_flag",
        "high_price_bucket_flag",
        "negative_bucket_flag",
        "residual_for_spike_module",
        "residual_for_negative_module",
    ]

    def test_output_columns_defined(self):
        assert isinstance(OUTPUT_COLUMNS, list)
        for name in self.REQUIRED_NAMES:
            assert name in OUTPUT_COLUMNS, f"Missing required column: {name}"

    def test_no_duplicate_columns(self):
        assert len(OUTPUT_COLUMNS) == len(set(OUTPUT_COLUMNS))

    def test_eval_only_is_subset(self):
        for col in EVAL_ONLY_COLUMNS:
            assert col in OUTPUT_COLUMNS, f"EVAL_ONLY column {col!r} not in OUTPUT_COLUMNS"


# ── Test 2: validate_predictions detects missing columns ─────────────


class TestValidatePredictionsMissingColumns:

    def test_all_missing(self):
        df = pd.DataFrame()
        missing = validate_predictions(df)
        assert set(missing) == set(OUTPUT_COLUMNS)

    def test_no_missing(self):
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        missing = validate_predictions(df)
        assert missing == []

    def test_partial_missing(self):
        keep = [c for c in OUTPUT_COLUMNS if c not in ("sgdfnet_pred", "trend_confidence")]
        df = pd.DataFrame(columns=keep)
        missing = validate_predictions(df)
        assert "sgdfnet_pred" in missing
        assert "trend_confidence" in missing
        assert "blend_pred" not in missing


# ── Test 3: build_prediction_row basic (deep_only) ──────────────────


class TestBuildPredictionRowBasic:

    def test_deep_only_values(self):
        row = build_prediction_row(
            business_day="2026-03-15",
            hour=10,
            da_anchor=200.0,
            deep_delta_pred=15.0,
        )

        assert row["business_day"] == pd.Timestamp("2026-03-15")
        assert row["hour_business"] == 10
        assert row["period"] == "9_16"
        assert row["da_anchor"] == pytest.approx(200.0)
        assert row["deep_delta_pred"] == pytest.approx(15.0)
        assert row["deep_rt_pred"] == pytest.approx(215.0)
        assert row["blend_pred"] == pytest.approx(215.0)
        assert row["trend_pred"] == pytest.approx(215.0)
        assert math.isnan(row["sgdfnet_pred"])
        assert math.isnan(row["y_true"])
        assert row["trend_model_name"] == "DeepSGDFDelta_V2_tcn"

    def test_period_1_8(self):
        row = build_prediction_row("2026-01-01", 3, 100.0, 5.0)
        assert row["period"] == "1_8"

    def test_period_17_24(self):
        row = build_prediction_row("2026-01-01", 20, 100.0, 5.0)
        assert row["period"] == "17_24"

    def test_boundary_hours(self):
        assert build_prediction_row("2026-01-01", 1, 100.0, 0.0)["period"] == "1_8"
        assert build_prediction_row("2026-01-01", 8, 100.0, 0.0)["period"] == "1_8"
        assert build_prediction_row("2026-01-01", 9, 100.0, 0.0)["period"] == "9_16"
        assert build_prediction_row("2026-01-01", 16, 100.0, 0.0)["period"] == "9_16"
        assert build_prediction_row("2026-01-01", 17, 100.0, 0.0)["period"] == "17_24"
        assert build_prediction_row("2026-01-01", 24, 100.0, 0.0)["period"] == "17_24"

    def test_ds_derived_from_business_day_and_hour(self):
        row = build_prediction_row("2026-03-15", 5, 100.0, 0.0)
        assert row["ds"] == pd.Timestamp("2026-03-15 04:00:00")

    def test_ds_explicit(self):
        row = build_prediction_row("2026-03-15", 5, 100.0, 0.0, ds="2026-03-15 05:30:00")
        assert row["ds"] == pd.Timestamp("2026-03-15 05:30:00")

    def test_invalid_hour_raises(self):
        with pytest.raises(ValueError, match="hour_business"):
            build_prediction_row("2026-01-01", 0, 100.0, 0.0)

    def test_normal_trend_flag_default(self):
        row = build_prediction_row("2026-01-01", 12, 200.0, 10.0)
        assert row["normal_trend_flag"] == 1

    def test_all_output_keys_present(self):
        row = build_prediction_row("2026-01-01", 12, 200.0, 10.0)
        for col in OUTPUT_COLUMNS:
            assert col in row, f"Row missing key: {col}"


# ── Test 4: build_prediction_row with SGDFNet blend ──────────────────


class TestBuildPredictionRowWithBlend:

    def test_sgdfnet_blend(self):
        """blend = w * sgdfnet + (1-w) * deep_rt"""
        row = build_prediction_row(
            business_day="2026-03-15",
            hour=12,
            da_anchor=200.0,
            deep_delta_pred=10.0,
            sgdfnet_pred=220.0,
            blend_mode="sgdfnet_blend",
            blend_weight=0.6,
        )
        # deep_rt = 200 + 10 = 210
        # blend = 0.6 * 220 + 0.4 * 210 = 132 + 84 = 216
        assert row["deep_rt_pred"] == pytest.approx(210.0)
        assert row["blend_pred"] == pytest.approx(216.0)
        assert row["trend_pred"] == pytest.approx(216.0)
        assert row["sgdfnet_pred"] == pytest.approx(220.0)

    def test_sgdfnet_residual(self):
        """blend = sgdfnet + deep_delta (deep predicts residual)"""
        row = build_prediction_row(
            business_day="2026-03-15",
            hour=12,
            da_anchor=200.0,
            deep_delta_pred=5.0,
            sgdfnet_pred=215.0,
            blend_mode="sgdfnet_residual",
        )
        # blend = 215 + 5 = 220
        assert row["blend_pred"] == pytest.approx(220.0)

    def test_sgdfnet_none_falls_back_to_deep_only(self):
        row = build_prediction_row(
            business_day="2026-03-15",
            hour=12,
            da_anchor=200.0,
            deep_delta_pred=10.0,
            sgdfnet_pred=None,
            blend_mode="sgdfnet_blend",
            blend_weight=0.6,
        )
        assert math.isnan(row["sgdfnet_pred"])
        assert row["blend_pred"] == pytest.approx(210.0)

    def test_unknown_blend_mode_raises(self):
        with pytest.raises(ValueError, match="blend_mode"):
            build_prediction_row(
                "2026-01-01", 12, 200.0, 10.0,
                sgdfnet_pred=210.0,
                blend_mode="unknown_mode",
            )

    def test_dataframe_from_rows(self):
        rows = [
            build_prediction_row("2026-03-15", h, 200.0, float(h))
            for h in range(1, 25)
        ]
        df = build_predictions_dataframe(rows)
        assert list(df.columns) == OUTPUT_COLUMNS
        assert len(df) == 24
        assert df["hour_business"].tolist() == list(range(1, 25))

    def test_empty_dataframe(self):
        df = build_predictions_dataframe([])
        assert list(df.columns) == OUTPUT_COLUMNS
        assert len(df) == 0


# ── Test 5: add_eval_columns ─────────────────────────────────────────


class TestAddEvalColumns:

    def _make_df(self):
        """Build a small prediction DataFrame with known y_true values."""
        rows = []
        test_data = [
            # (hour, da, delta, y_true)
            (3, 200.0, 10.0, 215.0),      # normal
            (10, 200.0, 20.0, 600.0),     # spike (|y_true| > 500)
            (15, 200.0, -5.0, -30.0),     # negative
            (22, 200.0, 8.0, 190.0),      # normal
            (7, 200.0, 3.0, float("nan")),  # y_true unknown
        ]
        for hour, da, delta, yt in test_data:
            r = build_prediction_row("2026-04-01", hour, da, delta)
            r["y_true"] = yt
            r["trend_pred"] = da + delta  # explicit for clarity
            rows.append(r)
        return build_predictions_dataframe(rows)

    def test_flags_added(self):
        df = self._make_df()
        result = add_eval_columns(df)

        for col in EVAL_ONLY_COLUMNS:
            assert col in result.columns

    def test_high_price_bucket(self):
        df = self._make_df()
        result = add_eval_columns(df)
        flags = result["high_price_bucket_flag"].tolist()
        # row 0: |215| <= 500 → 0
        # row 1: |600| > 500  → 1
        # row 2: |-30| <= 500 → 0
        # row 3: |190| <= 500 → 0
        # row 4: NaN           → NA
        assert flags[0] == 0
        assert flags[1] == 1
        assert flags[2] == 0
        assert flags[3] == 0
        assert pd.isna(flags[4])

    def test_negative_bucket(self):
        df = self._make_df()
        result = add_eval_columns(df)
        flags = result["negative_bucket_flag"].tolist()
        assert flags[0] == 0
        assert flags[1] == 0
        assert flags[2] == 1  # y_true = -30 < 0
        assert flags[3] == 0
        assert pd.isna(flags[4])

    def test_normal_trend_flag(self):
        df = self._make_df()
        result = add_eval_columns(df)
        flags = result["normal_trend_flag"].tolist()
        assert flags[0] == 1  # normal
        assert flags[1] == 0  # spike
        assert flags[2] == 0  # negative
        assert flags[3] == 1  # normal
        assert pd.isna(flags[4])  # unknown

    def test_residuals(self):
        df = self._make_df()
        result = add_eval_columns(df)
        # row 0: y_true=215, trend_pred=210 → residual = 5
        assert result["residual_for_spike_module"].iloc[0] == pytest.approx(5.0)
        assert result["residual_for_negative_module"].iloc[0] == pytest.approx(5.0)
        # row 4: y_true=NaN → residual = NaN
        assert pd.isna(result["residual_for_spike_module"].iloc[4])

    def test_does_not_mutate_input(self):
        df = self._make_df()
        original_cols = set(df.columns)
        _ = add_eval_columns(df)
        assert set(df.columns) == original_cols


# ── Test 6: strip_eval_columns ───────────────────────────────────────


class TestStripEvalColumns:

    def test_removes_eval_columns(self):
        rows = [build_prediction_row("2026-04-01", h, 200.0, 5.0) for h in (1, 12, 20)]
        df = build_predictions_dataframe(rows)
        df = add_eval_columns(df)

        stripped = strip_eval_columns(df)
        for col in EVAL_ONLY_COLUMNS:
            assert col not in stripped.columns

    def test_keeps_non_eval_columns(self):
        rows = [build_prediction_row("2026-04-01", 10, 200.0, 5.0)]
        df = build_predictions_dataframe(rows)
        df = add_eval_columns(df)

        stripped = strip_eval_columns(df)
        for col in OUTPUT_COLUMNS:
            if col not in EVAL_ONLY_COLUMNS:
                assert col in stripped.columns

    def test_silent_on_missing_eval_columns(self):
        """strip_eval_columns should not raise if eval columns are absent."""
        rows = [build_prediction_row("2026-04-01", 10, 200.0, 5.0)]
        df = build_predictions_dataframe(rows)
        # Don't add eval columns — just strip.  Should not error.
        stripped = strip_eval_columns(df)
        assert len(stripped) == 1

    def test_does_not_mutate_input(self):
        rows = [build_prediction_row("2026-04-01", 10, 200.0, 5.0)]
        df = build_predictions_dataframe(rows)
        df = add_eval_columns(df)
        original_cols = set(df.columns)
        _ = strip_eval_columns(df)
        assert set(df.columns) == original_cols


# ── Test 7: EVAL_ONLY_COLUMNS contain y_true-dependent fields ────────


class TestEvalColumnsContainYtrueDependentFields:

    EXPECTED_EVAL_FIELDS = [
        "high_price_bucket_flag",
        "negative_bucket_flag",
        "residual_for_spike_module",
        "residual_for_negative_module",
    ]

    def test_eval_columns_contain_y_true_dependent_fields(self):
        for field in self.EXPECTED_EVAL_FIELDS:
            assert field in EVAL_ONLY_COLUMNS, (
                f"{field} depends on y_true and must be in EVAL_ONLY_COLUMNS"
            )

    def test_residuals_are_y_true_dependent(self):
        """Both residual columns are defined as y_true - trend_pred."""
        assert "residual_for_spike_module" in EVAL_ONLY_COLUMNS
        assert "residual_for_negative_module" in EVAL_ONLY_COLUMNS

    def test_bucket_flags_are_y_true_dependent(self):
        assert "high_price_bucket_flag" in EVAL_ONLY_COLUMNS
        assert "negative_bucket_flag" in EVAL_ONLY_COLUMNS


# ── Test 8: compute_trend_confidence range ───────────────────────────


class TestTrendConfidenceRange:

    def test_deep_only_returns_float(self):
        c = compute_trend_confidence(10.0, None, 200.0)
        assert isinstance(c, float)

    def test_deep_only_in_range(self):
        c = compute_trend_confidence(10.0, None, 200.0)
        assert 0.1 <= c <= 0.95

    def test_deep_only_small_delta_higher_confidence(self):
        c_small = compute_trend_confidence(5.0, None, 200.0)
        c_large = compute_trend_confidence(100.0, None, 200.0)
        assert c_small > c_large

    def test_blended_in_range(self):
        c = compute_trend_confidence(10.0, 215.0, 200.0)
        assert 0.1 <= c <= 0.95

    def test_blended_agreement_higher_than_disagreement(self):
        # deep_rt = 200 + 10 = 210, sgdfnet = 212 → close
        c_agree = compute_trend_confidence(10.0, 212.0, 200.0)
        # deep_rt = 200 + 10 = 210, sgdfnet = 350 → far
        c_disagree = compute_trend_confidence(10.0, 350.0, 200.0)
        assert c_agree > c_disagree

    def test_nan_sgdfnet_treated_as_unavailable(self):
        c = compute_trend_confidence(10.0, float("nan"), 200.0)
        assert 0.1 <= c <= 0.95

    def test_extreme_delta_clamped(self):
        # Very large delta relative to anchor → should still be >= 0.1
        c = compute_trend_confidence(500.0, None, 10.0)
        assert c >= 0.1

    def test_zero_anchor_does_not_crash(self):
        c = compute_trend_confidence(10.0, None, 0.0)
        assert 0.1 <= c <= 0.95
