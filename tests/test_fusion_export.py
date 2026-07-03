"""Tests for fusion export module (fusion_export.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.fusion_export import (
    FUSION_COLUMNS,
    validate_fusion_pack,
    build_fusion_row,
    build_fusion_dataframe,
    strip_eval_columns,
    add_eval_columns,
    _hour_to_period,
    _safe_float,
)


# ── Test: Helpers ────────────────────────────────────────────────────

class TestHelpers:
    def test_hour_to_period_1_8(self):
        assert _hour_to_period(1) == "1_8"
        assert _hour_to_period(8) == "1_8"

    def test_hour_to_period_9_16(self):
        assert _hour_to_period(9) == "9_16"
        assert _hour_to_period(16) == "9_16"

    def test_hour_to_period_17_24(self):
        assert _hour_to_period(17) == "17_24"
        assert _hour_to_period(24) == "17_24"

    def test_hour_to_period_invalid(self):
        with pytest.raises(ValueError):
            _hour_to_period(0)
        with pytest.raises(ValueError):
            _hour_to_period(25)

    def test_safe_float_none(self):
        assert np.isnan(_safe_float(None))

    def test_safe_float_number(self):
        assert _safe_float(42) == 42.0
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_invalid(self):
        assert np.isnan(_safe_float("not_a_number"))


# ── Test: build_fusion_row ──────────────────────────────────────────

class TestBuildFusionRow:
    def test_basic_row(self):
        row = build_fusion_row(
            business_day="2026-03-15",
            hour_business=10,
            ds="2026-03-15 09:00:00",
            trend_pred=320.5,
            trend_delta_pred=45.2,
            trend_confidence=0.78,
            shock_sensitivity=0.32,
        )
        assert row["business_day"] == pd.Timestamp("2026-03-15")
        assert row["hour_business"] == 10
        assert row["period"] == "9_16"
        assert row["trend_pred"] == pytest.approx(320.5)
        assert row["trend_confidence"] == pytest.approx(0.78)
        assert row["shock_sensitivity"] == pytest.approx(0.32)
        assert row["teacher_used"] == "none"
        assert row["model_name"] == "trendknight_x"

    def test_confidence_clipped(self):
        row = build_fusion_row(
            business_day="2026-03-15", hour_business=1,
            ds="2026-03-15 00:00:00",
            trend_pred=100, trend_delta_pred=10,
            trend_confidence=1.5,  # above 1
            shock_sensitivity=-0.1,  # below 0
        )
        assert row["trend_confidence"] == 1.0
        assert row["shock_sensitivity"] == 0.0

    def test_teacher_predictions_nan(self):
        row = build_fusion_row(
            business_day="2026-03-15", hour_business=1,
            ds="2026-03-15 00:00:00",
            trend_pred=100, trend_delta_pred=10,
            trend_confidence=0.5, shock_sensitivity=0.0,
        )
        assert np.isnan(row["sgdfnet_pred"])
        assert np.isnan(row["rt916_pred"])
        assert np.isnan(row["timemixer_pred"])


# ── Test: build_fusion_dataframe ─────────────────────────────────────

class TestBuildFusionDataFrame:
    def test_empty_rows(self):
        df = build_fusion_dataframe([])
        assert list(df.columns) == FUSION_COLUMNS
        assert len(df) == 0

    def test_column_order(self):
        rows = [
            build_fusion_row(
                business_day="2026-03-15", hour_business=h,
                ds=f"2026-03-15 {h-1:02d}:00:00",
                trend_pred=100 + h, trend_delta_pred=h,
                trend_confidence=0.5, shock_sensitivity=0.0,
            )
            for h in range(1, 25)
        ]
        df = build_fusion_dataframe(rows)
        assert list(df.columns) == FUSION_COLUMNS
        assert len(df) == 24

    def test_types(self):
        rows = [build_fusion_row(
            business_day="2026-03-15", hour_business=1,
            ds="2026-03-15 00:00:00",
            trend_pred=100, trend_delta_pred=10,
            trend_confidence=0.5, shock_sensitivity=0.0,
        )]
        df = build_fusion_dataframe(rows)
        assert pd.api.types.is_datetime64_any_dtype(df["business_day"])
        assert pd.api.types.is_datetime64_any_dtype(df["ds"])


# ── Test: validate_fusion_pack ──────────────────────────────────────

class TestValidateFusionPack:
    def _make_valid_df(self, n=24):
        rows = [
            build_fusion_row(
                business_day="2026-03-15", hour_business=h,
                ds=f"2026-03-15 {h-1:02d}:00:00",
                trend_pred=100 + h, trend_delta_pred=h,
                trend_confidence=0.5, shock_sensitivity=0.0,
            )
            for h in range(1, n + 1)
        ]
        return build_fusion_dataframe(rows)

    def test_valid_pack(self):
        df = self._make_valid_df()
        is_valid, errors = validate_fusion_pack(df)
        assert is_valid, f"Errors: {errors}"

    def test_missing_columns(self):
        df = pd.DataFrame({"foo": [1, 2]})
        is_valid, errors = validate_fusion_pack(df)
        assert not is_valid
        assert any("Missing required columns" in e for e in errors)

    def test_hour_out_of_range(self):
        df = self._make_valid_df()
        df.loc[0, "hour_business"] = 99
        is_valid, errors = validate_fusion_pack(df)
        assert not is_valid
        assert any("hour_business" in e for e in errors)

    def test_confidence_out_of_range(self):
        df = self._make_valid_df()
        df.loc[0, "trend_confidence"] = 2.0
        is_valid, errors = validate_fusion_pack(df)
        assert not is_valid
        assert any("trend_confidence" in e for e in errors)

    def test_nan_business_day(self):
        df = self._make_valid_df()
        df.loc[0, "business_day"] = pd.NaT
        is_valid, errors = validate_fusion_pack(df)
        assert not is_valid
        assert any("NaN" in e for e in errors)


# ── Test: Eval columns ──────────────────────────────────────────────

class TestEvalColumns:
    def test_strip_eval_columns(self):
        df = pd.DataFrame({
            "trend_pred": [100.0],
            "y_true": [105.0],
            "residual_for_spike": [5.0],
            "residual_for_negative": [5.0],
        })
        stripped = strip_eval_columns(df)
        assert "y_true" not in stripped.columns
        assert "residual_for_spike" not in stripped.columns
        assert "trend_pred" in stripped.columns

    def test_add_eval_columns(self):
        df = pd.DataFrame({
            "trend_pred": [100.0, 200.0],
            "y_true": [105.0, np.nan],
        })
        out = add_eval_columns(df)
        assert "residual_for_spike" in out.columns
        assert "residual_for_negative" in out.columns
        assert out["residual_for_spike"].iloc[0] == pytest.approx(5.0)
        assert np.isnan(out["residual_for_spike"].iloc[1])

    def test_add_eval_columns_missing_y_true(self):
        df = pd.DataFrame({"trend_pred": [100.0]})
        with pytest.raises(ValueError, match="y_true"):
            add_eval_columns(df)
