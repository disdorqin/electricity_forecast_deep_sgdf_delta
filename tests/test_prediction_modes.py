"""Tests for prediction modes — Phase 9."""
from __future__ import annotations

import pandas as pd
import pytest

from models.deep_sgdf_delta.prediction_modes import (
    PredictionMode,
    validate_feature_visibility,
    validate_intraday_cutoff,
)


class TestFullDayMode:
    """FULL_DAY mode: no D-day actuals allowed."""

    def test_previous_day_feature_allowed(self):
        """Feature from D-1 is allowed for FULL_DAY."""
        result = validate_feature_visibility(
            mode=PredictionMode.FULL_DAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-14 15:00:00",
        )
        assert result["valid"] is True

    def test_same_day_feature_blocked(self):
        """FULL_DAY: D-day hour 9 residual cannot be used for D-day hour 10."""
        result = validate_feature_visibility(
            mode=PredictionMode.FULL_DAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-15 09:00:00",  # same day, hour 9
        )
        assert result["valid"] is False
        assert "FULL_DAY" in result["reason"]

    def test_same_day_midnight_is_previous_day(self):
        """Timestamp D 00:00 maps to business_day D-1 hour 24, so it's allowed."""
        result = validate_feature_visibility(
            mode=PredictionMode.FULL_DAY,
            business_day="2026-02-15",
            target_hour=1,
            feature_timestamp="2026-02-15 00:00:00",  # → bd=Feb 14, hb=24
        )
        assert result["valid"] is True

    def test_future_day_blocked(self):
        """Feature from D+1 is blocked."""
        result = validate_feature_visibility(
            mode=PredictionMode.FULL_DAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-16 10:00:00",
        )
        assert result["valid"] is False
        assert "future" in result["reason"].lower()


class TestIntradayMode:
    """INTRADAY mode: D-day actuals up to cutoff_hour allowed."""

    def test_cutoff_observed_hour_allowed(self):
        """INTRADAY cutoff=9: D-day hour 9 residual allowed for hour 10."""
        result = validate_feature_visibility(
            mode=PredictionMode.INTRADAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-15 09:00:00",  # same day, hour 9
            cutoff_hour=9,
        )
        assert result["valid"] is True

    def test_cutoff_unobserved_hour_blocked(self):
        """INTRADAY cutoff=9: D-day hour 10 residual NOT allowed for hour 10 or 11."""
        result = validate_feature_visibility(
            mode=PredictionMode.INTRADAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-15 10:00:00",  # same day, hour 10
            cutoff_hour=9,
        )
        assert result["valid"] is False

    def test_cutoff_unobserved_blocked_for_future_target(self):
        """INTRADAY cutoff=9: hour 10 residual NOT allowed for target hour 11."""
        result = validate_feature_visibility(
            mode=PredictionMode.INTRADAY,
            business_day="2026-02-15",
            target_hour=11,
            feature_timestamp="2026-02-15 10:00:00",
            cutoff_hour=9,
        )
        assert result["valid"] is False

    def test_no_cutoff_hour_error(self):
        """INTRADAY without cutoff_hour raises error."""
        result = validate_feature_visibility(
            mode=PredictionMode.INTRADAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-15 09:00:00",
            cutoff_hour=None,
        )
        assert result["valid"] is False
        assert "cutoff_hour" in result["reason"]

    def test_previous_day_always_allowed(self):
        """Previous day features allowed in INTRADAY mode."""
        result = validate_feature_visibility(
            mode=PredictionMode.INTRADAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-14 16:00:00",
            cutoff_hour=9,
        )
        assert result["valid"] is True

    def test_future_day_blocked_intraday(self):
        """D+1 actuals blocked in INTRADAY mode."""
        result = validate_feature_visibility(
            mode=PredictionMode.INTRADAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-16 10:00:00",
            cutoff_hour=9,
        )
        assert result["valid"] is False

    def test_same_feature_as_target_blocked(self):
        """Feature from same hour as target is blocked (not yet observed)."""
        result = validate_feature_visibility(
            mode=PredictionMode.INTRADAY,
            business_day="2026-02-15",
            target_hour=10,
            feature_timestamp="2026-02-15 10:00:00",
            cutoff_hour=10,
        )
        # hour 10 >= target hour 10 → blocked
        assert result["valid"] is False


class TestIntradayCutoffValidation:
    """Test validate_intraday_cutoff helper."""

    def test_target_after_cutoff(self):
        assert validate_intraday_cutoff(9, 10) is True

    def test_target_before_cutoff(self):
        assert validate_intraday_cutoff(10, 9) is False

    def test_target_equals_cutoff(self):
        assert validate_intraday_cutoff(10, 10) is False
