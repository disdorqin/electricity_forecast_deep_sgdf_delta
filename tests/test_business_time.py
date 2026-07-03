"""Tests for unified business-day time alignment (business_time.py).

Phase 6 Task A: Single source of truth for business-day alignment.
"""
from __future__ import annotations

import pandas as pd
import pytest

from models.deep_sgdf_delta.business_time import (
    add_business_time_columns,
    compute_business_day,
    compute_hour_business,
    compute_period,
)


class TestAddBusinessTimeColumns:
    """Test the DataFrame-level function."""

    def test_midnight_maps_to_previous_day_hour_24(self):
        """2026-03-15 00:00 → business_day 2026-03-14, hour_business 24."""
        df = pd.DataFrame({"ds": [pd.Timestamp("2026-03-15 00:00:00")]})
        result = add_business_time_columns(df)
        assert result["business_day"].iloc[0] == pd.Timestamp("2026-03-14")
        assert result["hour_business"].iloc[0] == 24
        assert result["period"].iloc[0] == "17_24"

    def test_hour_01_maps_to_same_day_hour_1(self):
        """2026-03-15 01:00 → business_day 2026-03-15, hour_business 1."""
        df = pd.DataFrame({"ds": [pd.Timestamp("2026-03-15 01:00:00")]})
        result = add_business_time_columns(df)
        assert result["business_day"].iloc[0] == pd.Timestamp("2026-03-15")
        assert result["hour_business"].iloc[0] == 1
        assert result["period"].iloc[0] == "1_8"

    def test_hour_23_maps_to_same_day_hour_23(self):
        """2026-03-15 23:00 → business_day 2026-03-15, hour_business 23."""
        df = pd.DataFrame({"ds": [pd.Timestamp("2026-03-15 23:00:00")]})
        result = add_business_time_columns(df)
        assert result["business_day"].iloc[0] == pd.Timestamp("2026-03-15")
        assert result["hour_business"].iloc[0] == 23
        assert result["period"].iloc[0] == "17_24"

    def test_period_mapping_all_hours(self):
        """Verify period mapping for all 24 hours."""
        timestamps = []
        for h in range(24):
            timestamps.append(pd.Timestamp(f"2026-03-15 {h:02d}:00:00"))
        # Add midnight of next day (hour 0 → hour_business 24)
        timestamps.append(pd.Timestamp("2026-03-16 00:00:00"))

        df = pd.DataFrame({"ds": timestamps})
        result = add_business_time_columns(df)

        # Hours 1-8 → 1_8
        for h in range(1, 9):
            row = result[result["hour_business"] == h].iloc[0]
            assert row["period"] == "1_8", f"Hour {h} should be 1_8, got {row['period']}"

        # Hours 9-16 → 9_16
        for h in range(9, 17):
            row = result[result["hour_business"] == h].iloc[0]
            assert row["period"] == "9_16", f"Hour {h} should be 9_16, got {row['period']}"

        # Hours 17-24 → 17_24
        for h in range(17, 25):
            row = result[result["hour_business"] == h].iloc[0]
            assert row["period"] == "17_24", f"Hour {h} should be 17_24, got {row['period']}"

    def test_multiple_rows(self):
        """Test with multiple timestamps spanning different days."""
        df = pd.DataFrame({
            "ds": [
                pd.Timestamp("2026-01-01 00:00:00"),  # → 2025-12-31, h24
                pd.Timestamp("2026-01-01 12:00:00"),  # → 2026-01-01, h12
                pd.Timestamp("2026-01-02 00:00:00"),  # → 2026-01-01, h24
            ]
        })
        result = add_business_time_columns(df)

        assert result["business_day"].iloc[0] == pd.Timestamp("2025-12-31")
        assert result["hour_business"].iloc[0] == 24

        assert result["business_day"].iloc[1] == pd.Timestamp("2026-01-01")
        assert result["hour_business"].iloc[1] == 12

        assert result["business_day"].iloc[2] == pd.Timestamp("2026-01-01")
        assert result["hour_business"].iloc[2] == 24

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-03-15 00:00:00")]})
        result = add_business_time_columns(
            df, timestamp_col="timestamp",
            business_day_col="bd", hour_col="hb", period_col="seg",
        )
        assert "bd" in result.columns
        assert "hb" in result.columns
        assert "seg" in result.columns


class TestComputeBusinessDay:
    def test_midnight(self):
        ts = pd.Timestamp("2026-03-15 00:00:00")
        assert compute_business_day(ts) == pd.Timestamp("2026-03-14")

    def test_hour_1(self):
        ts = pd.Timestamp("2026-03-15 01:00:00")
        assert compute_business_day(ts) == pd.Timestamp("2026-03-15")

    def test_hour_23(self):
        ts = pd.Timestamp("2026-03-15 23:00:00")
        assert compute_business_day(ts) == pd.Timestamp("2026-03-15")


class TestComputeHourBusiness:
    def test_midnight(self):
        assert compute_hour_business(pd.Timestamp("2026-03-15 00:00:00")) == 24

    def test_hour_1(self):
        assert compute_hour_business(pd.Timestamp("2026-03-15 01:00:00")) == 1

    def test_hour_23(self):
        assert compute_hour_business(pd.Timestamp("2026-03-15 23:00:00")) == 23


class TestComputePeriod:
    def test_period_1_8(self):
        for h in range(1, 9):
            assert compute_period(h) == "1_8"

    def test_period_9_16(self):
        for h in range(9, 17):
            assert compute_period(h) == "9_16"

    def test_period_17_24(self):
        for h in range(17, 25):
            assert compute_period(h) == "17_24"

    def test_invalid_hour_raises(self):
        with pytest.raises(ValueError):
            compute_period(0)
        with pytest.raises(ValueError):
            compute_period(25)
