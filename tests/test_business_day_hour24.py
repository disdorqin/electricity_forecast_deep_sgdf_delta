"""Tests for the business-day / hour-24 alignment convention.

The project uses a "business day" convention where:
  - 00:00 of calendar day D maps to business_day = D-1 with hour_business = 24
  - 01:00 of calendar day D maps to business_day = D   with hour_business = 1
  - 23:00 of calendar day D maps to business_day = D   with hour_business = 23
  - 23:59 still maps to business_day = D, hour_business = 23 (sub-hour truncated)

These tests require the SGDFNet sibling project to be available, because
the ``add_business_time_columns`` function lives in ``sgdfnet.data_contract``.
When SGDFNet is absent (e.g. CI), the entire module is skipped.
"""
from __future__ import annotations

import pandas as pd
import pytest

# ── SGDFNet availability guard ────────────────────────────────────────

try:
    from sgdfnet.data_contract import add_business_time_columns
    _SGDFNET_AVAILABLE = True
except Exception:
    try:
        from models.deep_sgdf_delta.sgdfnet_bridge import add_business_time_columns
        _SGDFNET_AVAILABLE = True
    except Exception:
        _SGDFNET_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SGDFNET_AVAILABLE,
    reason="SGDFNet sibling project not available",
)


# ── Test: 00:00 maps to previous business_day with hour_business=24 ───


class TestMidnightMapping:
    """00:00 on calendar day D should be hour 24 of business day D-1."""

    def test_midnight_basic(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-03-15 00:00:00"]),
        })
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-03-14")
        assert row["target_hour"] == 24

    def test_midnight_january_first(self):
        """Midnight on Jan 1 should map to Dec 31 of the previous year."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2027-01-01 00:00:00"]),
        })
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-12-31")
        assert row["target_hour"] == 24

    def test_midnight_month_boundary(self):
        """Midnight on April 1 should map to March 31."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-04-01 00:00:00"]),
        })
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-03-31")
        assert row["target_hour"] == 24


# ── Test: 01:00 maps to same calendar day with hour_business=1 ────────


class TestOneAMMapping:
    """01:00 on calendar day D should be hour 1 of business day D."""

    def test_one_am_basic(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-03-15 01:00:00"]),
        })
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-03-15")
        assert row["target_hour"] == 1

    def test_one_am_weekend(self):
        """Business day alignment applies regardless of weekday/weekend."""
        # 2026-03-14 is a Saturday
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-03-14 01:00:00"]),
        })
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-03-14")
        assert row["target_hour"] == 1


# ── Test: 23:00 maps to same calendar day with hour_business=23 ──────


class TestElevenPMMapping:
    """23:00 on calendar day D should be hour 23 of business day D."""

    def test_eleven_pm_basic(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-03-15 23:00:00"]),
        })
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-03-15")
        assert row["target_hour"] == 23


# ── Test: 23:59 maps to same calendar day with hour_business=23 ──────


class TestSubHourTruncation:
    """23:59 (sub-hour) should still be hour 23, not rounded up."""

    def test_twenty_three_fifty_nine(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-03-15 23:59:00"]),
        })
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp("2026-03-15")
        assert row["target_hour"] == 23


# ── Test: multiple dates, weekends, month boundaries ─────────────────


class TestMultipleDatesAndBoundaries:
    """Verify alignment across a range of calendar situations."""

    @pytest.mark.parametrize(
        "ts,expected_bd,expected_hour",
        [
            # Standard weekday
            ("2026-03-16 00:00:00", "2026-03-15", 24),
            ("2026-03-16 01:00:00", "2026-03-16", 1),
            ("2026-03-16 12:00:00", "2026-03-16", 12),
            ("2026-03-16 23:00:00", "2026-03-16", 23),
            # Weekend: Saturday
            ("2026-03-14 00:00:00", "2026-03-13", 24),
            ("2026-03-14 08:00:00", "2026-03-14", 8),
            # Weekend: Sunday
            ("2026-03-15 00:00:00", "2026-03-14", 24),
            ("2026-03-15 17:00:00", "2026-03-15", 17),
            # Month boundary: last day of Feb -> March
            ("2026-03-01 00:00:00", "2026-02-28", 24),
            ("2026-03-01 01:00:00", "2026-03-01", 1),
            # Year boundary
            ("2027-01-01 00:00:00", "2026-12-31", 24),
            ("2027-01-01 06:00:00", "2027-01-01", 6),
        ],
    )
    def test_alignment(self, ts, expected_bd, expected_hour):
        df = pd.DataFrame({"timestamp": pd.to_datetime([ts])})
        result = add_business_time_columns(df, "timestamp")
        row = result.iloc[0]
        assert row["business_day"] == pd.Timestamp(expected_bd)
        assert row["target_hour"] == expected_hour

    def test_full_day_24_rows(self):
        """A full calendar day (01:00 to 00:00 next day) should produce hours 1-24."""
        timestamps = [pd.Timestamp("2026-03-15") + pd.Timedelta(hours=h) for h in range(24)]
        df = pd.DataFrame({"timestamp": timestamps})
        result = add_business_time_columns(df, "timestamp")

        # 00:00 -> hour 24 of previous business_day
        assert result.iloc[0]["target_hour"] == 24
        assert result.iloc[0]["business_day"] == pd.Timestamp("2026-03-14")

        # 01:00 through 23:00 -> hours 1-23 of same business_day
        for i in range(1, 24):
            assert result.iloc[i]["target_hour"] == i
            assert result.iloc[i]["business_day"] == pd.Timestamp("2026-03-15")
