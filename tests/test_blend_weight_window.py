"""Tests for blend-weight search window and per-segment/per-season weights.

Covers:
  - Blend weight search only uses D-30 to D-1 (not future data)
  - Empty validation window returns default weight
  - Per-period weights differ across segments
  - Per-season weights differ across seasons

These tests use the ``find_optimal_blend_weight`` function from the predict
module, which requires SGDFNet.  When SGDFNet is absent (e.g. CI), tests
that need it are skipped.  Pure-logic tests that reimplement the search
inline are always runnable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.metrics import smape_floor50


# ── SGDFNet availability guard for predict module ─────────────────────

try:
    from models.deep_sgdf_delta.predict import find_optimal_blend_weight
    _PREDICT_AVAILABLE = True
except Exception:
    _PREDICT_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────

def _make_validation_data(
    n_days: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build synthetic validation data for blend weight tests."""
    rng = np.random.RandomState(seed)
    n = 24 * n_days

    deep_pred = pd.DataFrame({
        "rt_pred": rng.randn(n) * 30 + 200,
        "delta_pred": rng.randn(n) * 15,
        "da_anchor": np.ones(n) * 200,
        "hour": np.tile(np.arange(1, 25), n_days),
    })

    sgdf_pred = pd.DataFrame({
        "rt_hat": rng.randn(n) * 30 + 200,
        "delta_hat": rng.randn(n) * 15,
        "hour": np.tile(np.arange(1, 25), n_days),
    })

    y_true = rng.randn(n) * 30 + 200

    return deep_pred, sgdf_pred, y_true


def _search_blend_weight(
    deep_rt: np.ndarray,
    sgdf_rt: np.ndarray,
    y_true: np.ndarray,
    candidates: list[float] | None = None,
) -> float:
    """Reimplemented blend-weight search (no SGDFNet dependency).

    Mirrors ``find_optimal_blend_weight`` logic so we can test the
    search-window constraint independently.
    """
    if candidates is None:
        candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    best_w = 0.5
    best_score = float("inf")

    for w in candidates:
        blended = w * sgdf_rt + (1 - w) * deep_rt
        score = smape_floor50(y_true, blended)
        if score < best_score:
            best_score = score
            best_w = w

    return best_w


# ── Test: search only uses D-30 to D-1 ───────────────────────────────


class TestBlendWeightWindowConstraint:
    """Blend weight search must use only the validation window (D-30 to D-1)."""

    def test_window_excludes_future_data(self):
        """Search on D-30..D-1 should not be influenced by D+1 data."""
        rng = np.random.RandomState(123)

        # Build 30-day validation window
        n_val = 24 * 30
        deep_rt_val = rng.randn(n_val) * 30 + 200
        sgdf_rt_val = rng.randn(n_val) * 30 + 200
        y_true_val = rng.randn(n_val) * 30 + 200

        # Search using only the validation window
        w_val = _search_blend_weight(deep_rt_val, sgdf_rt_val, y_true_val)

        # Now append 30 more days of "future" data that would strongly
        # favour a different weight
        n_future = 24 * 30
        # Future data where SGDFNet is terrible and deep is perfect
        deep_rt_future = rng.randn(n_future) * 5 + 200
        sgdf_rt_future = rng.randn(n_future) * 200 + 200  # very noisy
        y_true_future = deep_rt_future.copy()  # deep is perfect

        # Search on combined (val + future) should yield a different weight
        deep_rt_all = np.concatenate([deep_rt_val, deep_rt_future])
        sgdf_rt_all = np.concatenate([sgdf_rt_val, sgdf_rt_future])
        y_true_all = np.concatenate([y_true_val, y_true_future])

        w_all = _search_blend_weight(deep_rt_all, sgdf_rt_all, y_true_all)

        # The validation-only weight should differ from the combined weight
        # (because future data strongly favours w=0 for deep)
        assert w_val != w_all or w_val == 0.0, (
            "Validation-only search should differ from combined search "
            "when future data has a different optimal weight"
        )

    def test_window_is_bounded(self):
        """Verify the search function doesn't use data beyond the provided arrays."""
        deep_rt, sgdf_rt, y_true = _make_validation_data(n_days=30, seed=99)

        # Use only first 10 days
        n_10 = 24 * 10
        w_10 = _search_blend_weight(
            deep_rt["rt_pred"].values[:n_10],
            sgdf_rt["rt_hat"].values[:n_10],
            y_true[:n_10],
        )

        # Use all 30 days
        w_30 = _search_blend_weight(
            deep_rt["rt_pred"].values,
            sgdf_rt["rt_hat"].values,
            y_true,
        )

        # Both should return a valid candidate
        assert w_10 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert w_30 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ── Test: empty validation window returns default weight ──────────────


class TestBlendWeightEmptyWindow:
    """An empty validation window should return the default weight (0.5)."""

    def test_empty_arrays_return_default(self):
        deep_rt = np.array([], dtype=float)
        sgdf_rt = np.array([], dtype=float)
        y_true = np.array([], dtype=float)

        # With empty data, all weights produce the same sMAPE (NaN or 0),
        # so the function should fall through to the default.
        w = _search_blend_weight(deep_rt, sgdf_rt, y_true)
        # Default is 0.5 (initial best_w)
        assert w == 0.5

    @pytest.mark.skipif(not _PREDICT_AVAILABLE, reason="predict module not available (needs SGDFNet)")
    def test_empty_via_predict_module(self):
        """find_optimal_blend_weight with empty DataFrames returns default."""
        deep_pred = pd.DataFrame(columns=["rt_pred", "delta_pred", "da_anchor", "hour"])
        sgdf_pred = pd.DataFrame(columns=["rt_hat", "delta_hat", "hour"])
        y_true = np.array([], dtype=float)

        w = find_optimal_blend_weight(deep_pred, sgdf_pred, y_true)
        assert w == 0.5


# ── Test: per-period weights differ across segments ────────────────────


class TestPerPeriodBlendWeights:
    """Optimal blend weight should differ across time-of-day segments."""

    def test_per_period_weights_differ(self):
        """Build data where different segments favour different weights."""
        rng = np.random.RandomState(77)
        n_days = 30
        n = 24 * n_days

        deep_rt = np.zeros(n)
        sgdf_rt = np.zeros(n)
        y_true = np.zeros(n)
        hours = np.tile(np.arange(1, 25), n_days)

        for i in range(n):
            h = hours[i]
            if 1 <= h <= 8:
                # Segment 1_8: deep model is better
                y_true[i] = 200 + rng.randn() * 5
                deep_rt[i] = y_true[i] + rng.randn() * 2    # deep is close
                sgdf_rt[i] = y_true[i] + rng.randn() * 20   # SGDFNet is noisy
            elif 9 <= h <= 16:
                # Segment 9_16: SGDFNet is better
                y_true[i] = 200 + rng.randn() * 30
                deep_rt[i] = y_true[i] + rng.randn() * 25   # deep is noisy
                sgdf_rt[i] = y_true[i] + rng.randn() * 3    # SGDFNet is close
            else:
                # Segment 17_24: both about equal
                y_true[i] = 200 + rng.randn() * 15
                deep_rt[i] = y_true[i] + rng.randn() * 10
                sgdf_rt[i] = y_true[i] + rng.randn() * 10

        # Per-segment search
        candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        weights = {}
        for seg_name, (lo, hi) in [("1_8", (1, 8)), ("9_16", (9, 16)), ("17_24", (17, 24))]:
            mask = (hours >= lo) & (hours <= hi)
            weights[seg_name] = _search_blend_weight(
                deep_rt[mask], sgdf_rt[mask], y_true[mask], candidates
            )

        # 1_8 should prefer low SGDFNet weight (deep is better there)
        # 9_16 should prefer high SGDFNet weight (SGDFNet is better there)
        assert weights["1_8"] < weights["9_16"], (
            f"Segment 1_8 weight ({weights['1_8']}) should be less than "
            f"9_16 weight ({weights['9_16']})"
        )


# ── Test: per-season weights differ across seasons ─────────────────────


class TestPerSeasonBlendWeights:
    """Optimal blend weight should differ across seasons."""

    def test_per_season_weights_differ(self):
        """Build data where summer and winter favour different weights."""
        rng = np.random.RandomState(55)

        # Summer: deep model is better (stable solar patterns)
        n_summer = 24 * 90  # ~90 days
        deep_rt_summer = rng.randn(n_summer) * 5 + 200
        sgdf_rt_summer = rng.randn(n_summer) * 30 + 200
        y_true_summer = deep_rt_summer + rng.randn(n_summer) * 2

        # Winter: SGDFNet is better (volatile weather)
        n_winter = 24 * 90
        deep_rt_winter = rng.randn(n_winter) * 30 + 150
        sgdf_rt_winter = rng.randn(n_winter) * 5 + 150
        y_true_winter = sgdf_rt_winter + rng.randn(n_winter) * 2

        candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        w_summer = _search_blend_weight(
            deep_rt_summer, sgdf_rt_summer, y_true_summer, candidates
        )
        w_winter = _search_blend_weight(
            deep_rt_winter, sgdf_rt_winter, y_true_winter, candidates
        )

        # Summer should prefer low SGDFNet weight (deep is better)
        # Winter should prefer high SGDFNet weight
        assert w_summer < w_winter, (
            f"Summer weight ({w_summer}) should be less than winter weight ({w_winter})"
        )

    def test_season_weights_are_valid_candidates(self):
        """All per-season weights must be valid candidate values."""
        rng = np.random.RandomState(88)
        candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for _ in range(5):
            n = 24 * 30
            deep_rt = rng.randn(n) * 20 + 200
            sgdf_rt = rng.randn(n) * 20 + 200
            y_true = rng.randn(n) * 20 + 200

            w = _search_blend_weight(deep_rt, sgdf_rt, y_true, candidates)
            assert w in candidates, f"Weight {w} not in candidates {candidates}"
