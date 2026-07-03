"""Tests for RT916 teacher scope restriction (rt916_scope.py).

Phase 5 Task C: RT916 only as local teacher.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.deep_sgdf_delta.rt916_scope import (
    RT916ScopeConfig,
    compute_scope_mask,
    apply_rt916_scope,
    evaluate_rt916_local_quality,
)


class TestComputeScopeMask:
    def test_default_allows_916_and_1724(self):
        """Default config allows 9_16 and 17_24 periods."""
        config = RT916ScopeConfig()
        rt = np.zeros((2, 24))
        seg = np.zeros((2, 24), dtype=np.int64)
        seg[:, :8] = 0    # 1_8
        seg[:, 8:16] = 1  # 9_16
        seg[:, 16:] = 2   # 17_24
        delta = np.zeros((2, 24))

        mask = compute_scope_mask(rt, seg, delta, config=config)
        # 9_16 and 17_24 should be allowed
        assert mask[:, 8:16].all()   # 9_16 allowed
        assert mask[:, 16:].all()    # 17_24 allowed
        # 1_8 should NOT be allowed (normal price, normal volatility)
        assert not mask[:, :8].any()

    def test_spike_allows_all_periods(self):
        """High price bucket allows all periods."""
        config = RT916ScopeConfig()
        rt = np.zeros((1, 24))
        rt[0, 3] = 600.0  # spike in 1_8 period
        seg = np.zeros((1, 24), dtype=np.int64)
        seg[:, :8] = 0
        seg[:, 8:16] = 1
        seg[:, 16:] = 2
        delta = np.zeros((1, 24))

        mask = compute_scope_mask(rt, seg, delta, config=config)
        assert mask[0, 3]  # spike hour allowed
        assert mask[0, 8]  # 9_16 still allowed by period
        assert not mask[0, 0]  # normal 1_8 not allowed

    def test_high_volatility_allows(self):
        """High volatility bucket allows hours."""
        config = RT916ScopeConfig(volatility_threshold=50.0)
        rt = np.zeros((1, 24))
        seg = np.zeros((1, 24), dtype=np.int64)
        delta = np.zeros((1, 24))
        delta[0, 2] = 80.0  # high volatility in 1_8

        mask = compute_scope_mask(rt, seg, delta, config=config)
        assert mask[0, 2]  # high-vol hour allowed

    def test_disabled_allows_all(self):
        """When disabled, all hours are allowed."""
        config = RT916ScopeConfig(enabled=False)
        rt = np.zeros((1, 24))
        seg = np.zeros((1, 24), dtype=np.int64)
        delta = np.zeros((1, 24))

        mask = compute_scope_mask(rt, seg, delta, config=config)
        assert mask.all()

    def test_custom_periods(self):
        """Custom allowed_periods."""
        config = RT916ScopeConfig(allowed_periods=["9_16"])
        rt = np.zeros((1, 24))
        seg = np.zeros((1, 24), dtype=np.int64)
        seg[:, :8] = 0     # 1_8
        seg[:, 8:16] = 1   # 9_16
        seg[:, 16:] = 2    # 17_24
        delta = np.zeros((1, 24))

        mask = compute_scope_mask(rt, seg, delta, config=config)
        assert mask[:, 8:16].all()   # 9_16 allowed
        assert not mask[:, 16:].any()  # 17_24 NOT allowed


class TestApplyRT916Scope:
    def test_blocks_disallowed_hours(self):
        """RT916 predictions in disallowed hours are set to NaN."""
        config = RT916ScopeConfig()
        tp = np.ones((2, 3, 24), dtype=np.float32) * 10.0
        tm = np.ones((2, 3), dtype=np.float32)
        names = ["sgdfnet", "rt916", "timemixer"]

        rt = np.zeros((2, 24))
        seg = np.zeros((2, 24), dtype=np.int64)
        seg[:, :8] = 0
        seg[:, 8:16] = 1
        seg[:, 16:] = 2
        delta = np.zeros((2, 24))

        tp_out, tm_out, stats = apply_rt916_scope(
            tp, tm, names, rt916_idx=1,
            rt_actual=rt, segment_ids=seg, delta_true=delta,
            config=config,
        )

        # 1_8 hours should be NaN for RT916
        assert np.isnan(tp_out[:, 1, :8]).all()
        # 9_16 and 17_24 should still have values
        assert np.isfinite(tp_out[:, 1, 8:16]).all()
        assert np.isfinite(tp_out[:, 1, 16:]).all()
        # SGDFNet (index 0) should be untouched
        assert np.isfinite(tp_out[:, 0, :]).all()
        # Stats
        assert stats["blocked_hours"] > 0
        assert stats["allowed_hours"] > 0

    def test_no_rt916_no_change(self):
        """If RT916 not in teacher names, no change."""
        config = RT916ScopeConfig()
        tp = np.ones((1, 2, 24), dtype=np.float32) * 5.0
        tm = np.ones((1, 2), dtype=np.float32)
        names = ["sgdfnet", "timemixer"]

        rt = np.zeros((1, 24))
        seg = np.zeros((1, 24), dtype=np.int64)
        delta = np.zeros((1, 24))

        tp_out, tm_out, stats = apply_rt916_scope(
            tp, tm, names, rt916_idx=1,
            rt_actual=rt, segment_ids=seg, delta_true=delta,
            config=config,
        )

        np.testing.assert_array_equal(tp_out, tp)
        assert not stats["enabled"] or stats["blocked_hours"] == 0

    def test_auto_disable_when_all_blocked(self):
        """If all RT916 hours blocked, auto-disable."""
        config = RT916ScopeConfig(allowed_periods=[])  # no periods allowed
        config.spike_threshold = 99999  # no spikes
        config.volatility_threshold = 99999  # no volatility

        tp = np.ones((1, 3, 24), dtype=np.float32) * 10.0
        tm = np.ones((1, 3), dtype=np.float32)
        names = ["sgdfnet", "rt916", "timemixer"]

        rt = np.zeros((1, 24))
        seg = np.zeros((1, 24), dtype=np.int64)
        delta = np.zeros((1, 24))

        tp_out, tm_out, stats = apply_rt916_scope(
            tp, tm, names, rt916_idx=1,
            rt_actual=rt, segment_ids=seg, delta_true=delta,
            config=config,
        )

        assert stats["auto_disabled"]
        assert tm_out[0, 1] == 0.0  # RT916 mask zeroed


class TestEvaluateRT916LocalQuality:
    def test_no_rt916_data(self):
        """No RT916 data → recommend disable."""
        tp = np.full((1, 3, 24), np.nan, dtype=np.float32)
        tm = np.zeros((1, 3), dtype=np.float32)
        delta = np.zeros((1, 24))

        result = evaluate_rt916_local_quality(
            tp, tm, ["sgdfnet", "rt916", "timemixer"],
            rt916_idx=1, delta_true=delta,
        )
        assert result["recommendation"] == "disable"

    def test_with_rt916_data(self):
        """With RT916 data → returns quality metrics."""
        tp = np.ones((1, 3, 24), dtype=np.float32) * 5.0
        tm = np.ones((1, 3), dtype=np.float32)
        delta = np.ones((1, 24), dtype=np.float32) * 3.0

        result = evaluate_rt916_local_quality(
            tp, tm, ["sgdfnet", "rt916", "timemixer"],
            rt916_idx=1, delta_true=delta,
            sgdfnet_pred=tp[:, 0:1, :].squeeze(1),
            sgdfnet_idx=0,
        )
        assert "rt916_local_smape" in result
        assert "recommendation" in result
