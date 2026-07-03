"""Tests for teacher registry (teacher_registry.py)."""
from __future__ import annotations

import pytest
import pandas as pd

from models.deep_sgdf_delta.teacher_registry import (
    TeacherRegistry,
    TeacherStatus,
    TEACHER_NAMES,
)


class TestTeacherRegistry:
    def test_default_init(self):
        reg = TeacherRegistry()
        assert set(reg.teachers.keys()) == set(TEACHER_NAMES)
        for name in TEACHER_NAMES:
            assert reg.teachers[name].availability == "unavailable"
            assert reg.predictions[name] is None

    def test_summary_empty(self):
        reg = TeacherRegistry()
        s = reg.summary()
        assert len(s) == len(TEACHER_NAMES)
        for name in TEACHER_NAMES:
            assert s[name]["availability"] == "unavailable"
            assert s[name]["n_predictions"] == 0

    def test_unknown_teacher_load(self):
        reg = TeacherRegistry()
        status = reg.load_teacher("nonexistent_teacher")
        assert status.availability == "unavailable"
        assert "Unknown teacher" in status.error

    def test_get_merged_predictions_empty(self):
        reg = TeacherRegistry()
        result = reg.get_merged_predictions()
        assert result is None

    def test_get_wide_predictions_empty(self):
        reg = TeacherRegistry()
        result = reg.get_wide_predictions()
        assert result is None

    def test_manual_teacher_injection(self):
        """Manually inject predictions and verify merge."""
        reg = TeacherRegistry()

        df1 = pd.DataFrame({
            "business_day": pd.Timestamp("2026-03-15"),
            "hour_business": [1, 2],
            "period": ["1_8", "1_8"],
            "teacher_name": "sgdfnet",
            "teacher_pred": [100.0, 200.0],
            "teacher_delta_pred": [10.0, 20.0],
            "teacher_available": [True, True],
            "teacher_source": ["test"] * 2,
        })
        reg.predictions["sgdfnet"] = df1
        reg.teachers["sgdfnet"] = TeacherStatus(
            name="sgdfnet", availability="available", n_predictions=2,
        )

        merged = reg.get_merged_predictions()
        assert merged is not None
        assert len(merged) == 2

    def test_load_all_with_unknown_teachers(self):
        reg = TeacherRegistry()
        # load_all with unknown names should not crash
        result = reg.load_all(teachers=["unknown1", "unknown2"])
        assert result["unknown1"].availability == "unavailable"
        assert result["unknown2"].availability == "unavailable"


class TestTeacherStatus:
    def test_status_creation(self):
        s = TeacherStatus(name="test", availability="available", n_predictions=100)
        assert s.name == "test"
        assert s.availability == "available"
        assert s.n_predictions == 100
        assert s.source_path is None
        assert s.error is None

    def test_status_with_error(self):
        s = TeacherStatus(name="test", availability="unavailable", error="something broke")
        assert s.error == "something broke"
