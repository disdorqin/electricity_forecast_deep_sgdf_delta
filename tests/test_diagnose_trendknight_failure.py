"""Tests for diagnose_trendknight_failure.py."""
from __future__ import annotations

from pathlib import Path
import pytest


class TestDiagnoseFailure:
    """Diagnosis module import test."""

    def test_module_importable(self):
        from scripts import diagnose_trendknight_failure
        assert hasattr(diagnose_trendknight_failure, "main")
