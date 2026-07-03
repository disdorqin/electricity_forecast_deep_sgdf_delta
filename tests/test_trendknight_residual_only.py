"""Tests for residual-only TrendKnightRT profiles."""
from __future__ import annotations

import pytest


class TestResidualOnlyProfiles:
    """Residual-only profile tests."""

    def test_residual_profiles_in_script(self):
        """Check that residual profiles exist in training script."""
        import sys
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_realtime_deep_model.py"
        content = script_path.read_text(encoding="utf-8")
        # Check the profiles dict contains residual profiles
        assert "residual" in content
