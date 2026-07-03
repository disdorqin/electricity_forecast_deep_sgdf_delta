"""Tests for generate_sgdfnet_predictions_for_deep.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestGenerateSGDFNetPredictions:
    """Basic CLI and dry-run tests."""

    def test_dry_run(self):
        """--dry-run works without errors."""
        sgdfnet_root = (
            Path(__file__).resolve().parent.parent.parent
            / "electricity_forecast_model2.0_exp" / "SGDFNet"
        )
        if not sgdfnet_root.exists():
            pytest.skip("SGDFNet root not found")

        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_sgdfnet_predictions_for_deep",
             "--sgdfnet-root", str(sgdfnet_root),
             "--dry-run"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "SGDFNet" in output

    def test_help(self):
        """--help works."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_sgdfnet_predictions_for_deep", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
