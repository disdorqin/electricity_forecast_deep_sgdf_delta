"""Tests for find_sgdfnet_predictions.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestFindSGDFNetPredictions:
    """Basic CLI and search logic tests."""

    def test_help(self):
        """--help works."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.find_sgdfnet_predictions", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "--out-dir" in result.stdout + result.stderr
        assert "--quick" in result.stdout + result.stderr
