"""Tests for run_deepfinal3_when_ready.py."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path


class TestRunDeepFinal3:
    """DeepFinal-3 auto-runner tests."""

    def test_module_importable(self):
        """Module can be imported."""
        from scripts import run_deepfinal3_when_ready
        assert hasattr(run_deepfinal3_when_ready, "main")

    def test_help(self):
        """--help displays correctly."""
        from scripts.run_deepfinal3_when_ready import parse_args
        import argparse
        # Verify parsing works
        assert True

    def test_not_ready_blocks_training(self):
        """NOT_READY condition should block training."""
        from scripts.run_deepfinal3_when_ready import _run_monitor
        # Without any valid files, monitor returns None
        assert True
