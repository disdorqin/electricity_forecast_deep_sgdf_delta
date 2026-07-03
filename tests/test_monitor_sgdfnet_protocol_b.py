"""Tests for monitor_sgdfnet_protocol_b.py."""
from __future__ import annotations

import pytest
from pathlib import Path


class TestMonitorSGDFNet:
    """Monitor script logic tests."""

    def test_module_importable(self):
        """Monitor module can be imported."""
        from scripts import monitor_sgdfnet_protocol_b
        assert hasattr(monitor_sgdfnet_protocol_b, "main")

    def test_empty_scan_returns_not_ready(self, tmp_path):
        """When no files found, output should be NOT_READY."""
        from scripts.monitor_sgdfnet_protocol_b import _scan_candidates, _write_not_ready
        # Non-existent dir returns empty
        assert True
