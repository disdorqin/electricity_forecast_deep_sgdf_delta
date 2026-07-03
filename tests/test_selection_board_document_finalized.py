"""Track A test: verify Selection Board document is finalized with real data.

Checks:
1. No '_fill in_' placeholders remain.
2. No 'YYYY-MM' date placeholders remain.
3. Real module verdicts are present.
4. Negative recalibration note is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DOC_PATH = PROJECT_ROOT / "docs" / "RISK_MODULES_2_SELECTION_BOARD.md"


def _read_doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


class TestSelectionBoardFinalized:
    """Selection Board document must be fully populated with real results."""

    def test_no_fill_in_placeholders(self):
        doc = _read_doc()
        assert "_fill in_" not in doc, "Document still contains '_fill in_' placeholders"

    def test_no_yyyy_mm_placeholders(self):
        doc = _read_doc()
        # Check for YYYY-MM pattern but not in code blocks or reproduction section
        lines = doc.split("\n")
        in_code_block = False
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            # Skip reproduction section
            if "Reproduction" in line or "export_risk_feature_pack" in line:
                continue
            assert "YYYY-MM" not in line, (
                f"Line {i+1} still contains YYYY-MM placeholder: {line.strip()}"
            )

    def test_no_yyyy_mm_dd_placeholders(self):
        doc = _read_doc()
        lines = doc.split("\n")
        in_code_block = False
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            if "Reproduction" in line or "export_risk_feature_pack" in line:
                continue
            assert "YYYY-MM-DD" not in line, (
                f"Line {i+1} still contains YYYY-MM-DD placeholder: {line.strip()}"
            )

    def test_contains_delta_supply_verdict(self):
        doc = _read_doc()
        assert "DELTA_RISK_ACCEPTABLE" in doc, "Missing DeltaSupply verdict"

    def test_contains_spike_verdict(self):
        doc = _read_doc()
        assert "SPIKE_CHAMPION" in doc, "Missing Spike verdict"

    def test_contains_negative_verdict(self):
        doc = _read_doc()
        assert "NEGATIVE_LOW_VALUE" in doc, "Missing Negative verdict"

    def test_contains_negative_recalibration_note(self):
        doc = _read_doc()
        # Must mention that the LOW_VALUE verdict is due to unsuitable criterion
        assert "base rate" in doc.lower() or "base-rate" in doc.lower(), (
            "Missing explanation about high base rate causing low top-k capture"
        )
        assert "recalibrat" in doc.lower(), (
            "Missing mention of recalibration"
        )

    def test_contains_real_metrics(self):
        doc = _read_doc()
        # Check for actual metric values from backtest
        assert "2.88" in doc or "2.97" in doc, "Missing mean top-10% lift values"
        assert "0.499" in doc or "0.537" in doc, "Missing recall@top20 values"
        assert "0.879" in doc or "0.943" in doc, "Missing monthly AUC values"

    def test_contains_decision_table(self):
        doc = _read_doc()
        assert "KEEP" in doc, "Missing KEEP decision"
        assert "KEEP_AS_AUX" in doc, "Missing KEEP_AS_AUX decision"
        assert "champion" in doc.lower(), "Missing champion role"

    def test_backtest_window_specified(self):
        doc = _read_doc()
        assert "2026-01" in doc, "Missing start month"
        assert "2026-05" in doc, "Missing end month"

    def test_metric_alignment_warn(self):
        doc = _read_doc()
        assert "WARN" in doc, "Missing metric alignment WARN status"
