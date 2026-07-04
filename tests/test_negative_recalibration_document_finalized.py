"""
Test that NEGATIVE_RISK_2_RECALIBRATION.md is fully finalized.

All TODO placeholders must be replaced with real results.
This test prevents template residues from silently shipping in docs.
"""

import pathlib
import pytest

DOC_PATH = pathlib.Path("docs/NEGATIVE_RISK_2_RECALIBRATION.md")


def _read_doc() -> str:
    if not DOC_PATH.exists():
        pytest.skip(f"{DOC_PATH} not found")
    return DOC_PATH.read_text(encoding="utf-8")


class TestNegativeRecalibrationDocumentFinalized:
    """Ensure the recalibration report contains real results, not TODO placeholders."""

    def test_no_todo_placeholder(self):
        """Document must not contain any TODO placeholder."""
        content = _read_doc()
        assert "TODO" not in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md still contains TODO placeholders. "
            "Replace all TODO with real recalibration results."
        )

    def test_no_fill_in_placeholder(self):
        """Document must not contain _fill in_ style placeholders."""
        content = _read_doc()
        assert "_fill in_" not in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md contains '_fill in_' placeholder."
        )
        # Also check for common variant
        assert "___" not in content or "TODO" not in content, (
            "Possible unfilled placeholder detected."
        )

    def test_contains_negative_champion(self):
        """Document must reflect the new NEGATIVE_CHAMPION verdict."""
        content = _read_doc()
        assert "NEGATIVE_CHAMPION" in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must contain NEGATIVE_CHAMPION verdict."
        )

    def test_contains_mean_auc_label(self):
        """Document must contain mean_auc or Mean AUC label."""
        content = _read_doc()
        has_label = "mean_auc" in content or "Mean AUC" in content
        assert has_label, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must contain 'mean_auc' or 'Mean AUC'."
        )

    def test_contains_mean_auc_value(self):
        """Document must contain the actual mean AUC value (~0.946)."""
        content = _read_doc()
        assert "0.946" in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must contain the mean AUC value 0.946."
        )

    def test_contains_mean_f1_value(self):
        """Document must contain the actual mean F1 value (~0.777)."""
        content = _read_doc()
        assert "0.777" in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must contain the mean F1 value 0.777."
        )

    def test_contains_normalized_recall_top10_value(self):
        """Document must contain the mean normalised recall@top10 value (~0.860)."""
        content = _read_doc()
        assert "0.860" in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must contain "
            "mean normalised recall@top10 = 0.860."
        )

    def test_contains_base_rate_aware_explanation(self):
        """Document must explain why base-rate aware metrics are needed."""
        content = _read_doc()
        has_explanation = (
            "base-rate" in content.lower() or "base rate" in content.lower()
        ) and (
            "normalised recall" in content.lower()
            or "normalized recall" in content.lower()
        )
        assert has_explanation, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must explain base-rate awareness "
            "and normalised recall. The old raw top-k metric is invalid for "
            "high base-rate events like negative prices."
        )

    def test_old_verdict_table_present(self):
        """Document must show the old verdict for comparison."""
        content = _read_doc()
        assert "NEGATIVE_LOW_VALUE" in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must document the old verdict "
            "(NEGATIVE_LOW_VALUE) for before/after comparison."
        )

    def test_new_verdict_table_present(self):
        """Document must show the new champion criteria pass table."""
        content = _read_doc()
        # Check for the champion criteria pass table
        assert "mean_auc" in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md must contain the new verdict criteria table."
        )
        assert "0.946" in content, (
            "New verdict table must contain the actual mean_auc value."
        )

    def test_no_template_section(self):
        """Document must not contain the 'How to Populate' template section."""
        content = _read_doc()
        assert "How to Populate" not in content, (
            "NEGATIVE_RISK_2_RECALIBRATION.md still contains the template "
            "'How to Populate' section. Remove it in the finalized report."
        )
