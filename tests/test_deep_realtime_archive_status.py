"""Tests for deep realtime model archive status."""
from __future__ import annotations

from pathlib import Path
import pytest


class TestArchiveDocs:
    """Archive document tests."""

    @pytest.fixture(scope="class")
    def project_root(self):
        return Path(__file__).resolve().parent.parent

    def test_archive_decision_exists(self, project_root):
        doc = project_root / "docs" / "DEEP_REALTIME_MODEL_ARCHIVE_DECISION.md"
        assert doc.exists(), "Archive decision doc missing"
        content = doc.read_text(encoding="utf-8")
        assert "ARCHIVED" in content

    def test_handoff_no_misleading_claims(self, project_root):
        doc = project_root / "docs" / "DEEP_REALTIME_MODEL_FINAL_HANDOFF.md"
        assert doc.exists()
        content = doc.read_text(encoding="utf-8")
        assert "MODEL_NO_GO" in content, "Handoff must contain MODEL_NO_GO"

    def test_results_no_expected_15_18(self, project_root):
        doc = project_root / "docs" / "DEEP_REALTIME_FINAL_RESULTS.md"
        assert doc.exists()
        content = doc.read_text(encoding="utf-8")
        assert "expected 15-18" not in content.lower(), \
            "Do not claim expected 15-18 without real SGDFNet"

    def test_diagnosis_no_old_next_steps(self, project_root):
        doc = project_root / "docs" / "DEEPFINAL_4_FAILURE_DIAGNOSIS_REPORT.md"
        assert doc.exists()
        content = doc.read_text(encoding="utf-8")
        assert "下一步建议" not in content, \
            "Should not still contain old next-steps section"
        assert "ARCHIVE_DEEP_MODEL" in content

    def test_archive_script_runnable(self, project_root):
        script = project_root / "scripts" / "check_deep_realtime_archive_status.py"
        assert script.exists()
        # Module importable
        import importlib
        spec = importlib.util.spec_from_file_location("check_archive", script)
        assert spec is not None


class TestArchiveStatusCheck:
    """Archive status check logic."""

    def test_check_function_returns_dict(self):
        from scripts.check_deep_realtime_archive_status import check
        result = check()
        assert isinstance(result, dict)
        assert "checks" in result
        assert "all_passed" in result
        assert result["all_passed"], f"Archive checks failed: {result}"
