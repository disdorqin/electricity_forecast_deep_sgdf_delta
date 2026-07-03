#!/usr/bin/env python
"""Check the archive status of the deep realtime model project.

Verifies that all documentation is correctly updated for the archive decision.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("check_archive_status")


def check() -> dict:
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checks": {},
        "all_passed": False,
    }

    # Check 1: Archive decision doc exists
    archive_doc = PROJECT_ROOT / "docs" / "DEEP_REALTIME_MODEL_ARCHIVE_DECISION.md"
    exists = archive_doc.exists()
    results["checks"]["archive_decision_doc_exists"] = exists
    if exists:
        content = archive_doc.read_text(encoding="utf-8")
        results["checks"]["archive_decision_contains_ARCHIVED"] = "ARCHIVED" in content
        results["checks"]["archive_decision_contains_reopening_conditions"] = "重新打开" in content or "reopen" in content.lower()
    else:
        results["checks"]["archive_decision_contains_ARCHIVED"] = False
        results["checks"]["archive_decision_contains_reopening_conditions"] = False

    # Check 2: DeepFinal-4 report contains ARCHIVE_DEEP_MODEL
    diag_doc = PROJECT_ROOT / "docs" / "DEEPFINAL_4_FAILURE_DIAGNOSIS_REPORT.md"
    if diag_doc.exists():
        content = diag_doc.read_text(encoding="utf-8")
        results["checks"]["diagnosis_contains_ARCHIVE_DEEP_MODEL"] = "ARCHIVE_DEEP_MODEL" in content
        results["checks"]["diagnosis_no_longer_suggests_residual_baseline_lab"] = \
            "运行 residual baseline lab" not in content
    else:
        results["checks"]["diagnosis_contains_ARCHIVE_DEEP_MODEL"] = False
        results["checks"]["diagnosis_no_longer_suggests_residual_baseline_lab"] = False

    # Check 3: Final results contains MODEL_NO_GO
    results_doc = PROJECT_ROOT / "docs" / "DEEP_REALTIME_FINAL_RESULTS.md"
    if results_doc.exists():
        content = results_doc.read_text(encoding="utf-8")
        results["checks"]["results_contains_MODEL_NO_GO"] = "MODEL_NO_GO" in content
        results["checks"]["results_no_expected_15_18"] = "expected 15-18" not in content.lower()
    else:
        results["checks"]["results_contains_MODEL_NO_GO"] = False
        results["checks"]["results_no_expected_15_18"] = False

    # Check 4: Handoff doc has final verdict
    handoff_doc = PROJECT_ROOT / "docs" / "DEEP_REALTIME_MODEL_FINAL_HANDOFF.md"
    if handoff_doc.exists():
        content = handoff_doc.read_text(encoding="utf-8")
        results["checks"]["handoff_contains_MODEL_NO_GO"] = "MODEL_NO_GO" in content
        results["checks"]["handoff_no_misleading_claims"] = \
            "预期" not in content.split("## 6. 是否达到目标")[-1] if "## 6. 是否达到目标" in content else True
    else:
        results["checks"]["handoff_contains_MODEL_NO_GO"] = False
        results["checks"]["handoff_no_misleading_claims"] = False

    # Summary
    all_checks = list(results["checks"].values())
    results["all_passed"] = all(all_checks)
    results["n_checks"] = len(all_checks)
    results["n_passed"] = sum(1 for c in all_checks if c)

    return results


def main():
    results = check()
    out_dir = PROJECT_ROOT / "reports" / "local" / "deep_final" / "archive_status"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "archive_status.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    status = "PASS" if results["all_passed"] else "FAIL"
    lines = [
        "# Deep Realtime Model — Archive Status Check",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        f"## Status: **{status}**",
        f"- Checks passed: {results['n_passed']}/{results['n_checks']}",
        "",
        "## Checks",
    ]
    for check_name, passed in results["checks"].items():
        symbol = "✅" if passed else "❌"
        lines.append(f"- {symbol} {check_name}")

    (out_dir / "archive_status.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nArchive Status: {status}")
    print(f"  {results['n_passed']}/{results['n_checks']} checks passed")
    for check_name, passed in results["checks"].items():
        print(f"  {'✅' if passed else '❌'} {check_name}")


if __name__ == "__main__":
    main()
