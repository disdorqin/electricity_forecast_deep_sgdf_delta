#!/usr/bin/env python
"""Check that no hand-rolled business_day logic exists outside business_time.py.

Scans all .py files under models/ and scripts/ for patterns that suggest
manual business-day alignment instead of using business_time.py.

Patterns detected:
  - dt.normalize() combined with Timedelta(days=1) for business_day
  - hour == 0 used to adjust business_day
  - hour_of_day == 0 used to adjust business_day

Usage:
    python scripts/check_no_handrolled_business_time.py

Exit code 0 = PASS (no violations), 1 = FAIL (violations found).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Patterns that indicate hand-rolled business_day logic
PATTERNS = [
    # Pattern 1: normalize() - Timedelta(days=1)
    re.compile(
        r"normalize\s*\(\s*\)\s*[-]\s*pd\.Timedelta\s*\(\s*days\s*=\s*1\s*\)",
        re.IGNORECASE,
    ),
    # Pattern 2: normalize() - timedelta(days=1) (lowercase)
    re.compile(
        r"normalize\s*\(\s*\)\s*[-]\s*timedelta\s*\(\s*days\s*=\s*1\s*\)",
        re.IGNORECASE,
    ),
    # Pattern 3: hour == 0 ... business_day (within 3 lines)
    # We'll handle this with a line-pair check below
    # Pattern 4: hour_of_day == 0 ... business_day adjustment
    # Also handled with line-pair check
]

# Files that are ALLOWED to contain these patterns (the source of truth itself)
ALLOWED_FILES = {
    "business_time.py",
    "check_no_handrolled_business_time.py",
}


def check_file(filepath: Path) -> list[str]:
    """Check a single file for violations. Returns list of violation descriptions."""
    violations = []

    if filepath.name in ALLOWED_FILES:
        return violations

    try:
        text = filepath.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
    except (UnicodeDecodeError, PermissionError):
        try:
            text = filepath.read_text(encoding="gbk")
            lines = text.splitlines(keepends=True)
        except (UnicodeDecodeError, PermissionError):
            return violations

    content = "".join(lines)

    # Check regex patterns
    for pattern in PATTERNS:
        for match in pattern.finditer(content):
            # Find line number
            line_num = content[: match.start()].count("\n") + 1
            violations.append(
                f"  {filepath}:{line_num}: Hand-rolled pattern found: "
                f"{match.group(0)[:60]}"
            )

    # Check line-pair patterns: hour == 0 near business_day adjustment
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Look for hour == 0 or hour_of_day == 0 patterns
        if re.search(r"hour(?:_of_day)?\s*==\s*0", stripped):
            # Check surrounding lines (±5) for business_day adjustment
            context_start = max(0, i - 5)
            context_end = min(len(lines), i + 6)
            context = "".join(lines[context_start:context_end])
            if re.search(
                r"business_day.*Timedelta|Timedelta.*business_day", context
            ):
                violations.append(
                    f"  {filepath}:{i + 1}: hour==0 with business_day Timedelta "
                    f"adjustment detected"
                )

    return violations


def main():
    scan_dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "scripts",
    ]

    all_violations = []
    files_checked = 0

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in sorted(scan_dir.rglob("*.py")):
            files_checked += 1
            violations = check_file(py_file)
            all_violations.extend(violations)

    print(f"Checked {files_checked} Python files in models/ and scripts/")

    if all_violations:
        print(f"\nFAILED: {len(all_violations)} violation(s) found:")
        for v in all_violations:
            print(v)
        print(
            "\nAll business_day logic must use models/deep_sgdf_delta/business_time.py"
        )
        sys.exit(1)
    else:
        print("PASSED: No hand-rolled business_day logic detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
