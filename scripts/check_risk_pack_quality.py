#!/usr/bin/env python
"""Risk Pack Quality Gate.

Validates the quality of an exported risk feature pack against its manifest.

Checks (each produces PASS/FAIL):
  1. row_count == manifest unique_keys
  2. no y_true in online mode (check manifest mode field)
  3. business_day/hour_business unique within each target_month
  4. all probability columns in [0,1] or NaN
  5. risk_feature_version startswith "v1."
  6. metric_alignment_status in (PASS, WARN)
  7. module_status columns not all UNKNOWN
  8. target_month rows match monthly_manifest if available
  9. no duplicate ds/hour within same target_month
 10. NaN risk columns only allowed for NO-GO / INSUFFICIENT modules

Overall verdict:
  PASS  -- all checks pass
  WARN  -- all critical checks pass but some non-critical warnings
  FAIL  -- any critical check fails

Critical checks: 1, 3, 4, 5, 6, 7
Non-critical: 2, 8, 9, 10

Usage:
    python scripts/check_risk_pack_quality.py \
      --pack reports/local/risk_modules/risk_feature_pack_2026_01_05/risk_feature_pack.csv \
      --manifest reports/local/risk_modules/risk_feature_pack_2026_01_05/manifest.json \
      --out-dir reports/local/risk_modules/risk_pack_quality_2026_01_05
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_risk_feature_pack_multimonth import ONLINE_COLUMNS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("check_risk_pack_quality")

# -- Constants ----------------------------------------------------------------

# Probability columns: numeric risk columns that must be in [0, 1] or NaN.
PROBABILITY_COLUMNS = [
    "deviation_up_prob",
    "deviation_down_prob",
    "deviation_large_abs_prob",
    "deviation_risk_score",
    "spike_prob",
    "extreme_spike_prob",
    "relative_spike_prob",
    "spike_risk_score",
    "negative_prob",
    "deep_negative_prob",
    "relative_down_prob",
    "negative_risk_score",
]

# Risk columns that may be NaN only for NO-GO / INSUFFICIENT modules.
RISK_COLUMNS = PROBABILITY_COLUMNS  # same set

MODULE_STATUS_COLS = [
    "module_status_delta_supply",
    "module_status_spike",
    "module_status_negative",
]

# Critical vs non-critical check indices (1-based).
CRITICAL_CHECKS = {1, 3, 4, 5, 6, 7}
NON_CRITICAL_CHECKS = {2, 8, 9, 10}


# -- Helpers ------------------------------------------------------------------

def _resolve_path(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _check(name: str, index: int, passed: bool, detail: str = "") -> dict[str, Any]:
    """Build a single check result dict."""
    return {
        "index": index,
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
        "critical": index in CRITICAL_CHECKS,
    }


# -- Core quality checks ------------------------------------------------------

def check_risk_pack_quality(
    pack_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run all quality checks on a risk feature pack.

    Parameters
    ----------
    pack_path : Path to the risk_feature_pack.csv
    manifest_path : Path to the manifest.json

    Returns
    -------
    dict with keys: verdict, checks, n_checks, n_pass, n_fail
    """
    pack_path = Path(pack_path)
    manifest_path = Path(manifest_path)

    # Load inputs.
    df = pd.read_csv(pack_path, encoding="utf-8-sig")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Try to load monthly_manifest.csv (optional).
    monthly_manifest_path = manifest_path.parent / "monthly_manifest.csv"
    monthly_manifest = None
    if monthly_manifest_path.exists():
        monthly_manifest = pd.read_csv(monthly_manifest_path, encoding="utf-8-sig")

    checks: list[dict[str, Any]] = []

    # -- Check 1: row_count == manifest unique_keys ---------------------------
    n_rows = len(df)
    unique_keys = manifest.get("unique_keys", -1)
    check1_pass = n_rows == unique_keys
    checks.append(_check(
        "row_count_matches_manifest_unique_keys",
        1,
        check1_pass,
        f"pack rows={n_rows}, manifest unique_keys={unique_keys}",
    ))

    # -- Check 2: no y_true in online mode ------------------------------------
    mode = manifest.get("mode", "online")
    has_y_true = "y_true" in df.columns
    if mode == "online":
        check2_pass = not has_y_true
        checks.append(_check(
            "no_y_true_in_online_mode",
            2,
            check2_pass,
            f"mode={mode}, y_true present={has_y_true}",
        ))
    else:
        # In eval mode this check is not applicable; auto-pass.
        checks.append(_check(
            "no_y_true_in_online_mode",
            2,
            True,
            f"mode={mode} (not online, check skipped)",
        ))

    # -- Check 3: business_day/hour_business unique within target_month -------
    key_cols = ["business_day", "hour_business"]
    if "target_month" in df.columns:
        dup_mask = df.duplicated(subset=key_cols + ["target_month"], keep=False)
        n_dups = dup_mask.sum()
    else:
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        n_dups = dup_mask.sum()
    check3_pass = n_dups == 0
    checks.append(_check(
        "unique_keys_within_target_month",
        3,
        check3_pass,
        f"duplicate rows={n_dups}",
    ))

    # -- Check 4: all probability columns in [0,1] or NaN ---------------------
    prob_cols_present = [c for c in PROBABILITY_COLUMNS if c in df.columns]
    out_of_range = 0
    bad_cols: list[str] = []
    for col in prob_cols_present:
        series = df[col]
        non_null = series.dropna()
        if len(non_null) > 0:
            violations = ((non_null < 0) | (non_null > 1)).sum()
            if violations > 0:
                out_of_range += int(violations)
                bad_cols.append(f"{col}({violations})")
    check4_pass = out_of_range == 0
    checks.append(_check(
        "probability_columns_in_unit_interval",
        4,
        check4_pass,
        f"out-of-range values={out_of_range}" + (f" in {bad_cols}" if bad_cols else ""),
    ))

    # -- Check 5: risk_feature_version startswith "v1." -----------------------
    if "risk_feature_version" in df.columns:
        versions = df["risk_feature_version"].dropna().unique().tolist()
        all_v1 = all(str(v).startswith("v1.") for v in versions)
        check5_pass = all_v1 and len(versions) > 0
        checks.append(_check(
            "risk_feature_version_starts_with_v1",
            5,
            check5_pass,
            f"versions found={versions}",
        ))
    else:
        checks.append(_check(
            "risk_feature_version_starts_with_v1",
            5,
            False,
            "column 'risk_feature_version' not found in pack",
        ))

    # -- Check 6: metric_alignment_status in (PASS, WARN) ---------------------
    if "metric_alignment_status" in df.columns:
        statuses = df["metric_alignment_status"].dropna().unique().tolist()
        valid_statuses = {"PASS", "WARN"}
        all_valid = all(s in valid_statuses for s in statuses)
        check6_pass = all_valid and len(statuses) > 0
        checks.append(_check(
            "metric_alignment_status_valid",
            6,
            check6_pass,
            f"statuses found={statuses}",
        ))
    else:
        checks.append(_check(
            "metric_alignment_status_valid",
            6,
            False,
            "column 'metric_alignment_status' not found in pack",
        ))

    # -- Check 7: module_status columns not all UNKNOWN -----------------------
    status_cols_present = [c for c in MODULE_STATUS_COLS if c in df.columns]
    if status_cols_present:
        all_unknown = True
        for col in status_cols_present:
            non_unknown = df[col][df[col] != "UNKNOWN"]
            if len(non_unknown) > 0:
                all_unknown = False
                break
        check7_pass = not all_unknown
        checks.append(_check(
            "module_status_not_all_unknown",
            7,
            check7_pass,
            f"status columns checked={status_cols_present}, all_unknown={all_unknown}",
        ))
    else:
        checks.append(_check(
            "module_status_not_all_unknown",
            7,
            False,
            "no module_status columns found in pack",
        ))

    # -- Check 8: target_month rows match monthly_manifest (if available) -----
    if monthly_manifest is not None and "target_month" in df.columns:
        mismatch_details: list[str] = []
        for _, mrow in monthly_manifest.iterrows():
            month = str(mrow["target_month"])
            expected_n = int(mrow["n_rows"])
            actual_n = len(df[df["target_month"] == month])
            if actual_n != expected_n:
                mismatch_details.append(f"{month}: expected={expected_n}, actual={actual_n}")
        check8_pass = len(mismatch_details) == 0
        checks.append(_check(
            "target_month_rows_match_monthly_manifest",
            8,
            check8_pass,
            "; ".join(mismatch_details) if mismatch_details else "all months match",
        ))
    else:
        checks.append(_check(
            "target_month_rows_match_monthly_manifest",
            8,
            True,
            "monthly_manifest.csv not available, check skipped",
        ))

    # -- Check 9: no duplicate ds/hour within same target_month ---------------
    if "ds" in df.columns and "target_month" in df.columns:
        ds_dup_mask = df.duplicated(subset=["ds", "target_month"], keep=False)
        n_ds_dups = ds_dup_mask.sum()
        # Also check hour_business + ds uniqueness within target_month.
        if "hour_business" in df.columns:
            dh_dup_mask = df.duplicated(
                subset=["ds", "hour_business", "target_month"], keep=False
            )
            n_dh_dups = dh_dup_mask.sum()
        else:
            n_dh_dups = 0
        total_dups = n_ds_dups + n_dh_dups
        check9_pass = total_dups == 0
        checks.append(_check(
            "no_duplicate_ds_hour_within_target_month",
            9,
            check9_pass,
            f"ds duplicates={n_ds_dups}, ds+hour duplicates={n_dh_dups}",
        ))
    else:
        checks.append(_check(
            "no_duplicate_ds_hour_within_target_month",
            9,
            True,
            "ds or target_month column not found, check skipped",
        ))

    # -- Check 10: NaN risk columns only for NO-GO / INSUFFICIENT modules -----
    risk_cols_present = [c for c in RISK_COLUMNS if c in df.columns]
    status_cols_for_check = [c for c in MODULE_STATUS_COLS if c in df.columns]
    if risk_cols_present and status_cols_for_check and "target_month" in df.columns:
        invalid_nan_details: list[str] = []
        # Map module status col to its risk cols.
        module_risk_map = {
            "module_status_delta_supply": [
                "deviation_up_prob", "deviation_down_prob",
                "deviation_large_abs_prob", "deviation_risk_score",
            ],
            "module_status_spike": [
                "spike_prob", "extreme_spike_prob",
                "relative_spike_prob", "spike_risk_score",
            ],
            "module_status_negative": [
                "negative_prob", "deep_negative_prob",
                "relative_down_prob", "negative_risk_score",
            ],
        }
        for status_col, risk_col_list in module_risk_map.items():
            if status_col not in df.columns:
                continue
            for col in risk_col_list:
                if col not in df.columns:
                    continue
                # Find rows where risk col is NaN.
                nan_mask = df[col].isna()
                if nan_mask.sum() == 0:
                    continue
                # Check if those rows have a valid reason (NO-GO or INSUFFICIENT).
                nan_rows = df.loc[nan_mask]
                for _, row in nan_rows.iterrows():
                    status_val = str(row.get(status_col, ""))
                    if status_val not in ("NO-GO", "INSUFFICIENT"):
                        month = row.get("target_month", "?")
                        invalid_nan_details.append(
                            f"{col} NaN at month={month} but status={status_val}"
                        )
                        if len(invalid_nan_details) >= 10:
                            break
                    if len(invalid_nan_details) >= 10:
                        break
                if len(invalid_nan_details) >= 10:
                    break
            if len(invalid_nan_details) >= 10:
                break
        check10_pass = len(invalid_nan_details) == 0
        detail_msg = (
            "; ".join(invalid_nan_details[:10])
            if invalid_nan_details
            else "all NaN risk columns have valid NO-GO/INSUFFICIENT status"
        )
        if len(invalid_nan_details) > 10:
            detail_msg += f" ... and {len(invalid_nan_details) - 10} more"
        checks.append(_check(
            "nan_risk_columns_only_for_nogo_or_insufficient",
            10,
            check10_pass,
            detail_msg,
        ))
    else:
        checks.append(_check(
            "nan_risk_columns_only_for_nogo_or_insufficient",
            10,
            True,
            "risk/status columns not available, check skipped",
        ))

    # -- Compute overall verdict ----------------------------------------------
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")

    critical_failures = [
        c for c in checks if c["status"] == "FAIL" and c["critical"]
    ]
    non_critical_failures = [
        c for c in checks if c["status"] == "FAIL" and not c["critical"]
    ]

    if critical_failures:
        verdict = "FAIL"
    elif non_critical_failures:
        verdict = "WARN"
    else:
        verdict = "PASS"

    report = {
        "verdict": verdict,
        "checks": checks,
        "n_checks": len(checks),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "timestamp": datetime.now().isoformat(),
        "pack_path": str(pack_path),
        "manifest_path": str(manifest_path),
    }

    return report


# -- Output writers -----------------------------------------------------------

def write_quality_json(report: dict, out_dir: Path) -> Path:
    """Write risk_pack_quality.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "risk_pack_quality.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Quality JSON -> %s", path)
    return path


def write_quality_md(report: dict, out_dir: Path) -> Path:
    """Write risk_pack_quality.md human-readable report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "risk_pack_quality.md"

    lines: list[str] = []
    lines.append("# Risk Pack Quality Report")
    lines.append("")
    lines.append(f"**Verdict:** {report['verdict']}")
    lines.append(f"**Timestamp:** {report.get('timestamp', 'N/A')}")
    lines.append(f"**Pack:** `{report.get('pack_path', 'N/A')}`")
    lines.append(f"**Manifest:** `{report.get('manifest_path', 'N/A')}`")
    lines.append("")
    lines.append(f"**Checks:** {report['n_pass']}/{report['n_checks']} passed, "
                 f"{report['n_fail']} failed")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    lines.append("| # | Check | Status | Critical | Detail |")
    lines.append("|---|-------|--------|----------|--------|")
    for c in report["checks"]:
        crit = "Yes" if c.get("critical") else "No"
        status_icon = "PASS" if c["status"] == "PASS" else "**FAIL**"
        detail = c.get("detail", "").replace("|", "\\|")
        lines.append(f"| {c['index']} | {c['name']} | {status_icon} | {crit} | {detail} |")
    lines.append("")

    if report["verdict"] == "FAIL":
        lines.append("## Failure Summary")
        lines.append("")
        for c in report["checks"]:
            if c["status"] == "FAIL":
                crit = "(critical)" if c.get("critical") else "(non-critical)"
                lines.append(f"- **{c['name']}** {crit}: {c.get('detail', '')}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Quality MD -> %s", path)
    return path


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Risk Pack Quality Gate -- validate exported risk feature packs",
    )
    parser.add_argument(
        "--pack", type=str, required=True,
        help="Path to risk_feature_pack.csv",
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for quality report files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pack_path = _resolve_path(args.pack)
    manifest_path = _resolve_path(args.manifest)
    out_dir = _resolve_path(args.out_dir)

    if not pack_path.exists():
        logger.error("Pack file not found: %s", pack_path)
        sys.exit(1)
    if not manifest_path.exists():
        logger.error("Manifest file not found: %s", manifest_path)
        sys.exit(1)

    report = check_risk_pack_quality(pack_path, manifest_path)

    # Write outputs.
    write_quality_json(report, out_dir)
    write_quality_md(report, out_dir)

    logger.info(
        "Quality gate verdict: %s (%d/%d checks passed)",
        report["verdict"], report["n_pass"], report["n_checks"],
    )

    # Exit with non-zero on FAIL.
    if report["verdict"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
