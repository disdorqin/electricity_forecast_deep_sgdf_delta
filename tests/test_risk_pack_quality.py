"""Tests for check_risk_pack_quality -- the Risk Pack Quality Gate.

Covers:
  1. Valid pack -> PASS
  2. Duplicate keys -> FAIL
  3. Probability out of [0,1] -> FAIL
  4. y_true in online mode -> detected
  5. All UNKNOWN status -> FAIL
  6. Wrong version -> FAIL
  7. FAIL alignment status -> FAIL
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_risk_pack_quality import check_risk_pack_quality  # noqa: E402


# -- Synthetic data builders --------------------------------------------------

def _build_valid_pack(n_rows: int = 24) -> pd.DataFrame:
    """Create a minimal valid risk feature pack DataFrame."""
    rng = np.random.RandomState(42)
    base_day = pd.Timestamp("2026-01-05")
    rows = []
    for i in range(n_rows):
        day = base_day + pd.Timedelta(days=i // 24)
        hour = (i % 24) + 1
        ds = day + pd.Timedelta(hours=hour)
        rows.append({
            "business_day": day.strftime("%Y-%m-%d"),
            "hour_business": hour,
            "ds": ds.strftime("%Y-%m-%d %H:%M:%S"),
            "target_month": "2026-01",
            # DeltaSupply probs
            "deviation_up_prob": round(rng.beta(2, 5), 6),
            "deviation_down_prob": round(rng.beta(2, 5), 6),
            "deviation_large_abs_prob": round(rng.beta(2, 5), 6),
            "deviation_risk_score": round(rng.uniform(0, 1), 6),
            # Spike probs
            "spike_prob": round(rng.beta(2, 8), 6),
            "extreme_spike_prob": round(rng.beta(1, 15), 6),
            "relative_spike_prob": round(rng.beta(2, 5), 6),
            "spike_risk_score": round(rng.uniform(0, 1), 6),
            # Negative probs
            "negative_prob": round(rng.beta(1, 10), 6),
            "deep_negative_prob": round(rng.beta(1, 20), 6),
            "relative_down_prob": round(rng.beta(2, 5), 6),
            "negative_risk_score": round(rng.uniform(0, 1), 6),
            # Module status -- at least one non-UNKNOWN
            "module_status_delta_supply": "GO",
            "module_status_spike": "GO",
            "module_status_negative": "GO",
            # Metadata
            "threshold_version": "v1.0.0",
            "risk_feature_version": "v1.1.0",
            "metric_alignment_status": "PASS",
            "metric_alignment_warning_reason": "",
        })
    return pd.DataFrame(rows)


def _build_manifest(df: pd.DataFrame, mode: str = "online") -> dict:
    """Build a manifest dict matching the pack DataFrame."""
    key_cols = ["business_day", "hour_business", "target_month"]
    return {
        "timestamp": "2026-01-05T12:00:00",
        "risk_feature_version": "v1.1.0",
        "threshold_version": "v1.0.0",
        "mode": mode,
        "metric_alignment_status": "PASS",
        "metric_alignment_warning_reason": "",
        "n_rows": len(df),
        "n_months": int(df["target_month"].nunique()) if "target_month" in df.columns else 0,
        "columns": list(df.columns),
        "column_types": {col: str(df[col].dtype) for col in df.columns},
        "key_columns": key_cols,
        "unique_keys": int(df.drop_duplicates(subset=key_cols).shape[0]),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "date_range": {
            "start": str(df["business_day"].min()) if "business_day" in df.columns else None,
            "end": str(df["business_day"].max()) if "business_day" in df.columns else None,
        },
        "target_months": sorted(df["target_month"].unique().tolist()) if "target_month" in df.columns else [],
        "module_nogo_months": {
            "delta_supply": [],
            "spike": [],
            "negative": [],
        },
        "status_sources": {
            "delta_supply": "monthly_verdicts",
            "spike": "monthly_verdicts",
            "negative": "monthly_verdicts",
        },
    }


def _write_pack_and_manifest(
    tmp_path: Path,
    df: pd.DataFrame,
    manifest: dict,
    monthly_manifest: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Write pack CSV and manifest JSON to tmp_path, return (pack_path, manifest_path)."""
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    pack_path = pack_dir / "risk_feature_pack.csv"
    df.to_csv(pack_path, index=False, encoding="utf-8-sig")

    manifest_path = pack_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if monthly_manifest is not None:
        monthly_path = pack_dir / "monthly_manifest.csv"
        monthly_manifest.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    return pack_path, manifest_path


# -- Tests --------------------------------------------------------------------

class TestValidPackPasses:
    """1. Valid pack -> PASS."""

    def test_all_checks_pass(self, tmp_path):
        df = _build_valid_pack()
        manifest = _build_manifest(df)
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        assert report["verdict"] == "PASS"
        assert report["n_fail"] == 0
        assert report["n_pass"] == report["n_checks"]
        for check in report["checks"]:
            assert check["status"] == "PASS", (
                f"Check {check['index']} ({check['name']}) failed: {check['detail']}"
            )


class TestDuplicateKeysFail:
    """2. Duplicate keys -> FAIL (critical check 3)."""

    def test_duplicate_keys_detected(self, tmp_path):
        df = _build_valid_pack(n_rows=12)
        # Append duplicate rows (same business_day, hour_business, target_month).
        dup_rows = df.head(3).copy()
        df_dup = pd.concat([df, dup_rows], ignore_index=True)

        # Manifest reports unique_keys as the count WITHOUT dedup,
        # so unique_keys will be less than n_rows.
        manifest = _build_manifest(df_dup)
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df_dup, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        assert report["verdict"] == "FAIL"
        # Check 3 (unique keys) must fail.
        check3 = [c for c in report["checks"] if c["index"] == 3][0]
        assert check3["status"] == "FAIL"
        # Check 1 (row count vs unique_keys) should also fail since rows > unique_keys.
        check1 = [c for c in report["checks"] if c["index"] == 1][0]
        assert check1["status"] == "FAIL"


class TestProbabilityOutOfRangeFail:
    """3. Probability out of [0,1] -> FAIL (critical check 4)."""

    def test_out_of_range_probability(self, tmp_path):
        df = _build_valid_pack()
        # Set a probability column to an invalid value.
        df.loc[0, "spike_prob"] = 1.5
        df.loc[1, "deviation_up_prob"] = -0.3

        manifest = _build_manifest(df)
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        assert report["verdict"] == "FAIL"
        check4 = [c for c in report["checks"] if c["index"] == 4][0]
        assert check4["status"] == "FAIL"
        assert "out-of-range" in check4["detail"]


class TestYtrueInOnlineModeDetected:
    """4. y_true in online mode -> detected (non-critical check 2)."""

    def test_y_true_in_online_mode(self, tmp_path):
        df = _build_valid_pack()
        # Add y_true column even though mode is online.
        df["y_true"] = np.random.RandomState(99).uniform(0, 500, len(df))

        manifest = _build_manifest(df, mode="online")
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        # y_true in online mode is non-critical, so verdict should be WARN, not FAIL.
        check2 = [c for c in report["checks"] if c["index"] == 2][0]
        assert check2["status"] == "FAIL"
        # Since check 2 is non-critical, overall verdict should be WARN.
        assert report["verdict"] in ("WARN", "FAIL")
        # It should be WARN specifically (no critical failures).
        critical_failures = [c for c in report["checks"]
                             if c["status"] == "FAIL" and c["critical"]]
        if not critical_failures:
            assert report["verdict"] == "WARN"


class TestAllUnknownStatusFail:
    """5. All UNKNOWN status -> FAIL (critical check 7)."""

    def test_all_unknown_module_status(self, tmp_path):
        df = _build_valid_pack()
        # Set all module status columns to UNKNOWN.
        df["module_status_delta_supply"] = "UNKNOWN"
        df["module_status_spike"] = "UNKNOWN"
        df["module_status_negative"] = "UNKNOWN"

        manifest = _build_manifest(df)
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        assert report["verdict"] == "FAIL"
        check7 = [c for c in report["checks"] if c["index"] == 7][0]
        assert check7["status"] == "FAIL"


class TestWrongVersionFail:
    """6. Wrong version -> FAIL (critical check 5)."""

    def test_version_not_v1(self, tmp_path):
        df = _build_valid_pack()
        # Set version to something that doesn't start with "v1."
        df["risk_feature_version"] = "v2.0.0"

        manifest = _build_manifest(df)
        manifest["risk_feature_version"] = "v2.0.0"
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        assert report["verdict"] == "FAIL"
        check5 = [c for c in report["checks"] if c["index"] == 5][0]
        assert check5["status"] == "FAIL"


class TestFailAlignmentStatusFail:
    """7. FAIL alignment status -> FAIL (critical check 6)."""

    def test_fail_alignment_status(self, tmp_path):
        df = _build_valid_pack()
        df["metric_alignment_status"] = "FAIL"

        manifest = _build_manifest(df)
        manifest["metric_alignment_status"] = "FAIL"
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        assert report["verdict"] == "FAIL"
        check6 = [c for c in report["checks"] if c["index"] == 6][0]
        assert check6["status"] == "FAIL"


class TestWarnVerdict:
    """Extra: non-critical failure yields WARN, not FAIL."""

    def test_warn_verdict_for_non_critical_only(self, tmp_path):
        df = _build_valid_pack()
        # Add y_true in online mode (non-critical check 2).
        df["y_true"] = 42.0
        manifest = _build_manifest(df, mode="online")
        pack_path, manifest_path = _write_pack_and_manifest(tmp_path, df, manifest)

        report = check_risk_pack_quality(pack_path, manifest_path)

        # Only non-critical check 2 should fail.
        critical_failures = [c for c in report["checks"]
                             if c["status"] == "FAIL" and c["critical"]]
        assert len(critical_failures) == 0
        assert report["verdict"] == "WARN"


class TestMonthlyManifestCheck:
    """Extra: monthly_manifest mismatch is detected (non-critical check 8)."""

    def test_monthly_manifest_mismatch(self, tmp_path):
        df = _build_valid_pack(n_rows=24)
        manifest = _build_manifest(df)

        # Create a monthly_manifest that claims different row counts.
        monthly_df = pd.DataFrame([{
            "target_month": "2026-01",
            "n_rows": 999,  # wrong!
            "module_status_delta_supply": "GO",
            "module_status_spike": "GO",
            "module_status_negative": "GO",
        }])

        pack_path, manifest_path = _write_pack_and_manifest(
            tmp_path, df, manifest, monthly_manifest=monthly_df,
        )

        report = check_risk_pack_quality(pack_path, manifest_path)

        check8 = [c for c in report["checks"] if c["index"] == 8][0]
        assert check8["status"] == "FAIL"
        # Non-critical, so verdict should be WARN.
        assert report["verdict"] == "WARN"
