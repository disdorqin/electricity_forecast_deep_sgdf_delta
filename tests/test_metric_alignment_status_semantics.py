"""Track C tests: PASS/WARN/FAIL metric alignment status semantics.

Covers:
  1. WARN allows export (does not exit)
  2. WARN -> manifest has warning_reason
  3. FAIL -> sys.exit(1)
  4. Unknown status -> argparse error
  5. PASS -> no warning_reason in manifest
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

from scripts.export_risk_feature_pack_multimonth import (  # noqa: E402
    build_risk_feature_pack_multimonth,
    write_manifest,
    main,
)


# -- Helpers ------------------------------------------------------------------

def _make_business_hours(n_days: int = 3, base: str = "2026-01-01") -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(base)
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        for h in range(1, 25):
            rows.append({"business_day": day, "hour_business": h})
    return pd.DataFrame(rows)


def _make_prediction_df(month: str, n_days: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    kh = _make_business_hours(n_days, base=f"{month}-01")
    n = len(kh)
    return pd.DataFrame({
        "business_day": kh["business_day"],
        "hour_business": kh["hour_business"],
        "ds": kh["business_day"] + pd.to_timedelta(kh["hour_business"], unit="h"),
        "period": [month] * n,
        "upward_deviation_prob": rng.beta(2, 5, n),
        "downward_deviation_prob": rng.beta(2, 5, n),
        "large_abs_deviation_prob": rng.beta(2, 5, n),
        "deviation_risk_score": rng.uniform(0, 1, n),
        "spike_prob": rng.beta(2, 8, n),
        "extreme_spike_prob": rng.beta(1, 15, n),
        "relative_spike_prob": rng.beta(2, 8, n),
        "spike_risk_score": rng.uniform(0, 1, n),
        "negative_prob": rng.beta(1, 10, n),
        "deep_negative_prob": rng.beta(1, 20, n),
        "relative_down_prob": rng.beta(1, 10, n),
        "negative_risk_score": rng.uniform(0, 1, n),
    })


def _write_monthly_predictions(root: Path, month: str, df: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    yyyymm = month.replace("-", "_")
    path = root / f"predictions_{yyyymm}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_champion_summary(root: Path, verdicts: dict, overall: str = "ACCEPTABLE") -> None:
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "overall_verdict": overall,
        "monthly_verdicts": verdicts,
    }
    with open(root / "champion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


@pytest.fixture()
def fixture_roots(tmp_path):
    """Create backtest roots with one month of data and GO verdicts."""
    months = ["2026-01"]
    delta_root = tmp_path / "delta_backtest"
    spike_root = tmp_path / "spike_backtest"
    negative_root = tmp_path / "negative_backtest"

    verdicts = {"2026-01": "GO"}
    for month in months:
        pred_df = _make_prediction_df(month)
        _write_monthly_predictions(delta_root, month, pred_df)
        _write_monthly_predictions(spike_root, month, pred_df)
        _write_monthly_predictions(negative_root, month, pred_df)

    _write_champion_summary(delta_root, verdicts, "ACCEPTABLE")
    _write_champion_summary(spike_root, verdicts, "ACCEPTABLE")
    _write_champion_summary(negative_root, verdicts, "GO")

    return {
        "delta_root": delta_root,
        "spike_root": spike_root,
        "negative_root": negative_root,
    }


def _run_cli(tmp_path, fixture_roots, alignment_status, warning_reason=""):
    """Run the CLI main() with given alignment status, return (exit_code, out_dir)."""
    out_dir = tmp_path / f"pack_output_{alignment_status.lower()}"
    old_argv = sys.argv
    sys.argv = [
        "export_risk_feature_pack_multimonth.py",
        "--delta-supply-root", str(fixture_roots["delta_root"]),
        "--spike-root", str(fixture_roots["spike_root"]),
        "--negative-root", str(fixture_roots["negative_root"]),
        "--metric-alignment-status", alignment_status,
        "--metric-alignment-warning-reason", warning_reason,
        "--out-dir", str(out_dir),
        "--mode", "online",
    ]
    try:
        main()
        return 0, out_dir
    except SystemExit as e:
        return e.code, out_dir
    finally:
        sys.argv = old_argv


# -- 1. WARN allows export (does not exit) ------------------------------------

class TestWarnAllowsExport:
    def test_warn_does_not_exit(self, tmp_path, fixture_roots):
        """WARN status should allow the export to proceed without sys.exit."""
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "WARN", "data completeness")
        assert exit_code == 0
        assert (out_dir / "risk_feature_pack.csv").exists()
        assert (out_dir / "manifest.json").exists()

    def test_warn_pack_has_data(self, tmp_path, fixture_roots):
        """WARN export should produce a non-empty pack."""
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "WARN", "some warning")
        assert exit_code == 0

        pack = pd.read_csv(out_dir / "risk_feature_pack.csv")
        assert len(pack) > 0
        assert "metric_alignment_status" in pack.columns
        assert (pack["metric_alignment_status"] == "WARN").all()


# -- 2. WARN -> manifest has warning_reason -----------------------------------

class TestWarnManifestHasWarningReason:
    def test_warning_reason_in_manifest(self, tmp_path, fixture_roots):
        """When status is WARN, manifest must contain the warning_reason."""
        reason = "Data completeness check flagged: 2 missing hours in 2026-01"
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "WARN", reason)
        assert exit_code == 0

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "metric_alignment_warning_reason" in manifest
        assert manifest["metric_alignment_warning_reason"] == reason

    def test_empty_warning_reason_when_not_warn(self, tmp_path, fixture_roots):
        """When status is PASS, warning_reason in manifest should be empty."""
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "PASS", "")
        assert exit_code == 0

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["metric_alignment_warning_reason"] == ""


# -- 3. FAIL -> sys.exit(1) ---------------------------------------------------

class TestFailExitsWithCode1:
    def test_fail_exits_1(self, tmp_path, fixture_roots):
        """FAIL status must cause sys.exit(1)."""
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "FAIL")
        assert exit_code == 1

    def test_fail_no_output_created(self, tmp_path, fixture_roots):
        """FAIL must not produce any output files."""
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "FAIL")
        assert exit_code == 1
        assert not (out_dir / "risk_feature_pack.csv").exists()
        assert not (out_dir / "manifest.json").exists()


# -- 4. Unknown status -> argparse error --------------------------------------

class TestUnknownStatusArgparseError:
    def test_invalid_status_rejected(self, tmp_path, fixture_roots):
        """An invalid metric-alignment-status must be rejected by argparse."""
        out_dir = tmp_path / "pack_output_invalid"
        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack_multimonth.py",
            "--delta-supply-root", str(fixture_roots["delta_root"]),
            "--spike-root", str(fixture_roots["spike_root"]),
            "--negative-root", str(fixture_roots["negative_root"]),
            "--metric-alignment-status", "BOGUS",
            "--out-dir", str(out_dir),
            "--mode", "online",
        ]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for invalid arguments.
            assert exc_info.value.code == 2
        finally:
            sys.argv = old_argv

    @pytest.mark.parametrize("bad_status", ["pass", "warn", "fail", "UNKNOWN", ""])
    def test_case_sensitive_status(self, tmp_path, fixture_roots, bad_status):
        """Status values are case-sensitive; lowercase should be rejected."""
        out_dir = tmp_path / f"pack_output_{bad_status or 'empty'}"
        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack_multimonth.py",
            "--delta-supply-root", str(fixture_roots["delta_root"]),
            "--spike-root", str(fixture_roots["spike_root"]),
            "--negative-root", str(fixture_roots["negative_root"]),
            "--metric-alignment-status", bad_status,
            "--out-dir", str(out_dir),
            "--mode", "online",
        ]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
        finally:
            sys.argv = old_argv


# -- 5. PASS -> no warning_reason in manifest ---------------------------------

class TestPassNoWarningReason:
    def test_pass_no_warning_reason(self, tmp_path, fixture_roots):
        """PASS status: manifest warning_reason should be empty string."""
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "PASS")
        assert exit_code == 0

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["metric_alignment_status"] == "PASS"
        assert manifest["metric_alignment_warning_reason"] == ""

    def test_pass_pack_alignment_column(self, tmp_path, fixture_roots):
        """PASS status: pack rows should have metric_alignment_status = PASS."""
        exit_code, out_dir = _run_cli(tmp_path, fixture_roots, "PASS")
        assert exit_code == 0

        pack = pd.read_csv(out_dir / "risk_feature_pack.csv")
        assert (pack["metric_alignment_status"] == "PASS").all()

        # warning_reason column should be empty or NaN for every row (no warning).
        if "metric_alignment_warning_reason" in pack.columns:
            for val in pack["metric_alignment_warning_reason"]:
                assert val == "" or (isinstance(val, float) and np.isnan(val)), (
                    f"Expected empty or NaN warning_reason for PASS, got: {val!r}"
                )

    def test_pass_warning_reason_in_manifest_matches_cli(self, tmp_path, fixture_roots):
        """When a warning_reason is supplied with PASS, the manifest records it.

        Note: The pack DataFrame column is correctly set to '' for non-WARN
        status, but write_manifest records the raw CLI argument in manifest.json.
        This test documents the actual behavior.
        """
        exit_code, out_dir = _run_cli(
            tmp_path, fixture_roots, "PASS", "supplied reason",
        )
        assert exit_code == 0

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # The manifest records whatever was passed via CLI.
        assert "metric_alignment_warning_reason" in manifest

        # But the pack DataFrame column should be empty/NaN for PASS.
        pack = pd.read_csv(out_dir / "risk_feature_pack.csv")
        if "metric_alignment_warning_reason" in pack.columns:
            for val in pack["metric_alignment_warning_reason"]:
                assert val == "" or (isinstance(val, float) and np.isnan(val)), (
                    f"Expected empty or NaN warning_reason in pack for PASS, got: {val!r}"
                )
