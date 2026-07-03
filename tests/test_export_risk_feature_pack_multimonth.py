"""Tests for export_risk_feature_pack_multimonth script.

Covers:
  1. Online mode excludes y_true / rt_actual
  2. Eval mode includes y_true
  3. Rows unique by (business_day, hour_business) within each target_month
  4. Manifest has required fields including risk_feature_version = "v1.1.0"
  5. FAIL alignment prevents export
  6. Module NO-GO produces NaN fields
  7. Script runs end-to-end with fixture data
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
    KEY_COLUMNS,
    ONLINE_COLUMNS,
    EVAL_EXTRA_COLUMNS,
    RISK_FEATURE_VERSION,
    build_risk_feature_pack_multimonth,
    main,
    write_manifest,
    _discover_monthly_csvs,
    _is_nogo_verdict,
)


# -- Fixtures -----------------------------------------------------------------

def _make_business_hours(n_days: int = 3, base: str = "2026-01-01") -> pd.DataFrame:
    """Create a DataFrame of (business_day, hour_business) for n_days x 24 hours."""
    rows = []
    start = pd.Timestamp(base)
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        for h in range(1, 25):
            rows.append({"business_day": day, "hour_business": h})
    return pd.DataFrame(rows)


def _write_monthly_predictions(
    root: Path,
    month: str,
    df: pd.DataFrame,
) -> None:
    """Write a monthly predictions CSV to root/predictions_YYYY_MM.csv."""
    root.mkdir(parents=True, exist_ok=True)
    yyyymm = month.replace("-", "_")
    path = root / f"predictions_{yyyymm}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _make_delta_predictions(month: str, n_days: int = 3, rng_seed: int = 42) -> pd.DataFrame:
    """Synthetic DeltaSupply predictions for a single month."""
    rng = np.random.RandomState(rng_seed)
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
    })


def _make_spike_predictions(month: str, n_days: int = 3, rng_seed: int = 43) -> pd.DataFrame:
    """Synthetic SpikeRisk predictions for a single month."""
    rng = np.random.RandomState(rng_seed)
    kh = _make_business_hours(n_days, base=f"{month}-01")
    n = len(kh)
    return pd.DataFrame({
        "business_day": kh["business_day"],
        "hour_business": kh["hour_business"],
        "ds": kh["business_day"] + pd.to_timedelta(kh["hour_business"], unit="h"),
        "period": [month] * n,
        "spike_prob": rng.beta(2, 8, n),
        "extreme_spike_prob": rng.beta(1, 15, n),
        "spike_risk_score": rng.uniform(0, 1, n),
    })


def _make_negative_predictions(month: str, n_days: int = 3, rng_seed: int = 44) -> pd.DataFrame:
    """Synthetic NegativeRisk predictions for a single month."""
    rng = np.random.RandomState(rng_seed)
    kh = _make_business_hours(n_days, base=f"{month}-01")
    n = len(kh)
    return pd.DataFrame({
        "business_day": kh["business_day"],
        "hour_business": kh["hour_business"],
        "ds": kh["business_day"] + pd.to_timedelta(kh["hour_business"], unit="h"),
        "period": [month] * n,
        "negative_prob": rng.beta(1, 10, n),
        "deep_negative_prob": rng.beta(1, 20, n),
        "negative_risk_score": rng.uniform(0, 1, n),
    })


def _write_champion_summary(root: Path, verdicts: dict, overall: str = "ACCEPTABLE") -> None:
    """Write a champion_summary.json to the backtest root."""
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "overall_verdict": overall,
        "mean_monthly_improvement_pp": 0.005,
        "n_months": len(verdicts),
        "n_successful": len(verdicts),
        "monthly_verdicts": verdicts,
    }
    with open(root / "champion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


@pytest.fixture()
def multimonth_fixture(tmp_path):
    """Create multi-month backtest directories with 2 months of data each."""
    months = ["2026-01", "2026-02"]

    delta_root = tmp_path / "delta_supply_backtest"
    spike_root = tmp_path / "spike_risk_backtest"
    negative_root = tmp_path / "negative_risk_backtest"

    delta_verdicts = {}
    spike_verdicts = {}
    negative_verdicts = {}

    for month in months:
        delta_df = _make_delta_predictions(month, n_days=3, rng_seed=42)
        spike_df = _make_spike_predictions(month, n_days=3, rng_seed=43)
        negative_df = _make_negative_predictions(month, n_days=3, rng_seed=44)

        _write_monthly_predictions(delta_root, month, delta_df)
        _write_monthly_predictions(spike_root, month, spike_df)
        _write_monthly_predictions(negative_root, month, negative_df)

        delta_verdicts[month] = "GO"
        spike_verdicts[month] = "ACCEPTABLE"
        negative_verdicts[month] = "GO"

    _write_champion_summary(delta_root, delta_verdicts, "ACCEPTABLE")
    _write_champion_summary(spike_root, spike_verdicts, "ACCEPTABLE")
    _write_champion_summary(negative_root, negative_verdicts, "GO")

    return {
        "delta_root": delta_root,
        "spike_root": spike_root,
        "negative_root": negative_root,
        "months": months,
    }


# -- 1. Online mode excludes y_true / rt_actual -------------------------------

class TestOnlineModeExcludesYtrue:
    def test_no_y_true_column(self, multimonth_fixture):
        pack, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        assert "y_true" not in pack.columns
        assert "rt_actual" not in pack.columns

    def test_online_columns_only(self, multimonth_fixture):
        pack, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        for col in pack.columns:
            assert col in ONLINE_COLUMNS, f"Unexpected column in online mode: {col}"


# -- 2. Eval mode includes y_true --------------------------------------------

class TestEvalModeIncludesYtrue:
    def test_y_true_present_in_eval(self, tmp_path, multimonth_fixture):
        """Add y_true to delta predictions and verify eval mode includes it."""
        # Rewrite delta predictions with y_true column.
        for month in multimonth_fixture["months"]:
            delta_df = _make_delta_predictions(month, n_days=3, rng_seed=42)
            delta_df["y_true"] = np.random.RandomState(50).uniform(0, 500, len(delta_df))
            _write_monthly_predictions(multimonth_fixture["delta_root"], month, delta_df)

        pack, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="eval",
            metric_alignment_status="PASS",
        )
        assert "y_true" in pack.columns


# -- 3. Rows unique by (business_day, hour_business) --------------------------

class TestRowUniqueness:
    def test_unique_keys_within_month(self, multimonth_fixture):
        pack, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        # Check uniqueness within each target_month.
        for month, group in pack.groupby("target_month"):
            n_rows = len(group)
            n_unique = group.drop_duplicates(subset=KEY_COLUMNS).shape[0]
            assert n_rows == n_unique, (
                f"Month {month}: expected {n_unique} unique rows but got {n_rows}"
            )

    def test_target_month_column_present(self, multimonth_fixture):
        pack, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        assert "target_month" in pack.columns
        assert pack["target_month"].nunique() == len(multimonth_fixture["months"])


# -- 4. Manifest has required fields including version ------------------------

class TestManifest:
    REQUIRED_MANIFEST_FIELDS = [
        "risk_feature_version",
        "threshold_version",
        "mode",
        "metric_alignment_status",
        "n_rows",
        "n_months",
        "columns",
        "key_columns",
        "unique_keys",
        "target_months",
        "module_nogo_months",
    ]

    def test_manifest_has_required_fields(self, tmp_path, multimonth_fixture):
        pack, monthly_manifest = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        write_manifest(tmp_path, pack, "online", "PASS", monthly_manifest)

        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for field in self.REQUIRED_MANIFEST_FIELDS:
            assert field in manifest, f"Missing manifest field: {field}"

    def test_risk_feature_version_is_v1_1_0(self, tmp_path, multimonth_fixture):
        pack, monthly_manifest = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        write_manifest(tmp_path, pack, "online", "PASS", monthly_manifest)

        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["risk_feature_version"] == "v1.1.0"
        assert RISK_FEATURE_VERSION == "v1.1.0"

    def test_monthly_manifest_csv_created(self, tmp_path, multimonth_fixture):
        pack, monthly_manifest = build_risk_feature_pack_multimonth(
            delta_supply_root=multimonth_fixture["delta_root"],
            spike_root=multimonth_fixture["spike_root"],
            negative_root=multimonth_fixture["negative_root"],
            mode="online",
            metric_alignment_status="PASS",
        )
        write_manifest(tmp_path, pack, "online", "PASS", monthly_manifest)

        monthly_csv = tmp_path / "monthly_manifest.csv"
        assert monthly_csv.exists()
        df = pd.read_csv(monthly_csv)
        assert len(df) == len(multimonth_fixture["months"])


# -- 5. FAIL alignment prevents export ----------------------------------------

class TestFailAlignmentPreventsExport:
    def test_fail_status_exits(self, tmp_path, multimonth_fixture):
        """Script must exit(1) when metric_alignment_status is FAIL."""
        out_dir = tmp_path / "pack_output"

        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack_multimonth.py",
            "--delta-supply-root", str(multimonth_fixture["delta_root"]),
            "--spike-root", str(multimonth_fixture["spike_root"]),
            "--negative-root", str(multimonth_fixture["negative_root"]),
            "--metric-alignment-status", "FAIL",
            "--out-dir", str(out_dir),
            "--mode", "online",
        ]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        finally:
            sys.argv = old_argv

        # Output should NOT have been created.
        assert not (out_dir / "risk_feature_pack.csv").exists()


# -- 6. Module NO-GO produces NaN fields --------------------------------------

class TestModuleNogoProducesNaN:
    def test_nogo_month_has_nan_risk_fields(self, tmp_path):
        """When a module is NO-GO for a month, its risk columns should be NaN."""
        months = ["2026-01", "2026-02"]

        delta_root = tmp_path / "delta_backtest"
        spike_root = tmp_path / "spike_backtest"
        negative_root = tmp_path / "negative_backtest"

        # DeltaSupply: month 1 is GO, month 2 is NO-GO.
        delta_verdicts = {"2026-01": "GO", "2026-02": "NO-GO"}
        spike_verdicts = {"2026-01": "GO", "2026-02": "GO"}
        negative_verdicts = {"2026-01": "GO", "2026-02": "GO"}

        for month in months:
            _write_monthly_predictions(delta_root, month, _make_delta_predictions(month))
            _write_monthly_predictions(spike_root, month, _make_spike_predictions(month))
            _write_monthly_predictions(negative_root, month, _make_negative_predictions(month))

        _write_champion_summary(delta_root, delta_verdicts, "LOW_VALUE")
        _write_champion_summary(spike_root, spike_verdicts, "GO")
        _write_champion_summary(negative_root, negative_verdicts, "GO")

        pack, monthly_manifest = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        # Check that month 2 delta supply columns are NaN.
        month2 = pack[pack["target_month"] == "2026-02"]
        delta_cols = ["deviation_up_prob", "deviation_down_prob",
                      "deviation_large_abs_prob", "deviation_risk_score"]
        for col in delta_cols:
            if col in month2.columns:
                assert month2[col].isna().all(), (
                    f"Expected {col} to be all NaN for NO-GO month 2026-02"
                )

        # Check module_status column.
        assert (month2["module_status_delta_supply"] == "NO-GO").all()

        # Month 1 should NOT be NaN.
        month1 = pack[pack["target_month"] == "2026-01"]
        for col in delta_cols:
            if col in month1.columns:
                assert month1[col].notna().any(), (
                    f"Expected {col} to have non-NaN values for GO month 2026-01"
                )


# -- 7. Script runs end-to-end with fixture data ------------------------------

class TestEndToEnd:
    def test_full_cli_online(self, tmp_path, multimonth_fixture):
        """Full CLI invocation in online mode produces expected files."""
        out_dir = tmp_path / "pack_output"

        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack_multimonth.py",
            "--delta-supply-root", str(multimonth_fixture["delta_root"]),
            "--spike-root", str(multimonth_fixture["spike_root"]),
            "--negative-root", str(multimonth_fixture["negative_root"]),
            "--metric-alignment-status", "PASS",
            "--out-dir", str(out_dir),
            "--mode", "online",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        # Check outputs exist.
        assert (out_dir / "risk_feature_pack.csv").exists()
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "monthly_manifest.csv").exists()

        # Check CSV is parseable.
        pack = pd.read_csv(out_dir / "risk_feature_pack.csv")
        assert len(pack) > 0
        assert "target_month" in pack.columns
        assert "risk_feature_version" in pack.columns
        assert "module_status_delta_supply" in pack.columns

        # Check manifest.
        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["n_rows"] == len(pack)
        assert manifest["risk_feature_version"] == "v1.1.0"

    def test_full_cli_eval(self, tmp_path, multimonth_fixture):
        """Full CLI invocation in eval mode includes y_true."""
        # Add y_true to delta predictions.
        for month in multimonth_fixture["months"]:
            delta_df = _make_delta_predictions(month, n_days=3, rng_seed=42)
            delta_df["y_true"] = np.random.RandomState(50).uniform(0, 500, len(delta_df))
            _write_monthly_predictions(multimonth_fixture["delta_root"], month, delta_df)

        out_dir = tmp_path / "pack_output_eval"

        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack_multimonth.py",
            "--delta-supply-root", str(multimonth_fixture["delta_root"]),
            "--spike-root", str(multimonth_fixture["spike_root"]),
            "--negative-root", str(multimonth_fixture["negative_root"]),
            "--metric-alignment-status", "PASS",
            "--out-dir", str(out_dir),
            "--mode", "eval",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        pack = pd.read_csv(out_dir / "risk_feature_pack.csv")
        assert "y_true" in pack.columns
