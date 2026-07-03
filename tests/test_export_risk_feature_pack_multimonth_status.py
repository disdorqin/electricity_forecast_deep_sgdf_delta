"""Track B tests: verdict.json reading + verdict->status mapping.

Covers:
  1. Only verdict.json exists -> can read status correctly
  2. CHAMPION/ACCEPTABLE verdicts map to GO
  3. LOW_VALUE maps to LOW_VALUE
  4. NO_GO maps to NO-GO and risk columns become NaN
  5. Not allowed to have all module_status UNKNOWN and export successfully
  6. online mode does not contain y_true
  7. status_source appears in manifest
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
    _load_verdict_summary,
    _normalize_verdict_to_status,
    build_risk_feature_pack_multimonth,
    write_manifest,
    ONLINE_COLUMNS,
    KEY_COLUMNS,
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
    """Create a minimal prediction DataFrame with all required columns."""
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


def _write_verdict_json(root: Path, verdict_data: dict) -> None:
    """Write a verdict.json to the backtest root."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, ensure_ascii=False, indent=2)


def _write_champion_summary(root: Path, verdict_data: dict) -> None:
    """Write a champion_summary.json to the backtest root."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "champion_summary.json", "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, ensure_ascii=False, indent=2)


def _setup_three_module_roots(tmp_path, months, verdicts_delta, verdicts_spike,
                               verdicts_negative, use_verdict_json=False):
    """Set up three backtest root directories with predictions and verdicts."""
    delta_root = tmp_path / "delta_backtest"
    spike_root = tmp_path / "spike_backtest"
    negative_root = tmp_path / "negative_backtest"

    for month in months:
        pred_df = _make_prediction_df(month)
        _write_monthly_predictions(delta_root, month, pred_df)
        _write_monthly_predictions(spike_root, month, pred_df)
        _write_monthly_predictions(negative_root, month, pred_df)

    for root, verdicts in [
        (delta_root, verdicts_delta),
        (spike_root, verdicts_spike),
        (negative_root, verdicts_negative),
    ]:
        data = {
            "overall_verdict": "ACCEPTABLE",
            "monthly_verdicts": verdicts,
        }
        if use_verdict_json:
            _write_verdict_json(root, data)
        else:
            _write_champion_summary(root, data)

    return delta_root, spike_root, negative_root


# -- 1. Only verdict.json exists -> can read status correctly -----------------

class TestVerdictJsonFallback:
    def test_load_verdict_from_verdict_json(self, tmp_path):
        """When only verdict.json exists (no champion_summary.json), it is read."""
        root = tmp_path / "backtest"
        verdict_data = {
            "overall_verdict": "STRONG_GO",
            "monthly_verdicts": {"2026-01": "ACCEPTABLE", "2026-02": "CHAMPION"},
        }
        _write_verdict_json(root, verdict_data)

        result = _load_verdict_summary(root)
        assert result is not None
        assert result["overall_verdict"] == "STRONG_GO"
        assert result["monthly_verdicts"]["2026-01"] == "ACCEPTABLE"

    def test_champion_summary_takes_precedence(self, tmp_path):
        """champion_summary.json is preferred over verdict.json."""
        root = tmp_path / "backtest"
        root.mkdir(parents=True, exist_ok=True)

        champion_data = {
            "overall_verdict": "FROM_CHAMPION",
            "monthly_verdicts": {"2026-01": "GO"},
        }
        verdict_data = {
            "overall_verdict": "FROM_VERDICT",
            "monthly_verdicts": {"2026-01": "NO_GO"},
        }
        _write_champion_summary(root, champion_data)
        _write_verdict_json(root, verdict_data)

        result = _load_verdict_summary(root)
        assert result["overall_verdict"] == "FROM_CHAMPION"

    def test_verdict_json_only_no_champion_summary(self, tmp_path):
        """End-to-end: verdict.json-only roots produce correct module statuses."""
        months = ["2026-01"]
        verdicts = {"2026-01": "STRONG_GO"}
        delta_root, spike_root, negative_root = _setup_three_module_roots(
            tmp_path, months, verdicts, verdicts, verdicts, use_verdict_json=True,
        )

        # Verify no champion_summary.json exists.
        assert not (delta_root / "champion_summary.json").exists()

        pack, monthly_manifest, status_sources = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        # All modules should be GO since STRONG_GO maps to GO.
        assert (pack["module_status_delta_supply"] == "GO").all()
        assert (pack["module_status_spike"] == "GO").all()
        assert (pack["module_status_negative"] == "GO").all()

    def test_no_verdict_files_returns_none(self, tmp_path):
        """When neither champion_summary.json nor verdict.json exists, returns None."""
        root = tmp_path / "empty_backtest"
        root.mkdir(parents=True, exist_ok=True)
        result = _load_verdict_summary(root)
        assert result is None


# -- 2. CHAMPION/ACCEPTABLE verdicts map to GO --------------------------------

class TestVerdictToStatusGoMappings:
    @pytest.mark.parametrize("verdict", [
        "GO",
        "CHAMPION",
        "STRONG",
        "ACCEPTABLE",
        "OVERALL_CHAMPION",
        "MONTHLY_STRONG",
        "DELTA_ACCEPTABLE",
        "MODULE_GO",
    ])
    def test_go_like_verdicts_map_to_go(self, verdict):
        assert _normalize_verdict_to_status(verdict) == "GO"

    @pytest.mark.parametrize("verdict", [
        "strong_go",
        "overall_champion",
        "month_acceptable",
    ])
    def test_case_insensitive_go(self, verdict):
        assert _normalize_verdict_to_status(verdict) == "GO"


# -- 3. LOW_VALUE maps to LOW_VALUE -------------------------------------------

class TestVerdictToStatusLowValue:
    @pytest.mark.parametrize("verdict", [
        "LOW_VALUE",
        "OVERALL_LOW_VALUE",
        "MODULE_LOW_VALUE",
    ])
    def test_low_value_mapping(self, verdict):
        assert _normalize_verdict_to_status(verdict) == "LOW_VALUE"


# -- 4. NO_GO maps to NO-GO and risk columns become NaN -----------------------

class TestVerdictToStatusNoGo:
    @pytest.mark.parametrize("verdict", [
        "NOGO",
    ])
    def test_nogo_mapping(self, verdict):
        """Only 'NOGO' (no separator) correctly maps to 'NO-GO'.

        Note: The current implementation has a known issue where verdicts like
        'NO_GO', 'NO-GO', 'OVERALL_NO_GO' match the '_GO' suffix check before
        reaching the NO_GO check, and incorrectly return 'GO'. Only the bare
        string 'NOGO' (which doesn't end with '_GO') correctly maps to 'NO-GO'.
        """
        assert _normalize_verdict_to_status(verdict) == "NO-GO"

    def test_nogo_verdicts_with_separator_correctly_mapped(self):
        """After bug fix: NO_GO/NO-GO correctly map to 'NO-GO'.

        The NO_GO check now fires before the _GO suffix check, so these
        verdicts correctly return 'NO-GO' instead of 'GO'.
        """
        assert _normalize_verdict_to_status("NO_GO") == "NO-GO"
        assert _normalize_verdict_to_status("NO-GO") == "NO-GO"
        assert _normalize_verdict_to_status("OVERALL_NO_GO") == "NO-GO"
        # "MODULE_NOGO" contains "NOGO" but not "NO_GO" substring, and doesn't
        # exactly match "NOGO", so it falls through to UNKNOWN.
        assert _normalize_verdict_to_status("MODULE_NOGO") == "UNKNOWN"

    def test_nogo_risk_columns_become_nan(self, tmp_path):
        """When a module verdict is NOGO, its risk columns should be NaN."""
        months = ["2026-01", "2026-02"]
        # Use "NOGO" (no separator) which correctly maps to "NO-GO".
        verdicts_delta = {"2026-01": "GO", "2026-02": "NOGO"}
        verdicts_spike = {"2026-01": "GO", "2026-02": "GO"}
        verdicts_negative = {"2026-01": "GO", "2026-02": "GO"}

        delta_root, spike_root, negative_root = _setup_three_module_roots(
            tmp_path, months, verdicts_delta, verdicts_spike, verdicts_negative,
        )

        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        month2 = pack[pack["target_month"] == "2026-02"]
        assert (month2["module_status_delta_supply"] == "NO-GO").all()

        delta_risk_cols = [
            "deviation_up_prob", "deviation_down_prob",
            "deviation_large_abs_prob", "deviation_risk_score",
        ]
        for col in delta_risk_cols:
            if col in month2.columns:
                assert month2[col].isna().all(), (
                    f"Expected {col} to be all NaN for NO-GO month 2026-02"
                )

        # Month 1 should still have valid values.
        month1 = pack[pack["target_month"] == "2026-01"]
        assert (month1["module_status_delta_supply"] == "GO").all()
        for col in delta_risk_cols:
            if col in month1.columns:
                assert month1[col].notna().any(), (
                    f"Expected {col} to have non-NaN values for GO month 2026-01"
                )


# -- 5. Not allowed to have all module_status UNKNOWN and export successfully --

class TestAllUnknownStatusBlocked:
    def test_all_unknown_exits(self, tmp_path):
        """If all module statuses are UNKNOWN, the export must fail."""
        months = ["2026-01"]
        # No verdict files at all -> all UNKNOWN.
        delta_root = tmp_path / "delta_backtest"
        spike_root = tmp_path / "spike_backtest"
        negative_root = tmp_path / "negative_backtest"

        for month in months:
            pred_df = _make_prediction_df(month)
            _write_monthly_predictions(delta_root, month, pred_df)
            _write_monthly_predictions(spike_root, month, pred_df)
            _write_monthly_predictions(negative_root, month, pred_df)

        # No verdict.json or champion_summary.json in any root.
        out_dir = tmp_path / "pack_output"

        from scripts.export_risk_feature_pack_multimonth import main

        old_argv = sys.argv
        sys.argv = [
            "export_risk_feature_pack_multimonth.py",
            "--delta-supply-root", str(delta_root),
            "--spike-root", str(spike_root),
            "--negative-root", str(negative_root),
            "--metric-alignment-status", "PASS",
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


# -- 6. online mode does not contain y_true -----------------------------------

class TestOnlineModeNoYtrue:
    def test_online_mode_excludes_y_true(self, tmp_path):
        months = ["2026-01"]
        verdicts = {"2026-01": "GO"}
        delta_root, spike_root, negative_root = _setup_three_module_roots(
            tmp_path, months, verdicts, verdicts, verdicts,
        )

        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        assert "y_true" not in pack.columns
        assert "rt_actual" not in pack.columns

    def test_online_columns_subset_of_expected(self, tmp_path):
        months = ["2026-01"]
        verdicts = {"2026-01": "GO"}
        delta_root, spike_root, negative_root = _setup_three_module_roots(
            tmp_path, months, verdicts, verdicts, verdicts,
        )

        pack, _, _ = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        for col in pack.columns:
            assert col in ONLINE_COLUMNS, f"Unexpected column in online mode: {col}"


# -- 7. status_source appears in manifest -------------------------------------

class TestStatusSourceInManifest:
    def test_status_sources_in_manifest(self, tmp_path):
        """Manifest must contain status_sources mapping for each module."""
        months = ["2026-01"]
        verdicts = {"2026-01": "GO"}
        delta_root, spike_root, negative_root = _setup_three_module_roots(
            tmp_path, months, verdicts, verdicts, verdicts,
        )

        pack, monthly_manifest, status_sources = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        out_dir = tmp_path / "pack_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            out_dir, pack, "online", "PASS", monthly_manifest, status_sources,
        )

        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "status_sources" in manifest
        ss = manifest["status_sources"]
        assert "delta_supply" in ss
        assert "spike" in ss
        assert "negative" in ss

    def test_status_source_monthly_verdicts(self, tmp_path):
        """When monthly verdicts exist, status_source should be 'monthly_verdicts'."""
        months = ["2026-01", "2026-02"]
        verdicts = {"2026-01": "GO", "2026-02": "ACCEPTABLE"}
        delta_root, spike_root, negative_root = _setup_three_module_roots(
            tmp_path, months, verdicts, verdicts, verdicts,
        )

        _, _, status_sources = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        for module in ("delta_supply", "spike", "negative"):
            assert status_sources[module] == "monthly_verdicts"

    def test_status_source_none_when_no_verdict_files(self, tmp_path):
        """When no verdict files exist, status_source should be 'none'."""
        months = ["2026-01"]
        delta_root = tmp_path / "delta_backtest"
        spike_root = tmp_path / "spike_backtest"
        negative_root = tmp_path / "negative_backtest"

        for month in months:
            pred_df = _make_prediction_df(month)
            _write_monthly_predictions(delta_root, month, pred_df)
            _write_monthly_predictions(spike_root, month, pred_df)
            _write_monthly_predictions(negative_root, month, pred_df)

        # No verdict files -> status_sources should all be "none".
        _, _, status_sources = build_risk_feature_pack_multimonth(
            delta_supply_root=delta_root,
            spike_root=spike_root,
            negative_root=negative_root,
            mode="online",
            metric_alignment_status="PASS",
        )

        for module in ("delta_supply", "spike", "negative"):
            assert status_sources[module] == "none"


# -- Additional: INSUFFICIENT and unknown verdict mappings ----------------------

class TestVerdictToStatusEdgeCases:
    @pytest.mark.parametrize("verdict", [
        "INSUFFICIENT_DATA",
        "INSUFFICIENT_MONTHS",
        "INSUFFICIENT",
    ])
    def test_insufficient_mapping(self, verdict):
        assert _normalize_verdict_to_status(verdict) == "INSUFFICIENT"

    @pytest.mark.parametrize("verdict", [
        None,
        "SOMETHING_ELSE",
        "MAYBE",
        "",
    ])
    def test_unknown_mapping(self, verdict):
        assert _normalize_verdict_to_status(verdict) == "UNKNOWN"
