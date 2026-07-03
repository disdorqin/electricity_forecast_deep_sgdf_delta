"""Tests for select_risk_modules script.

Covers:
  1. Selection produces valid decisions (KEEP/KEEP_AS_AUX/DROP/NEEDS_MORE_DATA)
  2. GO module -> KEEP
  3. NO-GO module -> DROP
  4. LOW_VALUE module -> KEEP_AS_AUX
  5. Insufficient events -> NEEDS_MORE_DATA
  6. Next-phase recommendations present
  7. Output files created
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.select_risk_modules import (  # noqa: E402
    VALID_DECISIONS,
    _compute_decision,
    _build_next_phase_recommendations,
    main,
    select_risk_modules,
    write_outputs,
)


# -- Fixtures -----------------------------------------------------------------

def _write_champion_summary(root: Path, verdicts: dict, overall: str) -> None:
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


def _write_monthly_metrics(root: Path, months: list[str], auc: float = 0.85, f1: float = 0.5) -> None:
    """Write a monthly_metrics.csv to the backtest root."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for month in months:
        for target in ["upward_deviation_label", "downward_deviation_label"]:
            rows.append({
                "month": month,
                "target": target,
                "f1": f1,
                "roc_auc": auc,
                "precision": f1,
                "recall": f1,
            })
    pd.DataFrame(rows).to_csv(root / "monthly_metrics.csv", index=False)


@pytest.fixture()
def go_module_root(tmp_path):
    """Backtest root for a consistently GO module."""
    root = tmp_path / "go_module"
    verdicts = {"2026-01": "GO", "2026-02": "GO", "2026-03": "ACCEPTABLE"}
    _write_champion_summary(root, verdicts, "ACCEPTABLE")
    _write_monthly_metrics(root, list(verdicts.keys()), auc=0.90, f1=0.6)
    return root


@pytest.fixture()
def nogo_module_root(tmp_path):
    """Backtest root for a consistently NO-GO module."""
    root = tmp_path / "nogo_module"
    verdicts = {"2026-01": "NO-GO", "2026-02": "NO-GO", "2026-03": "NO-GO"}
    _write_champion_summary(root, verdicts, "NO-GO")
    _write_monthly_metrics(root, list(verdicts.keys()), auc=0.55, f1=0.1)
    return root


@pytest.fixture()
def low_value_module_root(tmp_path):
    """Backtest root for a LOW_VALUE module with high AUC."""
    root = tmp_path / "low_value_module"
    verdicts = {"2026-01": "LOW_VALUE", "2026-02": "LOW_VALUE", "2026-03": "LOW_VALUE"}
    _write_champion_summary(root, verdicts, "LOW_VALUE")
    _write_monthly_metrics(root, list(verdicts.keys()), auc=0.92, f1=0.3)
    return root


@pytest.fixture()
def insufficient_module_root(tmp_path):
    """Backtest root with insufficient data."""
    root = tmp_path / "insufficient_module"
    verdicts = {"2026-01": "GO"}
    summary = {
        "overall_verdict": "UNKNOWN",
        "mean_monthly_improvement_pp": 0.0,
        "n_months": 1,
        "n_successful": 1,
        "monthly_verdicts": verdicts,
    }
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "champion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return root


# -- 1. Selection produces valid decisions ------------------------------------

class TestValidDecisions:
    def test_all_decisions_are_valid(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        for d in decisions:
            assert d["decision"] in VALID_DECISIONS, (
                f"Invalid decision '{d['decision']}' for module {d['module_name']}"
            )

    def test_all_modules_have_names(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        names = {d["module_name"] for d in decisions}
        assert "DeltaSupplyRisk" in names
        assert "SpikeRisk" in names
        assert "NegativeRisk" in names


# -- 2. GO module -> KEEP -----------------------------------------------------

class TestGoModuleBecomesKeep:
    def test_go_module_is_keep(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        go_decision = next(d for d in decisions if d["module_name"] == "DeltaSupplyRisk")
        assert go_decision["decision"] == "KEEP"

    def test_go_module_reason_mentions_stable(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        go_decision = next(d for d in decisions if d["module_name"] == "DeltaSupplyRisk")
        assert "GO" in go_decision["reason"] or "ACCEPTABLE" in go_decision["reason"]


# -- 3. NO-GO module -> DROP --------------------------------------------------

class TestNogoModuleBecomesDrop:
    def test_nogo_module_is_drop(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        nogo_decision = next(d for d in decisions if d["module_name"] == "NegativeRisk")
        assert nogo_decision["decision"] == "DROP"

    def test_nogo_module_reason_mentions_nogo(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        nogo_decision = next(d for d in decisions if d["module_name"] == "NegativeRisk")
        assert "NO-GO" in nogo_decision["reason"] or "NO_GO" in nogo_decision["reason"]


# -- 4. LOW_VALUE module -> KEEP_AS_AUX ---------------------------------------

class TestLowValueModuleBecomesAux:
    def test_low_value_module_is_aux(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        aux_decision = next(d for d in decisions if d["module_name"] == "SpikeRisk")
        assert aux_decision["decision"] == "KEEP_AS_AUX"

    def test_low_value_module_has_high_auc(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        aux_decision = next(d for d in decisions if d["module_name"] == "SpikeRisk")
        assert aux_decision["key_metrics"].get("mean_roc_auc", 0) >= 0.85


# -- 5. Insufficient events -> NEEDS_MORE_DATA --------------------------------

class TestInsufficientDataNeedsMore:
    def test_insufficient_module_needs_more_data(
        self, insufficient_module_root, nogo_module_root, low_value_module_root,
    ):
        decisions = select_risk_modules(
            delta_supply_root=insufficient_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        insufficient_decision = next(d for d in decisions if d["module_name"] == "DeltaSupplyRisk")
        assert insufficient_decision["decision"] == "NEEDS_MORE_DATA"

    def test_no_champion_summary_needs_more_data(
        self, tmp_path, nogo_module_root, low_value_module_root,
    ):
        """Module with no champion_summary.json should get NEEDS_MORE_DATA."""
        empty_root = tmp_path / "empty_root"
        empty_root.mkdir(parents=True, exist_ok=True)

        decisions = select_risk_modules(
            delta_supply_root=empty_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        empty_decision = next(d for d in decisions if d["module_name"] == "DeltaSupplyRisk")
        assert empty_decision["decision"] == "NEEDS_MORE_DATA"


# -- 6. Next-phase recommendations present ------------------------------------

class TestNextPhaseRecommendations:
    def test_recommendations_present(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        recommendations = _build_next_phase_recommendations(decisions)
        assert "negative_risk" in recommendations
        assert "spike_risk" in recommendations
        assert "delta_supply_risk" in recommendations

    def test_recommendations_match_decisions(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        recommendations = _build_next_phase_recommendations(decisions)

        # DeltaSupplyRisk is GO -> champion.
        assert recommendations["delta_supply_risk"] == "champion"
        # NegativeRisk is NO-GO -> drop.
        assert recommendations["negative_risk"] == "drop"
        # SpikeRisk is LOW_VALUE with high AUC -> aux.
        assert recommendations["spike_risk"] == "aux"

    def test_recommendation_values_are_valid(self, go_module_root, nogo_module_root, low_value_module_root):
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        recommendations = _build_next_phase_recommendations(decisions)
        valid_rec_values = {"champion", "aux", "drop", "needs_more_data"}
        for key, val in recommendations.items():
            assert val in valid_rec_values, (
                f"Invalid recommendation '{val}' for {key}"
            )


# -- 7. Output files created --------------------------------------------------

class TestOutputFilesCreated:
    def test_json_and_csv_created(self, tmp_path, go_module_root, nogo_module_root, low_value_module_root):
        out_dir = tmp_path / "selection_output"
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        write_outputs(out_dir, decisions)

        assert (out_dir / "risk_module_selection.json").exists()
        assert (out_dir / "risk_module_selection.csv").exists()

    def test_json_is_parseable(self, tmp_path, go_module_root, nogo_module_root, low_value_module_root):
        out_dir = tmp_path / "selection_output"
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        write_outputs(out_dir, decisions)

        with open(out_dir / "risk_module_selection.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "modules" in data
        assert "next_phase_recommendations" in data
        assert "decision_summary" in data
        assert len(data["modules"]) == 3

    def test_csv_is_parseable(self, tmp_path, go_module_root, nogo_module_root, low_value_module_root):
        out_dir = tmp_path / "selection_output"
        decisions = select_risk_modules(
            delta_supply_root=go_module_root,
            negative_root=nogo_module_root,
            spike_root=low_value_module_root,
        )
        write_outputs(out_dir, decisions)

        df = pd.read_csv(out_dir / "risk_module_selection.csv")
        assert len(df) == 3
        assert "module_name" in df.columns
        assert "decision" in df.columns
        assert "next_phase_recommendation" in df.columns

    def test_full_cli(self, tmp_path, go_module_root, nogo_module_root, low_value_module_root):
        """Full CLI invocation produces expected files."""
        out_dir = tmp_path / "cli_output"

        old_argv = sys.argv
        sys.argv = [
            "select_risk_modules.py",
            "--delta-supply-backtest", str(go_module_root),
            "--negative-backtest", str(nogo_module_root),
            "--spike-backtest", str(low_value_module_root),
            "--out-dir", str(out_dir),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        assert (out_dir / "risk_module_selection.json").exists()
        assert (out_dir / "risk_module_selection.csv").exists()

        with open(out_dir / "risk_module_selection.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["decision_summary"]["DeltaSupplyRisk"] == "KEEP"
        assert data["decision_summary"]["NegativeRisk"] == "DROP"
        assert data["decision_summary"]["SpikeRisk"] == "KEEP_AS_AUX"
