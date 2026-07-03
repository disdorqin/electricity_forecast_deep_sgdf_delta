#!/usr/bin/env python
"""Risk Module Selection Board.

Reads multi-month backtest results for each risk module and produces
KEEP / KEEP_AS_AUX / DROP / NEEDS_MORE_DATA decisions, along with
next-phase recommendations.

Decision rules:
  KEEP            -- stable GO or ACCEPTABLE across months
  KEEP_AS_AUX    -- LOW_VALUE but useful as top-k auxiliary signal
  DROP            -- NO-GO across months (no usable signal)
  NEEDS_MORE_DATA -- insufficient events or months to make a call

Produces:
  <out-dir>/
    risk_module_selection.json    Machine-readable selection decisions
    risk_module_selection.csv     Human-readable selection table

Usage:
    python scripts/select_risk_modules.py \
      --delta-supply-backtest reports/local/risk_modules/delta_supply_risk_backtest_2026_01_05 \
      --negative-backtest reports/local/risk_modules/negative_risk_backtest_2026_01_05 \
      --spike-backtest reports/local/risk_modules/spike_risk_backtest_2026_01_05 \
      --out-dir reports/local/risk_modules/risk_module_selection
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("select_risk_modules")

# -- Constants ----------------------------------------------------------------

VALID_DECISIONS = {"KEEP", "KEEP_AS_AUX", "DROP", "NEEDS_MORE_DATA"}

# Verdicts that count as "good" for KEEP.
GOOD_VERDICTS = {"GO", "ACCEPTABLE", "STRONG",
                 "DELTA-RISK-STRONG", "DELTA-RISK-ACCEPTABLE",
                 "SPIKE-CHAMPION", "SPIKE-ACCEPTABLE",
                 "NEGATIVE-CHAMPION", "NEGATIVE-ACCEPTABLE"}
LOW_VALUE_VERDICTS = {"LOW-VALUE",
                      "DELTA-RISK-LOW-VALUE",
                      "SPIKE-LOW-VALUE",
                      "NEGATIVE-LOW-VALUE"}
NOGO_VERDICTS = {"NO-GO", "NOGO",
                 "DELTA-RISK-NO-GO",
                 "SPIKE-NO-GO",
                 "NEGATIVE-NO-GO"}

# Minimum number of successful months to avoid NEEDS_MORE_DATA.
MIN_SUCCESSFUL_MONTHS = 2
# Minimum number of total months to make a decision.
MIN_TOTAL_MONTHS = 2


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Risk Module Selection Board: KEEP / AUX / DROP / NEEDS_MORE_DATA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--delta-supply-backtest", type=str, required=True,
        help="Root directory of DeltaSupply backtest",
    )
    parser.add_argument(
        "--negative-backtest", type=str, required=True,
        help="Root directory of NegativeRisk backtest",
    )
    parser.add_argument(
        "--spike-backtest", type=str, required=True,
        help="Root directory of SpikeRisk backtest",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for selection results",
    )
    return parser.parse_args()


# -- Helpers ------------------------------------------------------------------

def _resolve_path(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_champion_summary(root: Path) -> Optional[dict]:
    """Load champion_summary.json from a backtest root."""
    summary_path = root / "champion_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("champion_summary.json not found in %s", root)
    return None


def _load_monthly_metrics(root: Path) -> Optional[pd.DataFrame]:
    """Load monthly_metrics.csv from a backtest root."""
    metrics_path = root / "monthly_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path, encoding="utf-8-sig")
    logger.warning("monthly_metrics.csv not found in %s", root)
    return None


def _normalize_verdict(v: str) -> str:
    """Normalize a verdict string to a canonical form."""
    if v is None:
        return "UNKNOWN"
    return v.strip().upper().replace(" ", "-").replace("_", "-")


def _compute_decision(
    module_name: str,
    champion_summary: Optional[dict],
    monthly_metrics: Optional[pd.DataFrame],
) -> dict:
    """Compute the selection decision for a single module.

    Returns a dict with:
      module_name, decision, reason, monthly_verdicts, key_metrics,
      next_phase_recommendation.
    """
    result = {
        "module_name": module_name,
        "decision": "NEEDS_MORE_DATA",
        "reason": "",
        "monthly_verdicts": {},
        "key_metrics": {},
        "next_phase_recommendation": "",
    }

    if champion_summary is None:
        result["reason"] = "No champion_summary.json found; cannot evaluate module."
        result["next_phase_recommendation"] = (
            f"Re-run backtest for {module_name} with more months of data."
        )
        return result

    overall_verdict = _normalize_verdict(champion_summary.get("overall_verdict", ""))
    monthly_verdicts_raw = champion_summary.get("monthly_verdicts", {})
    n_months = champion_summary.get("n_months", 0)
    n_successful = champion_summary.get("n_successful", 0)

    result["monthly_verdicts"] = {
        m: _normalize_verdict(v) for m, v in monthly_verdicts_raw.items()
    }

    # Count verdict categories.
    verdict_counts = Counter(result["monthly_verdicts"].values())
    n_good = sum(verdict_counts.get(v, 0) for v in GOOD_VERDICTS)
    n_low_value = sum(verdict_counts.get(v, 0) for v in LOW_VALUE_VERDICTS)
    n_nogo = sum(verdict_counts.get(v, 0) for v in NOGO_VERDICTS)

    # Extract key metrics from monthly_metrics if available.
    if monthly_metrics is not None and not monthly_metrics.empty:
        # Average AUC across months and targets.
        if "roc_auc" in monthly_metrics.columns:
            mean_auc = monthly_metrics["roc_auc"].mean()
            result["key_metrics"]["mean_roc_auc"] = round(float(mean_auc), 4)
        if "f1" in monthly_metrics.columns:
            mean_f1 = monthly_metrics["f1"].mean()
            result["key_metrics"]["mean_f1"] = round(float(mean_f1), 4)
        result["key_metrics"]["n_months_evaluated"] = int(
            monthly_metrics["month"].nunique() if "month" in monthly_metrics.columns else 0
        )

    result["key_metrics"]["overall_verdict"] = overall_verdict
    result["key_metrics"]["n_months_total"] = n_months
    result["key_metrics"]["n_months_successful"] = n_successful
    result["key_metrics"]["n_good_months"] = n_good
    result["key_metrics"]["n_low_value_months"] = n_low_value
    result["key_metrics"]["n_nogo_months"] = n_nogo

    # Decision logic.
    if n_successful < MIN_SUCCESSFUL_MONTHS and n_months < MIN_TOTAL_MONTHS:
        result["decision"] = "NEEDS_MORE_DATA"
        result["reason"] = (
            f"Only {n_successful}/{n_months} months completed successfully. "
            f"Need at least {MIN_SUCCESSFUL_MONTHS} successful months out of "
            f"{MIN_TOTAL_MONTHS} total to make a decision."
        )
        result["next_phase_recommendation"] = (
            f"Collect more data and re-run backtest for {module_name}."
        )
        return result

    # Check if all months are NO-GO.
    if n_nogo > 0 and n_good == 0 and n_low_value == 0:
        result["decision"] = "DROP"
        result["reason"] = (
            f"All {n_nogo} successful months returned NO-GO verdict. "
            f"Module provides no usable signal."
        )
        result["next_phase_recommendation"] = (
            f"Drop {module_name} from the risk module pipeline. "
            f"Revisit if new features or data sources become available."
        )
        return result

    # Check if overall verdict is NO-GO.
    if overall_verdict in NOGO_VERDICTS:
        result["decision"] = "DROP"
        result["reason"] = (
            f"Overall verdict is {overall_verdict}. "
            f"Module does not provide sufficient value for production use."
        )
        result["next_phase_recommendation"] = (
            f"Drop {module_name} from the risk module pipeline. "
            f"Consider as research-only signal."
        )
        return result

    # Check if module is consistently good.
    if n_good >= MIN_SUCCESSFUL_MONTHS:
        result["decision"] = "KEEP"
        result["reason"] = (
            f"{n_good} out of {n_successful} successful months returned GO/ACCEPTABLE. "
            f"Module provides stable risk signal."
        )
        result["next_phase_recommendation"] = (
            f"Promote {module_name} to champion/auxiliary in the next phase. "
            f"Continue monitoring monthly performance."
        )
        return result

    # Check if module is LOW_VALUE but has some useful signal.
    if n_low_value > 0:
        mean_auc = result["key_metrics"].get("mean_roc_auc", 0)
        if mean_auc >= 0.85:
            result["decision"] = "KEEP_AS_AUX"
            result["reason"] = (
                f"Module is LOW_VALUE overall but has strong discrimination "
                f"(mean AUC={mean_auc:.3f}). Useful as top-k auxiliary signal."
            )
            result["next_phase_recommendation"] = (
                f"Keep {module_name} as auxiliary feature for top-k risk hours. "
                f"Do not use as primary correction signal."
            )
            return result
        else:
            result["decision"] = "KEEP_AS_AUX"
            result["reason"] = (
                f"Module is LOW_VALUE for correction but provides some ranking signal. "
                f"Useful as auxiliary input for downstream fusion."
            )
            result["next_phase_recommendation"] = (
                f"Keep {module_name} as auxiliary feature. "
                f"Needs more spike/rare events for full validation."
            )
            return result

    # Default: need more data.
    result["decision"] = "NEEDS_MORE_DATA"
    result["reason"] = (
        f"Verdict distribution is ambiguous: {dict(verdict_counts)}. "
        f"Cannot make a clear KEEP/DROP decision."
    )
    result["next_phase_recommendation"] = (
        f"Collect more months of data for {module_name} and re-evaluate."
    )
    return result


# -- Next-phase recommendations -----------------------------------------------

def _build_next_phase_recommendations(decisions: list[dict]) -> dict:
    """Build consolidated next-phase recommendations from all module decisions.

    Returns a dict with per-module recommendations.
    """
    recommendations = {}

    for d in decisions:
        name = d["module_name"]
        decision = d["decision"]

        if name == "NegativeRisk":
            if decision == "KEEP":
                recommendations["negative_risk"] = "champion"
            elif decision == "KEEP_AS_AUX":
                recommendations["negative_risk"] = "aux"
            elif decision == "DROP":
                recommendations["negative_risk"] = "drop"
            else:
                recommendations["negative_risk"] = "needs_more_data"

        elif name == "SpikeRisk":
            if decision == "KEEP":
                recommendations["spike_risk"] = "champion"
            elif decision == "KEEP_AS_AUX":
                recommendations["spike_risk"] = "aux"
            elif decision == "DROP":
                recommendations["spike_risk"] = "drop"
            else:
                recommendations["spike_risk"] = "needs_more_data"

        elif name == "DeltaSupplyRisk":
            if decision == "KEEP":
                recommendations["delta_supply_risk"] = "champion"
            elif decision == "KEEP_AS_AUX":
                recommendations["delta_supply_risk"] = "aux"
            elif decision == "DROP":
                recommendations["delta_supply_risk"] = "drop"
            else:
                recommendations["delta_supply_risk"] = "needs_more_data"

    return recommendations


# -- Core logic ---------------------------------------------------------------

def select_risk_modules(
    delta_supply_root: Path,
    negative_root: Path,
    spike_root: Path,
) -> list[dict]:
    """Run the selection board for all three risk modules.

    Returns a list of decision dicts, one per module.
    """
    decisions = []

    # DeltaSupply Risk.
    delta_summary = _load_champion_summary(delta_supply_root)
    delta_metrics = _load_monthly_metrics(delta_supply_root)
    delta_decision = _compute_decision("DeltaSupplyRisk", delta_summary, delta_metrics)
    decisions.append(delta_decision)

    # Spike Risk.
    spike_summary = _load_champion_summary(spike_root)
    spike_metrics = _load_monthly_metrics(spike_root)
    spike_decision = _compute_decision("SpikeRisk", spike_summary, spike_metrics)
    decisions.append(spike_decision)

    # Negative Risk.
    negative_summary = _load_champion_summary(negative_root)
    negative_metrics = _load_monthly_metrics(negative_root)
    negative_decision = _compute_decision("NegativeRisk", negative_summary, negative_metrics)
    decisions.append(negative_decision)

    return decisions


def write_outputs(
    out_dir: Path,
    decisions: list[dict],
) -> None:
    """Write risk_module_selection.json and risk_module_selection.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build next-phase recommendations.
    recommendations = _build_next_phase_recommendations(decisions)

    # JSON output.
    selection_json = {
        "timestamp": datetime.now().isoformat(),
        "modules": decisions,
        "next_phase_recommendations": recommendations,
        "decision_summary": {
            d["module_name"]: d["decision"] for d in decisions
        },
    }

    json_path = out_dir / "risk_module_selection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selection_json, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Selection JSON -> %s", json_path)

    # CSV output.
    csv_rows = []
    for d in decisions:
        csv_rows.append({
            "module_name": d["module_name"],
            "decision": d["decision"],
            "reason": d["reason"],
            "overall_verdict": d["key_metrics"].get("overall_verdict", ""),
            "mean_roc_auc": d["key_metrics"].get("mean_roc_auc", ""),
            "mean_f1": d["key_metrics"].get("mean_f1", ""),
            "n_months_total": d["key_metrics"].get("n_months_total", ""),
            "n_months_successful": d["key_metrics"].get("n_months_successful", ""),
            "n_good_months": d["key_metrics"].get("n_good_months", ""),
            "n_low_value_months": d["key_metrics"].get("n_low_value_months", ""),
            "n_nogo_months": d["key_metrics"].get("n_nogo_months", ""),
            "next_phase_recommendation": d["next_phase_recommendation"],
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = out_dir / "risk_module_selection.csv"
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Selection CSV -> %s", csv_path)


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    delta_root = _resolve_path(args.delta_supply_backtest)
    negative_root = _resolve_path(args.negative_backtest)
    spike_root = _resolve_path(args.spike_backtest)
    out_dir = _resolve_path(args.out_dir)

    decisions = select_risk_modules(delta_root, negative_root, spike_root)

    # Log summary.
    for d in decisions:
        logger.info(
            "  %-20s -> %-18s (%s)",
            d["module_name"], d["decision"], d["reason"][:80],
        )

    write_outputs(out_dir, decisions)
    logger.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
