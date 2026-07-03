#!/usr/bin/env python
"""NegativeRisk Champion Recalibration -- base-rate aware metrics.

The original NegativeRisk backtest verdicts top-k capture using raw recall,
which is unfairly penalised when negative events have a high base rate
(~15-30 %).  When the positive rate is high, even a perfect ranker cannot
achieve 70 % recall at top-10 % because the theoretical ceiling is
``min(1.0, 0.10 / positive_rate)``.

This script re-evaluates the NegativeRisk module with *normalised* metrics:

1. **Normalised top-k recall** -- actual recall divided by the theoretical
   maximum recall achievable at that budget given the base rate.
2. **Alert-budget metrics** -- precision, recall, lift and F1 when the
   operator acts on the top *budget* % of hours ranked by negative_prob.

New champion criteria replace raw top-k capture with these base-rate aware
numbers so that a module with strong AUC and good alert-budget performance
is not downgraded merely because the base rate is high.

Usage:
    python scripts/recalibrate_negative_risk_selection.py \\
        --backtest-root reports/local/risk_modules/negative_risk_backtest_2026_01_05 \\
        --out-dir reports/local/risk_modules/negative_recalibration
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

TOP_K_PCTS = [5, 10, 20]
ALERT_BUDGET_PCTS = [10, 20, 30]
DIRECTION = "negative"  # primary direction for recalibration
MONTHS_GLOB = "predictions_2026_*.csv"


# ── Pure helpers (importable / testable) ─────────────────────────────────────

def max_possible_recall_at_topk(topk_pct: float, positive_rate: float) -> float:
    """Theoretical ceiling for recall when selecting the top *topk_pct* %.

    If the positive rate is 20 % and we select the top 10 % of hours, at
    most ``min(1.0, 0.10 / 0.20) = 0.50`` of all positives can be captured
    because there are only 10 % slots but 20 % of hours are positive.
    """
    if positive_rate <= 0:
        return 0.0
    return min(1.0, (topk_pct / 100.0) / positive_rate)


def normalised_recall(actual_recall: float, max_recall: float) -> float:
    """Recall divided by theoretical ceiling; 0.0 when ceiling is 0."""
    if max_recall <= 0:
        return 0.0
    return actual_recall / max_recall


def compute_alert_budget_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    budget_pct: float,
) -> dict:
    """Precision / recall / lift / F1 when alerting the top *budget_pct* %.

    Parameters
    ----------
    y_true : binary labels (0/1).
    y_prob : predicted probabilities (higher = more likely positive).
    budget_pct : percentage of hours to alert (e.g. 10 means top 10 %).

    Returns
    -------
    dict with keys: budget_pct, n_total, n_alerts, n_positive,
    true_positives, precision, recall, lift, f1, base_rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    n_total = len(y_true)
    n_positive = int((y_true >= 1).sum())
    base_rate = n_positive / n_total if n_total > 0 else 0.0

    n_alerts = max(1, int(n_total * budget_pct / 100.0))
    # Sort descending by probability; take top n_alerts
    top_idx = np.argsort(y_prob)[-n_alerts:]
    top_actual = y_true[top_idx]
    true_positives = int((top_actual >= 1).sum())

    precision = true_positives / n_alerts if n_alerts > 0 else 0.0
    recall = true_positives / n_positive if n_positive > 0 else 0.0
    lift = precision / base_rate if base_rate > 0 else 0.0
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "budget_pct": budget_pct,
        "n_total": n_total,
        "n_alerts": n_alerts,
        "n_positive": n_positive,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "f1": f1,
        "base_rate": base_rate,
    }


def compute_topk_capture_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    topk_pct: float,
) -> dict:
    """Recall (capture rate) when selecting top *topk_pct* % by probability."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    n = len(y_true)
    n_positive = int((y_true >= 1).sum())
    k = max(1, int(n * topk_pct / 100.0))
    top_idx = np.argsort(y_prob)[-k:]
    tp_captured = int((y_true[top_idx] >= 1).sum())
    recall = tp_captured / n_positive if n_positive > 0 else 0.0
    precision = tp_captured / k if k > 0 else 0.0

    return {
        "topk_pct": topk_pct,
        "k": k,
        "n_positive": n_positive,
        "tp_captured": tp_captured,
        "recall": recall,
        "precision": precision,
    }


def determine_recalibrated_verdict(
    mean_auc: float,
    mean_f1: float,
    mean_recall_at_20pct_alert: float,
    n_sufficient_months: int,
) -> str:
    """Apply the new base-rate aware champion criteria.

    NEGATIVE_CHAMPION   : mean_auc >= 0.90, mean_f1 >= 0.70,
                          mean_recall_at_20pct_alert >= 0.65,
                          >= 4 sufficient months
    NEGATIVE_ACCEPTABLE : mean_auc >= 0.85, mean_f1 >= 0.60
    NEGATIVE_AUX        : mean_auc >= 0.80
    NEGATIVE_NO_GO      : otherwise
    """
    if (mean_auc >= 0.90 and mean_f1 >= 0.70
            and mean_recall_at_20pct_alert >= 0.65
            and n_sufficient_months >= 4):
        return "NEGATIVE_CHAMPION"
    if mean_auc >= 0.85 and mean_f1 >= 0.60:
        return "NEGATIVE_ACCEPTABLE"
    if mean_auc >= 0.80:
        return "NEGATIVE_AUX"
    return "NEGATIVE_NO_GO"


# ── Main pipeline ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Recalibrate NegativeRisk selection verdict with base-rate aware metrics"
    )
    p.add_argument(
        "--backtest-root", required=True,
        help="Directory containing monthly_metrics.csv and predictions_*.csv",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory for recalibration artefacts",
    )
    return p.parse_args()


def load_monthly_metrics(backtest_root: Path) -> pd.DataFrame:
    """Load monthly_metrics.csv and filter to the primary direction with ok status."""
    path = backtest_root / "monthly_metrics.csv"
    df = pd.read_csv(path)
    ok = df[(df["direction"] == DIRECTION) & (df["status"] == "ok")].copy()
    logger.info("Loaded monthly_metrics: %d ok rows for direction=%s", len(ok), DIRECTION)
    return ok


def load_predictions(backtest_root: Path) -> dict[str, pd.DataFrame]:
    """Load all prediction CSVs keyed by month string (YYYY-MM)."""
    from glob import glob as file_glob

    pred_files = sorted(file_glob(str(backtest_root / "predictions_2026_*.csv")))
    predictions = {}
    for fpath in pred_files:
        fname = Path(fpath).name  # e.g. predictions_2026_01.csv
        month_str = fname.replace("predictions_", "").replace(".csv", "")  # 2026_01
        month_dash = month_str.replace("_", "-")  # 2026-01
        df = pd.read_csv(fpath)
        predictions[month_dash] = df
        logger.info("  Loaded %s: %d rows", fname, len(df))
    return predictions


def build_recalibration_table(
    monthly_metrics: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build per-month recalibration metrics.

    Returns a DataFrame with one row per month containing:
    - month, n_total, n_positive, positive_rate
    - For each topk_pct in TOP_K_PCTS: recall_topk, max_recall_topk, norm_recall_topk
    - For each budget_pct in ALERT_BUDGET_PCTS: precision, recall, lift, f1
    - roc_auc, f1 (from monthly_metrics)
    """
    rows = []
    for _, mrow in monthly_metrics.iterrows():
        month = mrow["month"]
        roc_auc = float(mrow["roc_auc"])
        f1_cls = float(mrow["f1"])
        n_positive = int(mrow["n_positive"])

        if month not in predictions:
            logger.warning("No prediction file for month %s -- skipping", month)
            continue

        pred_df = predictions[month]
        n_total = len(pred_df)
        positive_rate = n_positive / n_total if n_total > 0 else 0.0

        y_true = pred_df["negative_label"].values
        y_prob = pred_df["negative_prob"].values

        row = {
            "month": month,
            "n_total": n_total,
            "n_positive": n_positive,
            "positive_rate": positive_rate,
            "roc_auc": roc_auc,
            "f1_classification": f1_cls,
        }

        # ── Normalised top-k recall ──────────────────────────────────────
        for topk_pct in TOP_K_PCTS:
            cap = compute_topk_capture_from_predictions(y_true, y_prob, topk_pct)
            max_rec = max_possible_recall_at_topk(topk_pct, positive_rate)
            norm_rec = normalised_recall(cap["recall"], max_rec)
            row[f"recall_top{topk_pct}"] = cap["recall"]
            row[f"max_recall_top{topk_pct}"] = max_rec
            row[f"norm_recall_top{topk_pct}"] = norm_rec

        # ── Alert budget metrics ─────────────────────────────────────────
        for budget_pct in ALERT_BUDGET_PCTS:
            bm = compute_alert_budget_metrics(y_true, y_prob, budget_pct)
            row[f"precision_alert{budget_pct}"] = bm["precision"]
            row[f"recall_alert{budget_pct}"] = bm["recall"]
            row[f"lift_alert{budget_pct}"] = bm["lift"]
            row[f"f1_alert{budget_pct}"] = bm["f1"]

        rows.append(row)

    return pd.DataFrame(rows)


def compute_summary(recal_df: pd.DataFrame) -> dict:
    """Aggregate per-month metrics into summary statistics."""
    mean_auc = float(recal_df["roc_auc"].mean())
    mean_f1 = float(recal_df["f1_classification"].mean())
    mean_recall_20 = float(recal_df["recall_alert20"].mean()) if "recall_alert20" in recal_df else 0.0
    n_sufficient = len(recal_df)

    verdict = determine_recalibrated_verdict(mean_auc, mean_f1, mean_recall_20, n_sufficient)

    summary = {
        "mean_auc": mean_auc,
        "mean_f1": mean_f1,
        "mean_recall_at_20pct_alert": mean_recall_20,
        "n_sufficient_months": n_sufficient,
        "verdict": verdict,
    }

    # Per topk_pct averages
    for topk_pct in TOP_K_PCTS:
        col_recall = f"recall_top{topk_pct}"
        col_max = f"max_recall_top{topk_pct}"
        col_norm = f"norm_recall_top{topk_pct}"
        if col_recall in recal_df:
            summary[f"mean_recall_top{topk_pct}"] = float(recal_df[col_recall].mean())
            summary[f"mean_max_recall_top{topk_pct}"] = float(recal_df[col_max].mean())
            summary[f"mean_norm_recall_top{topk_pct}"] = float(recal_df[col_norm].mean())

    # Per alert-budget averages
    for budget_pct in ALERT_BUDGET_PCTS:
        for metric in ["precision", "recall", "lift", "f1"]:
            col = f"{metric}_alert{budget_pct}"
            if col in recal_df:
                summary[f"mean_{metric}_alert{budget_pct}"] = float(recal_df[col].mean())

    return summary


def write_json_summary(summary: dict, out_dir: Path) -> None:
    """Write negative_recalibration_summary.json."""
    out = {
        **summary,
        "created_at": datetime.now().isoformat(),
        "criteria": {
            "NEGATIVE_CHAMPION": (
                "mean_auc >= 0.90, mean_f1 >= 0.70, "
                "mean_recall_at_20pct_alert >= 0.65, >= 4 sufficient months"
            ),
            "NEGATIVE_ACCEPTABLE": "mean_auc >= 0.85, mean_f1 >= 0.60",
            "NEGATIVE_AUX": "mean_auc >= 0.80",
            "NEGATIVE_NO_GO": "otherwise",
        },
    }
    path = out_dir / "negative_recalibration_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", path)


def write_monthly_csv(recal_df: pd.DataFrame, out_dir: Path) -> None:
    """Write negative_recalibration_monthly.csv."""
    path = out_dir / "negative_recalibration_monthly.csv"
    recal_df.to_csv(path, index=False, float_format="%.6f")
    logger.info("Wrote %s", path)


def write_markdown_report(summary: dict, recal_df: pd.DataFrame, out_dir: Path) -> None:
    """Write negative_recalibration_report.md."""
    lines = [
        "# NegativeRisk Recalibration Report",
        "",
        f"**New Verdict: {summary['verdict']}**",
        "",
        f"*Generated: {datetime.now().isoformat()}*",
        "",
        "## Purpose",
        "",
        "The original NegativeRisk backtest assigned verdict **NEGATIVE_LOW_VALUE**",
        "because mean top-10% capture (0.365) fell below the 0.70 champion",
        "threshold.  However, negative price events have a high base rate",
        "(~15-30 %), which caps the theoretical maximum recall achievable at",
        "any fixed top-k budget.  This recalibration applies **base-rate aware",
        "metrics** to evaluate the module fairly.",
        "",
        "## Base-Rate Analysis",
        "",
        "| Month | N Total | N Positive | Positive Rate |",
        "|-------|---------|------------|---------------|",
    ]
    for _, r in recal_df.iterrows():
        lines.append(
            f"| {r['month']} | {int(r['n_total'])} | {int(r['n_positive'])} "
            f"| {r['positive_rate']:.4f} |"
        )
    lines.append("")

    lines += [
        "## Normalised Top-k Recall",
        "",
        "Normalised recall = actual recall / theoretical maximum recall at that budget.",
        "A value of 1.0 means the ranker captures every possible positive within the budget.",
        "",
    ]
    header_parts = ["| Month"]
    sep_parts = ["|-------"]
    for topk_pct in TOP_K_PCTS:
        header_parts.append(f"Recall@top{topk_pct}")
        header_parts.append(f"Max@top{topk_pct}")
        header_parts.append(f"Norm@top{topk_pct}")
        sep_parts.extend(["---", "---", "---"])
    header_parts.append("|")
    sep_parts.append("|")
    lines.append(" ".join(header_parts))
    lines.append(" ".join(sep_parts))

    for _, r in recal_df.iterrows():
        parts = [f"| {r['month']}"]
        for topk_pct in TOP_K_PCTS:
            parts.append(f"{r.get(f'recall_top{topk_pct}', float('nan')):.4f}")
            parts.append(f"{r.get(f'max_recall_top{topk_pct}', float('nan')):.4f}")
            parts.append(f"{r.get(f'norm_recall_top{topk_pct}', float('nan')):.4f}")
        parts.append("|")
        lines.append(" ".join(parts))
    lines.append("")

    # Averages
    lines.append("**Means across months:**")
    for topk_pct in TOP_K_PCTS:
        key = f"mean_norm_recall_top{topk_pct}"
        if key in summary:
            lines.append(f"- Normalised recall at top-{topk_pct}%: {summary[key]:.4f}")
    lines.append("")

    lines += [
        "## Alert Budget Metrics",
        "",
        "For each budget, hours are sorted by negative_prob descending and the",
        "top budget% are flagged as alerts.",
        "",
    ]
    for budget_pct in ALERT_BUDGET_PCTS:
        lines.append(f"### Budget = {budget_pct}%")
        lines.append("")
        lines.append("| Month | Precision | Recall | Lift | F1 |")
        lines.append("|-------|-----------|--------|------|----|")
        for _, r in recal_df.iterrows():
            p = r.get(f"precision_alert{budget_pct}", float("nan"))
            rc = r.get(f"recall_alert{budget_pct}", float("nan"))
            lf = r.get(f"lift_alert{budget_pct}", float("nan"))
            f1 = r.get(f"f1_alert{budget_pct}", float("nan"))
            lines.append(f"| {r['month']} | {p:.4f} | {rc:.4f} | {lf:.4f} | {f1:.4f} |")
        lines.append("")
        mp = summary.get(f"mean_precision_alert{budget_pct}", float("nan"))
        mr = summary.get(f"mean_recall_alert{budget_pct}", float("nan"))
        ml = summary.get(f"mean_lift_alert{budget_pct}", float("nan"))
        mf = summary.get(f"mean_f1_alert{budget_pct}", float("nan"))
        lines.append(f"**Means:** precision={mp:.4f}, recall={mr:.4f}, lift={ml:.4f}, f1={mf:.4f}")
        lines.append("")

    lines += [
        "## New Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        "| Criterion | Value | Threshold | Pass? |",
        "|-----------|-------|-----------|-------|",
        f"| mean_auc | {summary['mean_auc']:.4f} | >= 0.90 | "
        f"{'Yes' if summary['mean_auc'] >= 0.90 else 'No'} |",
        f"| mean_f1 | {summary['mean_f1']:.4f} | >= 0.70 | "
        f"{'Yes' if summary['mean_f1'] >= 0.70 else 'No'} |",
        f"| mean_recall_at_20pct_alert | {summary['mean_recall_at_20pct_alert']:.4f} | >= 0.65 | "
        f"{'Yes' if summary['mean_recall_at_20pct_alert'] >= 0.65 else 'No'} |",
        f"| n_sufficient_months | {summary['n_sufficient_months']} | >= 4 | "
        f"{'Yes' if summary['n_sufficient_months'] >= 4 else 'No'} |",
        "",
        "## Comparison with Old Verdict",
        "",
        "| Aspect | Old (raw top-k) | New (base-rate aware) |",
        "|--------|-----------------|----------------------|",
        "| Verdict | NEGATIVE_LOW_VALUE | " + summary['verdict'] + " |",
        "| Key metric | mean top10 capture = 0.365 | "
        f"mean norm recall@top10 = {summary.get('mean_norm_recall_top10', float('nan')):.4f} |",
        "| AUC | 0.880 | " + f"{summary['mean_auc']:.4f}" + " |",
        "| Issue | Raw capture penalised by high base rate | "
        "Normalised recall accounts for base rate ceiling |",
        "",
        "## Criteria Reference",
        "",
        "| Verdict | Condition |",
        "|---------|-----------|",
        "| NEGATIVE_CHAMPION | mean_auc >= 0.90, mean_f1 >= 0.70, mean_recall_at_20pct_alert >= 0.65, >= 4 sufficient months |",
        "| NEGATIVE_ACCEPTABLE | mean_auc >= 0.85, mean_f1 >= 0.60 |",
        "| NEGATIVE_AUX | mean_auc >= 0.80 |",
        "| NEGATIVE_NO_GO | otherwise |",
        "",
    ]

    path = out_dir / "negative_recalibration_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)


def main():
    args = parse_args()
    backtest_root = Path(args.backtest_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs
    monthly_metrics = load_monthly_metrics(backtest_root)
    predictions = load_predictions(backtest_root)

    # 2. Build recalibration table
    recal_df = build_recalibration_table(monthly_metrics, predictions)
    if recal_df.empty:
        logger.error("No months with both metrics and predictions -- aborting.")
        sys.exit(1)

    # 3. Compute summary & verdict
    summary = compute_summary(recal_df)
    logger.info("Recalibration verdict: %s", summary["verdict"])

    # 4. Write outputs
    write_json_summary(summary, out_dir)
    write_monthly_csv(recal_df, out_dir)
    write_markdown_report(summary, recal_df, out_dir)

    logger.info("Recalibration complete. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()
