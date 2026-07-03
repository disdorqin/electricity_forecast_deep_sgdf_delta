#!/usr/bin/env python
"""Calibrate DeltaSupply as a risk-only classifier (NOT for price correction).

Evaluates the deviation probability columns as a risk signal:
  - Threshold sweep (precision / recall / f1 / support / alert_rate)
  - Top-k capture (capture_rate / lift_vs_random / precision_at_k / recall_at_k)
  - Risk bucket calibration (predicted vs actual rate per decile)
  - Handoff verdict: RISK_FEATURE_GO / RISK_FEATURE_LOW_VALUE / RISK_FEATURE_NO_GO

Usage:
    python scripts/calibrate_delta_supply_risk.py \
        --predictions artifacts/delta_supply/exp_2026_02/predictions.csv \
        --out-dir reports/local/risk_modules/delta_supply_risk_calibration_2026_02
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Column mapping ──────────────────────────────────────────────────────────

RISK_TARGETS = [
    ("upward", "upward_deviation_prob", "upward_deviation_label"),
    ("downward", "downward_deviation_prob", "downward_deviation_label"),
    ("large_abs", "large_abs_deviation_prob", "large_abs_deviation_label"),
]

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()  # 0.05 .. 0.95
TOPK_PCTS = [1, 3, 5, 10, 20]
N_BUCKETS = 10


# ── Helpers ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Calibrate DeltaSupply risk classifier")
    p.add_argument("--predictions", required=True, help="Path to predictions.csv")
    p.add_argument("--out-dir", required=True, help="Output directory for calibration reports")
    return p.parse_args()


def _valid_mask(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Return boolean mask of rows with finite label and probability."""
    return np.isfinite(y_true) & np.isfinite(y_prob)


# ── 1. Threshold sweep ─────────────────────────────────────────────────────

def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Sweep thresholds and compute precision / recall / f1 / support / alert_rate.

    Parameters
    ----------
    y_true : array of 0/1 labels
    y_prob : array of predicted probabilities

    Returns
    -------
    DataFrame with one row per threshold.
    """
    mask = _valid_mask(y_true, y_prob)
    yt = y_true[mask].astype(int)
    yp = y_prob[mask]
    n = len(yt)
    n_pos = int(yt.sum())

    rows = []
    for thr in THRESHOLDS:
        y_pred = (yp >= thr).astype(int)
        tp = int(((y_pred == 1) & (yt == 1)).sum())
        fp = int(((y_pred == 1) & (yt == 0)).sum())
        fn = int(((y_pred == 0) & (yt == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = n_pos
        alert_rate = (tp + fp) / n if n > 0 else 0.0

        rows.append({
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "alert_rate": alert_rate,
        })

    return pd.DataFrame(rows)


def pick_best_thresholds(sweep_df: pd.DataFrame) -> dict:
    """Select best-f1, best-precision@recall>=0.30, best-recall@precision>=0.30."""
    if sweep_df.empty:
        return {"best_f1_threshold": None, "best_precision_at_recall_30": None,
                "best_recall_at_precision_30": None}

    # Best F1
    idx_f1 = sweep_df["f1"].idxmax()
    best_f1_thr = float(sweep_df.loc[idx_f1, "threshold"])

    # Best precision where recall >= 0.30
    eligible_prec = sweep_df[sweep_df["recall"] >= 0.30]
    if eligible_prec.empty:
        best_prec_thr = None
    else:
        idx = eligible_prec["precision"].idxmax()
        best_prec_thr = float(eligible_prec.loc[idx, "threshold"])

    # Best recall where precision >= 0.30
    eligible_rec = sweep_df[sweep_df["precision"] >= 0.30]
    if eligible_rec.empty:
        best_rec_thr = None
    else:
        idx = eligible_rec["recall"].idxmax()
        best_rec_thr = float(eligible_rec.loc[idx, "threshold"])

    return {
        "best_f1_threshold": best_f1_thr,
        "best_precision_at_recall_30": best_prec_thr,
        "best_recall_at_precision_30": best_rec_thr,
    }


# ── 2. Top-k capture ──────────────────────────────────────────────────────

def topk_capture(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Evaluate top-k risk hours.

    For each k%, select the top k% hours by predicted probability and compute:
      - capture_rate: fraction of actual positives captured
      - lift_vs_random: capture_rate / (k/100)
      - precision_at_k: among top-k, fraction that are actual positives
      - recall_at_k: among all positives, fraction in top-k
    """
    mask = _valid_mask(y_true, y_prob)
    yt = y_true[mask].astype(int)
    yp = y_prob[mask]
    n = len(yt)
    n_pos = int(yt.sum())
    base_rate = n_pos / n if n > 0 else 0.0

    # Sort descending by predicted probability
    order = np.argsort(-yp)
    yt_sorted = yt[order]

    rows = []
    for k_pct in TOPK_PCTS:
        k = max(1, int(n * k_pct / 100))
        top_k_labels = yt_sorted[:k]
        tp = int(top_k_labels.sum())

        capture_rate = tp / n_pos if n_pos > 0 else 0.0
        random_rate = k_pct / 100.0
        lift = capture_rate / random_rate if random_rate > 0 else 0.0
        precision_at_k = tp / k if k > 0 else 0.0
        recall_at_k = tp / n_pos if n_pos > 0 else 0.0

        rows.append({
            "topk_pct": k_pct,
            "k": k,
            "capture_rate": capture_rate,
            "lift_vs_random": lift,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
        })

    return pd.DataFrame(rows)


# ── 3. Risk bucket calibration ────────────────────────────────────────────

def bucket_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Bucket by predicted probability decile [0-0.1, 0.1-0.2, ..., 0.9-1.0].

    For each bucket compute:
      - bucket_pred_mean: mean predicted probability in bucket
      - bucket_actual_rate: fraction of actual positives
      - bucket_count: number of samples
      - calibration_error: |bucket_pred_mean - bucket_actual_rate|
    """
    mask = _valid_mask(y_true, y_prob)
    yt = y_true[mask].astype(int)
    yp = y_prob[mask]

    edges = np.linspace(0.0, 1.0, N_BUCKETS + 1)  # 0, 0.1, ..., 1.0
    rows = []
    for i in range(N_BUCKETS):
        lo, hi = edges[i], edges[i + 1]
        if i < N_BUCKETS - 1:
            bucket_mask = (yp >= lo) & (yp < hi)
        else:
            bucket_mask = (yp >= lo) & (yp <= hi)  # include right edge for last bucket

        count = int(bucket_mask.sum())
        if count > 0:
            pred_mean = float(yp[bucket_mask].mean())
            actual_rate = float(yt[bucket_mask].mean())
        else:
            pred_mean = float((lo + hi) / 2)
            actual_rate = 0.0

        rows.append({
            "bucket_lo": lo,
            "bucket_hi": hi,
            "bucket_pred_mean": pred_mean,
            "bucket_actual_rate": actual_rate,
            "bucket_count": count,
            "calibration_error": abs(pred_mean - actual_rate),
        })

    return pd.DataFrame(rows)


# ── 4. Handoff decision ──────────────────────────────────────────────────

def decide_verdict(topk_df: pd.DataFrame, bucket_df: pd.DataFrame) -> dict:
    """Determine risk feature handoff verdict.

    RISK_FEATURE_GO:
        top10% lift >= 2.0 AND recall_at_top20% >= 0.4 AND calibration roughly monotonic
    RISK_FEATURE_LOW_VALUE:
        top10% lift >= 1.3 but recall/calibration weak
    RISK_FEATURE_NO_GO:
        top10% lift < 1.3
    """
    # Extract top-10% lift
    row_10 = topk_df[topk_df["topk_pct"] == 10]
    lift_top10 = float(row_10["lift_vs_random"].iloc[0]) if not row_10.empty else 0.0

    # Extract recall at top-20%
    row_20 = topk_df[topk_df["topk_pct"] == 20]
    recall_top20 = float(row_20["recall_at_k"].iloc[0]) if not row_20.empty else 0.0

    # Check calibration monotonicity: actual_rate should generally increase across buckets
    nonempty = bucket_df[bucket_df["bucket_count"] > 0].copy()
    if len(nonempty) >= 2:
        actual_rates = nonempty["bucket_actual_rate"].values
        # Count how many consecutive pairs are monotonically non-decreasing
        n_increasing = sum(1 for i in range(len(actual_rates) - 1) if actual_rates[i + 1] >= actual_rates[i])
        n_pairs = len(actual_rates) - 1
        monotonic_ratio = n_increasing / n_pairs if n_pairs > 0 else 0.0
        calibration_monotonic = monotonic_ratio >= 0.5  # at least half the transitions increase
    else:
        calibration_monotonic = False

    # Decision logic
    if lift_top10 >= 2.0 and recall_top20 >= 0.4 and calibration_monotonic:
        verdict = "RISK_FEATURE_GO"
    elif lift_top10 >= 1.3:
        verdict = "RISK_FEATURE_LOW_VALUE"
    else:
        verdict = "RISK_FEATURE_NO_GO"

    return {
        "verdict": verdict,
        "lift_top10pct": lift_top10,
        "recall_top20pct": recall_top20,
        "calibration_monotonic": calibration_monotonic,
        "criteria": {
            "go_lift_threshold": 2.0,
            "go_recall_threshold": 0.4,
            "low_value_lift_threshold": 1.3,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_calibration(pred_df: pd.DataFrame) -> dict:
    """Run full risk calibration pipeline on a predictions DataFrame.

    Returns a dict with keys:
        per_direction: dict mapping direction name -> {threshold_sweep, topk_capture,
                       bucket_calibration, best_thresholds, verdict}
        sweep_dfs, topk_dfs, bucket_dfs, verdicts: aggregated outputs
    """
    results = {}

    for direction, prob_col, label_col in RISK_TARGETS:
        if prob_col not in pred_df.columns or label_col not in pred_df.columns:
            logger.warning("Missing columns for %s — skipping", direction)
            continue

        y_true = pred_df[label_col].values.astype(float)
        y_prob = pred_df[prob_col].values.astype(float)

        sweep_df = threshold_sweep(y_true, y_prob)
        best_thresholds = pick_best_thresholds(sweep_df)
        topk_df = topk_capture(y_true, y_prob)
        bucket_df = bucket_calibration(y_true, y_prob)
        verdict = decide_verdict(topk_df, bucket_df)

        results[direction] = {
            "threshold_sweep": sweep_df,
            "best_thresholds": best_thresholds,
            "topk_capture": topk_df,
            "bucket_calibration": bucket_df,
            "verdict": verdict,
        }

    return results


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(args.predictions)
    logger.info("Loaded %d predictions from %s", len(pred_df), args.predictions)

    results = run_calibration(pred_df)

    if not results:
        logger.error("No valid direction columns found — nothing to calibrate.")
        sys.exit(1)

    # ── Write per-direction outputs ───────────────────────────────────────
    all_sweeps = []
    all_topk = []
    all_buckets = []
    verdicts = {}

    for direction, res in results.items():
        sweep_df = res["threshold_sweep"]
        sweep_df.insert(0, "direction", direction)
        all_sweeps.append(sweep_df)

        topk_df = res["topk_capture"]
        topk_df.insert(0, "direction", direction)
        all_topk.append(topk_df)

        bucket_df = res["bucket_calibration"]
        bucket_df.insert(0, "direction", direction)
        all_buckets.append(bucket_df)

        verdicts[direction] = res["verdict"]

        logger.info(
            "[%s] verdict=%s  lift_top10%%=%.2f  recall_top20%%=%.2f  best_f1_thr=%.2f",
            direction,
            res["verdict"]["verdict"],
            res["verdict"]["lift_top10pct"],
            res["verdict"]["recall_top20pct"],
            res["best_thresholds"]["best_f1_threshold"]
            if res["best_thresholds"]["best_f1_threshold"] is not None else -1.0,
        )

    # ── Aggregate CSVs ────────────────────────────────────────────────────
    sweep_all = pd.concat(all_sweeps, ignore_index=True)
    sweep_all.to_csv(out_dir / "threshold_sweep.csv", index=False)

    topk_all = pd.concat(all_topk, ignore_index=True)
    topk_all.to_csv(out_dir / "topk_capture.csv", index=False)

    bucket_all = pd.concat(all_buckets, ignore_index=True)
    bucket_all.to_csv(out_dir / "bucket_calibration.csv", index=False)

    # ── Verdict JSON ──────────────────────────────────────────────────────
    risk_verdict = {
        "directions": verdicts,
        "overall_verdict": _overall_verdict(verdicts),
    }
    with open(out_dir / "risk_verdict.json", "w", encoding="utf-8") as f:
        json.dump(risk_verdict, f, ensure_ascii=False, indent=2)

    logger.info("Calibration complete. Outputs in %s", out_dir)
    logger.info("Overall verdict: %s", risk_verdict["overall_verdict"])


def _overall_verdict(direction_verdicts: dict) -> str:
    """Aggregate per-direction verdicts into one overall verdict.

    If any direction is GO → GO.
    If all are NO_GO → NO_GO.
    Otherwise → LOW_VALUE.
    """
    verdict_labels = [v["verdict"] for v in direction_verdicts.values()]
    if any(v == "RISK_FEATURE_GO" for v in verdict_labels):
        return "RISK_FEATURE_GO"
    if all(v == "RISK_FEATURE_NO_GO" for v in verdict_labels):
        return "RISK_FEATURE_NO_GO"
    return "RISK_FEATURE_LOW_VALUE"


if __name__ == "__main__":
    main()
