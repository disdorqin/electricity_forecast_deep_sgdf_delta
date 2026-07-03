#!/usr/bin/env python
"""Evaluate SpikeRisk module predictions.

Usage:
    python scripts/evaluate_spike_risk_module.py \
        --predictions artifacts/spike_risk/exp_2026_02/predictions.csv \
        --out-dir reports/local/risk_modules/spike_risk_eval_2026_02
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

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score,
)
from models.deep_sgdf_delta.metrics import smape_floor50 as _canonical_smape_floor50

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SpikeRisk module")
    p.add_argument("--predictions", required=True, help="Path to predictions.csv")
    p.add_argument("--out-dir", required=True, help="Output directory")
    return p.parse_args()


def compute_smape_floor50(y_true, y_pred):
    """sMAPE with floor-50 capping -- delegates to canonical implementation."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(yt) | np.isnan(yp))
    if valid.sum() == 0:
        return float("nan")
    return _canonical_smape_floor50(yt[valid], yp[valid])


def compute_classification_metrics(y_true, y_prob, label_name):
    """Compute classification metrics for a single label."""
    valid = y_true >= 0
    if valid.sum() < 5:
        return {"label": label_name, "n_valid": int(valid.sum()), "status": "insufficient_data"}

    yt = y_true[valid].astype(int)
    yp = y_prob[valid]

    if len(np.unique(yt)) < 2:
        return {"label": label_name, "n_valid": int(valid.sum()), "positive_rate": float(yt.mean()),
                "status": "single_class"}

    y_pred_binary = (yp >= 0.5).astype(int)

    metrics = {
        "label": label_name,
        "n_valid": int(valid.sum()),
        "n_positive": int(yt.sum()),
        "positive_rate": float(yt.mean()),
        "precision": float(precision_score(yt, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(yt, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(yt, y_pred_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(yt, yp)),
        "pr_auc": float(average_precision_score(yt, yp)),
    }
    return metrics


def compute_topk_capture(y_true, y_prob, label_name):
    """Compute top-k capture rates, lift, and alert rate."""
    valid = y_true >= 0
    if valid.sum() < 10:
        return []

    yt = y_true[valid].astype(int)
    yp = y_prob[valid]
    n = len(yt)
    overall_pos_rate = yt.mean()

    rows = []
    for k_pct in [1, 3, 5, 10, 20]:
        k = max(1, int(n * k_pct / 100))
        top_k_idx = np.argsort(yp)[-k:]
        top_k_actual = yt[top_k_idx]
        capture_rate = float(top_k_actual.mean())
        lift = capture_rate / overall_pos_rate if overall_pos_rate > 0 else float("nan")
        rows.append({
            "label": label_name,
            "top_k_pct": k_pct,
            "k": k,
            "n_top_k": k,
            "n_positive_in_top_k": int(top_k_actual.sum()),
            "capture_rate": capture_rate,
            "lift": lift,
            "alert_rate": k / n,
        })
    return rows


def determine_spike_verdict(topk_rows):
    """Determine SPIKE_GO / SPIKE_LOW_VALUE / SPIKE_NO_GO verdict.

    SPIKE_GO: top10% lift >= 2.0 AND recall_at_top20% >= 0.5
    SPIKE_LOW_VALUE: any top-k lift >= 1.3
    SPIKE_NO_GO: otherwise
    """
    if not topk_rows:
        return "SPIKE_NO_GO", "No top-k capture results"

    # Find top10% and top20% rows for the primary label (spike_label)
    lift_at_10 = None
    capture_at_20 = None
    max_lift = 0.0

    for row in topk_rows:
        if row["top_k_pct"] == 10:
            lift_at_10 = row["lift"]
        if row["top_k_pct"] == 20:
            capture_at_20 = row["capture_rate"]
        if not np.isnan(row.get("lift", 0)):
            max_lift = max(max_lift, row["lift"])

    # SPIKE_GO: top10% lift >= 2.0 AND recall_at_top20% >= 0.5
    if lift_at_10 is not None and capture_at_20 is not None:
        if lift_at_10 >= 2.0 and capture_at_20 >= 0.5:
            return "SPIKE_GO", (
                f"top10% lift={lift_at_10:.2f}>=2.0 AND "
                f"recall_at_top20%={capture_at_20:.3f}>=0.5"
            )

    # SPIKE_LOW_VALUE: any lift >= 1.3
    if max_lift >= 1.3:
        return "SPIKE_LOW_VALUE", f"Max lift across top-k = {max_lift:.2f} >= 1.3"

    return "SPIKE_NO_GO", f"Max lift across top-k = {max_lift:.2f} < 1.3"


def _df_to_md(df):
    """Simple DataFrame to markdown table."""
    if df.empty:
        return "(empty)"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
        ) + " |")
    return "\n".join([header, sep] + rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pred_df = pd.read_csv(args.predictions)
    logger.info("Loaded %d predictions", len(pred_df))

    # ── Classification metrics ────────────────────────────────────────
    class_metrics = []
    label_prob_pairs = [
        ("spike_label", "spike_prob"),
        ("extreme_spike_label", "extreme_spike_prob"),
        ("relative_spike_label", "relative_spike_prob"),
    ]

    for label, prob_col in label_prob_pairs:
        if label in pred_df.columns and prob_col in pred_df.columns:
            m = compute_classification_metrics(
                pred_df[label].values, pred_df[prob_col].values, label
            )
            class_metrics.append(m)

    class_df = pd.DataFrame(class_metrics)
    class_df.to_csv(out_dir / "classification_metrics.csv", index=False)
    logger.info("Classification metrics:\n%s", class_df.to_string())

    # ── Top-k capture, lift, alert_rate ───────────────────────────────
    all_topk_rows = []
    for label, prob_col in label_prob_pairs:
        if label in pred_df.columns and prob_col in pred_df.columns:
            rows = compute_topk_capture(
                pred_df[label].values, pred_df[prob_col].values, label
            )
            all_topk_rows.extend(rows)

    topk_df = pd.DataFrame(all_topk_rows)
    topk_df.to_csv(out_dir / "topk_capture.csv", index=False)
    logger.info("Top-k capture:\n%s", topk_df.to_string())

    # ── DA anchor sMAPE (canonical) ──────────────────────────────────
    da_smape = float("nan")
    if "da_anchor" in pred_df.columns and "rt_actual" in pred_df.columns:
        da_smape = compute_smape_floor50(
            pred_df["rt_actual"].values, pred_df["da_anchor"].values
        )
        logger.info("DA anchor sMAPE floor50: %.4f", da_smape)

    # ── Verdict ───────────────────────────────────────────────────────
    verdict, reason = determine_spike_verdict(all_topk_rows)
    logger.info("Verdict: %s -- %s", verdict, reason)

    # ── Metrics summary ──────────────────────────────────────────────
    summary = {
        "n_predictions": len(pred_df),
        "verdict": verdict,
        "verdict_reason": reason,
        "da_anchor_smape_floor50": da_smape if not np.isnan(da_smape) else None,
        "classification": class_metrics,
        "topk_capture": all_topk_rows,
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── GO/NO-GO report ──────────────────────────────────────────────
    go_nogo = f"""# SpikeRisk Evaluation: {verdict}

## Verdict: {verdict}

**Reason**: {reason}

## DA Anchor sMAPE (canonical floor50)

- DA anchor sMAPE floor50: {da_smape:.4f}

## Classification Metrics

{_df_to_md(class_df) if not class_df.empty else "No classification metrics available."}

## Top-k Capture, Lift, Alert Rate

{_df_to_md(topk_df) if not topk_df.empty else "No top-k capture results."}

## Summary

- Total predictions: {len(pred_df)}
- Verdict: {verdict}
"""
    with open(out_dir / "go_nogo.md", "w", encoding="utf-8") as f:
        f.write(go_nogo)

    logger.info("Evaluation complete. Verdict: %s", verdict)
    logger.info("Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
