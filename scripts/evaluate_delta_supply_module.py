#!/usr/bin/env python
"""Evaluate DeltaSupply module predictions.

Usage:
    python scripts/evaluate_delta_supply_module.py \
        --predictions artifacts/delta_supply/exp_2026_02/predictions.csv \
        --out-dir reports/local/delta_supply/eval_2026_02
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, mean_absolute_error, mean_squared_error,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeltaSupply module")
    p.add_argument("--predictions", required=True, help="Path to predictions.csv")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--correction-weights", default="0.0,0.1,0.2,0.3,0.5,1.0",
                   help="Comma-separated correction weights to test")
    return p.parse_args()


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

    # Top-k capture rate: among top-k predicted, how many are actual positives
    for k_pct in [5, 10, 20]:
        k = max(1, int(len(yt) * k_pct / 100))
        top_k_idx = np.argsort(yp)[-k:]
        top_k_actual = yt[top_k_idx]
        metrics[f"top_k_capture_rate_{k_pct}pct"] = float(top_k_actual.mean())

    return metrics


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid.sum() < 5:
        return {"status": "insufficient_data"}

    yt = y_true[valid]
    yp = y_pred[valid]

    corr = np.corrcoef(yt, yp)[0, 1] if len(yt) > 2 else 0.0

    return {
        "n_valid": int(valid.sum()),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "magnitude_corr": float(corr) if not np.isnan(corr) else 0.0,
    }


from models.deep_sgdf_delta.metrics import smape_floor50 as _canonical_smape_floor50


def compute_smape_floor50(y_true, y_pred):
    """sMAPE with floor-50 capping — delegates to canonical implementation."""
    valid = ~(np.isnan(np.asarray(y_true, dtype=float)) | np.isnan(np.asarray(y_pred, dtype=float)))
    if valid.sum() == 0:
        return np.nan
    return _canonical_smape_floor50(np.asarray(y_true)[valid], np.asarray(y_pred)[valid])


def run_correction_simulation(pred_df, correction_weights):
    """Simulate price correction: corrected = da_anchor + magnitude_pred * weight.

    Only uses model-predicted deviation_magnitude_pred, not test actual.
    Test actual is only used for evaluation of corrected predictions.
    """
    results = []
    da = pred_df["da_anchor"].values
    rt = pred_df["rt_actual"].values
    mag_pred = pred_df["deviation_magnitude_pred"].values

    valid = ~(np.isnan(da) | np.isnan(rt) | np.isnan(mag_pred))
    if valid.sum() == 0:
        return pd.DataFrame()

    da_v = da[valid]
    rt_v = rt[valid]
    mag_v = mag_pred[valid]

    # DA anchor baseline
    da_smape = compute_smape_floor50(rt_v, da_v)

    for w in correction_weights:
        corrected = da_v + mag_v * w
        corrected_smape = compute_smape_floor50(rt_v, corrected)
        improvement = da_smape - corrected_smape

        # Bucket analysis
        # Normal: rt_v in [0, 500)
        normal_mask = (rt_v >= 0) & (rt_v < 500)
        neg_mask = rt_v < 0
        spike_mask = rt_v >= 500

        normal_da = compute_smape_floor50(rt_v[normal_mask], da_v[normal_mask]) if normal_mask.any() else np.nan
        normal_corr = compute_smape_floor50(rt_v[normal_mask], corrected[normal_mask]) if normal_mask.any() else np.nan

        neg_da = compute_smape_floor50(rt_v[neg_mask], da_v[neg_mask]) if neg_mask.any() else np.nan
        neg_corr = compute_smape_floor50(rt_v[neg_mask], corrected[neg_mask]) if neg_mask.any() else np.nan

        spike_da = compute_smape_floor50(rt_v[spike_mask], da_v[spike_mask]) if spike_mask.any() else np.nan
        spike_corr = compute_smape_floor50(rt_v[spike_mask], corrected[spike_mask]) if spike_mask.any() else np.nan

        results.append({
            "correction_weight": w,
            "da_anchor_smape": da_smape,
            "corrected_smape": corrected_smape,
            "improvement_pp": improvement,
            "normal_bucket_da": normal_da,
            "normal_bucket_corrected": normal_corr,
            "normal_bucket_delta": (normal_corr - normal_da) if not np.isnan(normal_da) else np.nan,
            "negative_bucket_da": neg_da,
            "negative_bucket_corrected": neg_corr,
            "negative_bucket_delta": (neg_corr - neg_da) if not np.isnan(neg_da) else np.nan,
            "spike_bucket_da": spike_da,
            "spike_bucket_corrected": spike_corr,
            "spike_bucket_delta": (spike_corr - spike_da) if not np.isnan(spike_da) else np.nan,
        })

    return pd.DataFrame(results)


def determine_go_nogo(correction_df):
    """Determine GO / LOW_VALUE / NO-GO verdict."""
    if correction_df.empty:
        return "NO-GO", "No correction simulation results"

    best_idx = correction_df["improvement_pp"].idxmax()
    best = correction_df.loc[best_idx]
    improvement = best["improvement_pp"]
    neg_delta = best.get("negative_bucket_delta", 0)

    if improvement >= 0.005:  # >= 0.5pp
        if not np.isnan(neg_delta) and neg_delta > 0.01:  # worsens > 1pp
            return "LOW_VALUE", f"Improvement {improvement:.4f} but negative bucket worsens {neg_delta:.4f}"
        return "GO", f"Best correction weight={best['correction_weight']}, improvement={improvement:.4f}"
    elif improvement > 0.001:  # 0.1~0.5pp
        return "LOW_VALUE", f"Marginal improvement {improvement:.4f}"
    else:
        return "NO-GO", f"No improvement (best={improvement:.4f})"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pred_df = pd.read_csv(args.predictions)
    logger.info("Loaded %d predictions", len(pred_df))

    correction_weights = [float(w) for w in args.correction_weights.split(",")]

    # ── Classification metrics ────────────────────────────────────────
    class_metrics = []

    for label, prob_col in [
        ("upward_deviation_label", "upward_deviation_prob"),
        ("downward_deviation_label", "downward_deviation_prob"),
        ("large_abs_deviation_label", "large_abs_deviation_prob"),
    ]:
        if label in pred_df.columns and prob_col in pred_df.columns:
            m = compute_classification_metrics(
                pred_df[label].values, pred_df[prob_col].values, label
            )
            class_metrics.append(m)

    class_df = pd.DataFrame(class_metrics)
    class_df.to_csv(out_dir / "classification_metrics.csv", index=False)
    logger.info("Classification metrics:\n%s", class_df.to_string())

    # ── Regression metrics ────────────────────────────────────────────
    reg_metrics = {}
    if "deviation_magnitude_target" in pred_df.columns and "deviation_magnitude_pred" in pred_df.columns:
        reg_metrics = compute_regression_metrics(
            pred_df["deviation_magnitude_target"].values,
            pred_df["deviation_magnitude_pred"].values,
        )

    reg_df = pd.DataFrame([reg_metrics])
    reg_df.to_csv(out_dir / "regression_metrics.csv", index=False)
    logger.info("Regression metrics: %s", reg_metrics)

    # ── Correction simulation ─────────────────────────────────────────
    if all(c in pred_df.columns for c in ["da_anchor", "rt_actual", "deviation_magnitude_pred"]):
        correction_df = run_correction_simulation(pred_df, correction_weights)
        correction_df.to_csv(out_dir / "correction_simulation.csv", index=False)
        logger.info("Correction simulation:\n%s", correction_df.to_string())

        verdict, reason = determine_go_nogo(correction_df)
        best_idx = correction_df["improvement_pp"].idxmax()
        best_weight = correction_df.loc[best_idx, "correction_weight"]
        best_improvement = correction_df.loc[best_idx, "improvement_pp"]
        da_smape = correction_df.loc[best_idx, "da_anchor_smape"]
        best_smape = correction_df.loc[best_idx, "corrected_smape"]
    else:
        correction_df = pd.DataFrame()
        verdict = "NO-GO"
        reason = "Missing required columns for correction simulation"
        best_weight = 0.0
        best_improvement = 0.0
        da_smape = np.nan
        best_smape = np.nan

    # ── Hourly / period / bucket metrics ──────────────────────────────
    hourly_metrics = []
    if "hour_business" in pred_df.columns:
        for hour, group in pred_df.groupby("hour_business"):
            if "da_anchor" in group.columns and "rt_actual" in group.columns:
                smape = compute_smape_floor50(group["rt_actual"].values, group["da_anchor"].values)
                hourly_metrics.append({"hour_business": hour, "da_anchor_smape": smape, "n": len(group)})
    pd.DataFrame(hourly_metrics).to_csv(out_dir / "hourly_metrics.csv", index=False)

    period_metrics = []
    if "period" in pred_df.columns:
        for period, group in pred_df.groupby("period"):
            if "da_anchor" in group.columns and "rt_actual" in group.columns:
                smape = compute_smape_floor50(group["rt_actual"].values, group["da_anchor"].values)
                period_metrics.append({"period": period, "da_anchor_smape": smape, "n": len(group)})
    pd.DataFrame(period_metrics).to_csv(out_dir / "period_metrics.csv", index=False)

    bucket_metrics = []
    if "rt_actual" in pred_df.columns:
        for bucket_name, mask_fn in [
            ("normal", lambda x: (x >= 0) & (x < 500)),
            ("negative", lambda x: x < 0),
            ("spike", lambda x: x >= 500),
        ]:
            mask = mask_fn(pred_df["rt_actual"].values)
            if mask.any() and "da_anchor" in pred_df.columns:
                smape = compute_smape_floor50(
                    pred_df.loc[mask, "rt_actual"].values,
                    pred_df.loc[mask, "da_anchor"].values,
                )
                bucket_metrics.append({"bucket": bucket_name, "da_anchor_smape": smape, "n": int(mask.sum())})
    pd.DataFrame(bucket_metrics).to_csv(out_dir / "bucket_metrics.csv", index=False)

    # ── Feature importance ────────────────────────────────────────────
    # Copy from training artifacts if available
    pred_path = Path(args.predictions)
    fi_path = pred_path.parent / "feature_importance.csv"
    if fi_path.exists():
        import shutil
        shutil.copy(fi_path, out_dir / "feature_importance.csv")

    # ── Metrics summary ───────────────────────────────────────────────
    summary = {
        "n_predictions": len(pred_df),
        "classification": class_metrics,
        "regression": reg_metrics,
        "correction_simulation": {
            "best_weight": best_weight,
            "da_anchor_smape": da_smape if not np.isnan(da_smape) else None,
            "best_corrected_smape": best_smape if not np.isnan(best_smape) else None,
            "improvement_pp": best_improvement,
            "verdict": verdict,
            "reason": reason,
        },
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── GO/NO-GO report ──────────────────────────────────────────────
    def _df_to_md(df):
        """Simple DataFrame to markdown table without tabulate."""
        if df.empty:
            return "(empty)"
        header = "| " + " | ".join(str(c) for c in df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join([header, sep] + rows)

    go_nogo = f"""# DeltaSupply Evaluation: {verdict}

## Verdict: {verdict}

**Reason**: {reason}

## Classification Metrics

{_df_to_md(class_df) if not class_df.empty else "No classification metrics available."}

## Regression Metrics

{json.dumps(reg_metrics, indent=2)}

## Correction Simulation

{_df_to_md(correction_df) if not correction_df.empty else "No correction simulation results."}

## Summary

- DA anchor sMAPE: {da_smape:.4f}
- Best corrected sMAPE: {best_smape:.4f}
- Best correction weight: {best_weight}
- Improvement: {best_improvement:.4f} ({best_improvement*100:.2f}pp)
"""
    with open(out_dir / "go_nogo.md", "w", encoding="utf-8") as f:
        f.write(go_nogo)

    logger.info("Evaluation complete. Verdict: %s", verdict)
    logger.info("Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
