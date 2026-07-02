"""Evaluation and go/no-go reporting for DeepSGDFDelta.

Outputs:
  - metrics_summary.json
  - monthly_metrics.csv
  - predictions.csv
  - go_nogo.md
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .metrics import compute_full_metrics, compute_monthly_metrics, smape_floor50

logger = logging.getLogger(__name__)

# ── Go/No-Go thresholds ─────────────────────────────────────────────
PASS_THRESHOLD = 15.0
SOFT_PASS_THRESHOLD = 15.8
BASELINE_SGDFNET = 16.5902


def evaluate_predictions(
    pred_df: pd.DataFrame,
    *,
    run_id: str,
    output_dir: Path,
    sgdfnet_baseline: float = BASELINE_SGDFNET,
    spike_threshold: float = 500.0,
) -> dict:
    """Evaluate predictions and write all report files.

    pred_df must contain: business_day, hour (target_hour), period (segment),
    ds (timestamp), y_true (rt_actual), y_pred (rt_pred), delta_pred, da_anchor
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standardise column names
    col_map = {}
    if "rt_actual" in pred_df.columns:
        col_map["rt_actual"] = "y_true"
    if "rt_pred" in pred_df.columns:
        col_map["rt_pred"] = "y_pred"
    if "target_hour" in pred_df.columns:
        col_map["target_hour"] = "hour"
    if "segment" in pred_df.columns:
        col_map["segment"] = "period"

    df = pred_df.rename(columns=col_map).copy()

    # Ensure required columns
    required = ["y_true", "y_pred", "delta_pred", "da_anchor"]
    if "hour" not in df.columns and "target_hour" in df.columns:
        df["hour"] = df["target_hour"]

    # Compute metrics
    metrics_df = df.rename(columns={"y_true": "rt_actual", "y_pred": "rt_pred"})
    if "hour" in metrics_df.columns:
        metrics_df["hour"] = metrics_df["hour"].astype(int)

    full_metrics = compute_full_metrics(metrics_df, spike_threshold=spike_threshold)
    monthly_df = compute_monthly_metrics(metrics_df)

    # Monthly average
    if not monthly_df.empty:
        monthly_avg = float(monthly_df["sMAPE_floor50"].mean())
    else:
        monthly_avg = float("nan")

    full_metrics["monthly_avg_sMAPE_floor50"] = monthly_avg

    # Go/No-Go decision
    overall = full_metrics.get("overall_sMAPE_floor50", float("nan"))
    if overall < PASS_THRESHOLD:
        verdict = "PASS"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} < {PASS_THRESHOLD}"
    elif overall <= SOFT_PASS_THRESHOLD:
        verdict = "SOFT_PASS"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} <= {SOFT_PASS_THRESHOLD}, awaiting spike/negative module fusion"
    elif overall < sgdfnet_baseline:
        verdict = "BASELINE_PASS"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} < SGDFNet baseline {sgdfnet_baseline:.4f}"
    else:
        verdict = "NO-GO"
        verdict_detail = f"Overall sMAPE_floor50={overall:.4f} >= SGDFNet baseline {sgdfnet_baseline:.4f}"

    full_metrics["verdict"] = verdict
    full_metrics["verdict_detail"] = verdict_detail

    # Write outputs
    # 1. predictions.csv
    pred_output_cols = ["business_day", "hour", "period", "ds", "y_true", "y_pred", "delta_pred", "da_anchor"]
    available_cols = [c for c in pred_output_cols if c in df.columns]
    df[available_cols].to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # 2. metrics_summary.json
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, ensure_ascii=False, indent=2)

    # 3. monthly_metrics.csv
    if not monthly_df.empty:
        monthly_df.to_csv(output_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")

    # 4. go_nogo.md
    go_nogo_lines = [
        f"# Go/No-Go Report: {run_id}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Verdict:** {verdict}",
        "",
        f"{verdict_detail}",
        "",
        "## Key Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall sMAPE_floor50 | {overall:.4f} |",
        f"| Monthly Avg sMAPE_floor50 | {monthly_avg:.4f} |",
        f"| 9_16 sMAPE_floor50 | {full_metrics.get('9_16_sMAPE_floor50', float('nan')):.4f} |",
        f"| Normal sMAPE_floor50 | {full_metrics.get('normal_sMAPE_floor50', float('nan')):.4f} |",
        f"| Delta MAE | {full_metrics.get('delta_mae', float('nan')):.4f} |",
        f"| Rows Total | {full_metrics.get('rows_total', 0)} |",
        f"| Rows Missing | {full_metrics.get('rows_missing', 0)} |",
        "",
        "## Thresholds",
        "",
        f"- PASS: monthly avg sMAPE_floor50 < {PASS_THRESHOLD}",
        f"- SOFT PASS: overall sMAPE_floor50 <= {SOFT_PASS_THRESHOLD}",
        f"- BASELINE PASS: better than SGDFNet corrected baseline {sgdfnet_baseline:.4f}",
        f"- NO-GO: worse than SGDFNet baseline or leakage detected",
        "",
    ]

    if not monthly_df.empty:
        go_nogo_lines.append("## Monthly Breakdown")
        go_nogo_lines.append("")
        go_nogo_lines.append("| Month | sMAPE_floor50 | Count |")
        go_nogo_lines.append("|-------|---------------|-------|")
        for _, row in monthly_df.iterrows():
            go_nogo_lines.append(f"| {row['month']} | {row['sMAPE_floor50']:.4f} | {row['count']} |")
        go_nogo_lines.append("")

    (output_dir / "go_nogo.md").write_text("\n".join(go_nogo_lines), encoding="utf-8")

    logger.info(f"Evaluation complete: {verdict} — {verdict_detail}")
    return full_metrics
