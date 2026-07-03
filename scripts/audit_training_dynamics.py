#!/usr/bin/env python
"""Audit training dynamics from training curves.

Reads training_curves.csv and metrics_summary.json to diagnose
why the model didn't learn.

Usage:
    python scripts/audit_training_dynamics.py \
        --model-dir artifacts/trendknight_rt/exp_tcn_real_sgdfnet_2026_02 \
        --out-dir reports/local/deep_final/training_dynamics
"""
from __future__ import annotations

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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_training_dynamics")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Audit training dynamics")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "reports" / "local" / "deep_final" / "training_dynamics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    curves_path = model_dir / "training_curves.csv"
    metrics_path = model_dir / "metrics_summary.json"

    if not curves_path.exists():
        logger.error("No training_curves.csv found in %s", model_dir)
        return
    if not metrics_path.exists():
        logger.error("No metrics_summary.json found in %s", model_dir)
        return

    curves = pd.read_csv(curves_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    n_epochs = len(curves)

    # Training loss trends
    train_loss_first = curves["train_loss"].iloc[0]
    train_loss_last = curves["train_loss"].iloc[-1]
    train_loss_min = curves["train_loss"].min()
    train_loss_trend = "down" if train_loss_last < train_loss_first * 0.95 else (
        "flat" if abs(train_loss_last - train_loss_first) < train_loss_first * 0.05 else "up"
    )

    # Val sMAPE trend
    val_smape_first = curves["val_smape_floor50"].iloc[0]
    val_smape_last = curves["val_smape_floor50"].iloc[-1]
    val_smape_min = curves["val_smape_floor50"].min()
    best_epoch = int(curves.loc[curves["val_smape_floor50"].idxmin(), "epoch"])
    val_trend = "up" if val_smape_last > val_smape_first * 1.05 else (
        "flat" if abs(val_smape_last - val_smape_first) < val_smape_first * 0.05 else "down"
    )

    # Overfitting check
    overfitting = train_loss_trend == "down" and val_trend == "up"

    # LR analysis
    lr_first = curves["lr"].iloc[0]
    lr_last = curves["lr"].iloc[-1]

    # Prediction residual std (from training curves)
    residual_mae_first = curves.get("train_residual_mae", pd.Series([np.nan])).iloc[0]
    residual_mae_last = curves.get("train_residual_mae", pd.Series([np.nan])).iloc[-1]

    # Diagnoses
    suggestions = []
    if val_trend == "up" and train_loss_trend == "down":
        suggestions.append("OVERFITTING")
        suggestions.append("STRONGER_REGULARIZATION")
    if val_trend == "up" and train_loss_trend in ("flat", "up"):
        suggestions.append("LOWER_LR")
    if best_epoch <= 2:
        suggestions.append("BEST_EPOCH_TOO_EARLY")
    if val_smape_first < 32 and val_smape_last >= val_smape_first:
        suggestions.append("NO_LEARNING_PROGRESS")

    suggestions = suggestions or ["NORMAL"]
    is_early_stop = n_epochs < curves["epoch"].max()

    # Report
    lines = [
        "# Training Dynamics Audit Report",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Model: {model_dir.name}*",
        "",
        "## Summary",
        f"- **Epochs run**: {n_epochs} (max: {int(curves['epoch'].max())})",
        f"- **Early stopping**: {is_early_stop}",
        f"- **Best epoch**: {best_epoch}",
        f"- **Best val sMAPE**: {val_smape_min:.4f}",
        f"- **Test sMAPE**: {metrics.get('best_val_smape_floor50', 'N/A')}",
        "",
        "## Training Loss",
        f"- **First**: {train_loss_first:.4f}",
        f"- **Last**: {train_loss_last:.4f}",
        f"- **Min**: {train_loss_min:.4f}",
        f"- **Trend**: {train_loss_trend}",
        "",
        "## Validation sMAPE",
        f"- **First**: {val_smape_first:.4f}",
        f"- **Last**: {val_smape_last:.4f}",
        f"- **Min**: {val_smape_min:.4f}",
        f"- **Trend**: {val_trend}",
        f"- **Overfitting**: {overfitting}",
        "",
        "## Learning Rate",
        f"- **First**: {lr_first:.6f}",
        f"- **Last**: {lr_last:.6f}",
        "",
        "## Residual MAE",
        f"- **First**: {residual_mae_first:.4f}",
        f"- **Last**: {residual_mae_last:.4f}",
        "",
        "## Diagnoses",
    ]
    for s in suggestions:
        lines.append(f"- **{s}**")

    lines += [
        "",
        "## Recommendations",
    ]
    if "NO_LEARNING_PROGRESS" in suggestions:
        lines.append("- Model shows no learning progress across epochs.")
        lines.append("- Val sMAPE increases from epoch 1, indicating the model is not learning meaningful patterns.")
        lines.append("- Suggested action: HGB_BASELINE_FIRST — check if residual signal exists before optimizing deep model.")
    if "OVERFITTING" in suggestions:
        lines.append("- Model is overfitting: train loss drops but val sMAPE increases.")
    if "LOWER_LR" in suggestions:
        lines.append("- Learning rate may be too high for this task.")

    lines += [
        "",
        "## Verdict",
    ]
    if "NO_LEARNING_PROGRESS" in suggestions:
        lines.append("**NO_DEEP_SIGNAL** — Deep model not learning. Recommend HGB baseline first.")
    elif "OVERFITTING" in suggestions:
        lines.append("**OVERFITTING** — Need stronger regularization.")
    else:
        lines.append("**NORMAL** — Training dynamics are reasonable.")

    (out_dir / "training_dynamics_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-20:]))


if __name__ == "__main__":
    main()
