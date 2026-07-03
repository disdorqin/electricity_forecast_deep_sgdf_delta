#!/usr/bin/env python
"""Train Solar916 residual correction model.

Walk-forward training:
  - Train on data before target month
  - Validate on last 30 days before target month
  - Test on target month

Outputs to reports/local/phase7/solar916/:
  predictions.csv
  metrics_summary.json
  hourly_metrics.csv
  bucket_metrics.csv
  feature_importance.csv
  go_nogo.md

Usage:
    python scripts/train_solar916_residual.py \\
        --target-month 2026-02 \\
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx \\
        --start-date 2026-01-01 --end-date 2026-02-28
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 50.0) -> float:
    yt = np.clip(np.abs(y_true), floor, None)
    yp = np.clip(np.abs(y_pred), floor, None)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


def main():
    parser = argparse.ArgumentParser(description="Train Solar916 Residual Model")
    parser.add_argument("--target-month", type=str, default=None, help="e.g. 2026-02")
    parser.add_argument("--target-months", type=str, default=None,
                        help="Comma-separated months for multi-month walk-forward, e.g. 2026-01,2026-02,2026-03")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="hist_gradient_boosting",
                        choices=["hist_gradient_boosting", "catboost", "lightgbm"])
    parser.add_argument("--source-repo-root", type=str, default=None,
                        help="Path to source repo for SGDFNet teacher adapter")
    parser.add_argument("--sgdfnet-predictions", type=str, default=None,
                        help="Path to SGDFNet predictions CSV (overrides teacher adapter)")
    parser.add_argument("--out-dir", type=str, default="reports/local/phase7/solar916")
    parser.add_argument("--auto-simplify", action="store_true", default=False,
                        help="Use simpler model for small datasets")
    parser.add_argument("--phase-label", type=str, default="Phase 7",
                        help="Label for logging/output (e.g. 'Phase 8 No-Leak')")
    args = parser.parse_args()

    if not args.target_month and not args.target_months:
        parser.error("Either --target-month or --target-months is required")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Solar916 Residual Training — %s", args.phase_label)
    logger.info("=" * 60)
    logger.info("  Target month(s): %s", args.target_months or args.target_month)
    logger.info("  Model type: %s", args.model_type)
    logger.info("  Auto-simplify: %s", args.auto_simplify)

    # ── Build dataset ────────────────────────────────────────────────
    from models.deep_sgdf_delta.solar916_dataset import build_solar916_dataset
    from models.deep_sgdf_delta.solar916_model import (
        Solar916Config, train_walk_forward, train_multi_month_walk_forward,
        smape_floor50 as smape_fn,
    )

    # Load SGDFNet predictions
    sgdf_preds = None
    if args.sgdfnet_predictions:
        sgdf_preds = pd.read_csv(args.sgdfnet_predictions, encoding="utf-8-sig")
        logger.info("Loaded SGDFNet predictions from %s (%d rows)",
                    args.sgdfnet_predictions, len(sgdf_preds))
    elif args.source_repo_root:
        try:
            from models.deep_sgdf_delta.teacher_adapters import sgdfnet_teacher
            sgdf_preds = sgdfnet_teacher.load_predictions(
                source_repo_root=args.source_repo_root,
            )
            if sgdf_preds is not None:
                logger.info("Loaded SGDFNet predictions via teacher adapter (%d rows)",
                            len(sgdf_preds))
        except Exception as exc:
            logger.warning("Teacher adapter failed: %s", exc)

    # Use a wide training window
    start = args.start_date or "2026-01-01"
    end = args.end_date or args.target_month + "-28"

    dataset, ds_info = build_solar916_dataset(
        data_path=args.data_path,
        sgdfnet_predictions=sgdf_preds,
        start_date=start,
        end_date=end,
        output_dir=str(out_dir / "dataset"),
    )

    logger.info("Dataset: %d samples, %d with SGDFNet predictions",
                ds_info["n_samples"], ds_info["n_sgdfnet_aligned"])

    # Drop rows without SGDFNet predictions
    dataset_valid = dataset.dropna(subset=["sgdfnet_pred", "sgdfnet_residual"]).copy()
    if len(dataset_valid) < 100:
        logger.error("Insufficient data with SGDFNet predictions: %d rows", len(dataset_valid))
        sys.exit(1)

    # ── Train ────────────────────────────────────────────────────────
    config = Solar916Config(model_type=args.model_type, auto_simplify=args.auto_simplify)

    # Multi-month mode
    if args.target_months:
        months = [m.strip() for m in args.target_months.split(",")]
        results = train_multi_month_walk_forward(dataset_valid, months, config)

        # Aggregate monthly metrics
        monthly_rows = []
        all_test_dfs = []
        for r in results:
            if "error" in r:
                logger.warning("Month %s failed: %s", r["target_month"], r["error"])
                continue
            month = r["target_month"]
            test_df = r["test_df"].copy()
            test_pred = r["test_pred"]
            sgdf_pred_vals = test_df["sgdfnet_pred"].values
            corrected_pred = sgdf_pred_vals + test_pred
            rt_vals = test_df["rt_actual"].values
            base_s = smape_floor50(rt_vals, sgdf_pred_vals)
            corr_s = smape_floor50(rt_vals, corrected_pred)
            monthly_rows.append({
                "target_month": month,
                "train_rows": r["train_rows"],
                "test_rows": r["test_rows"],
                "baseline_smape": base_s,
                "corrected_smape": corr_s,
                "improvement": base_s - corr_s,
            })
            test_df["solar916_residual_pred"] = test_pred
            test_df["solar916_corrected_pred"] = corrected_pred
            all_test_dfs.append(test_df)

        monthly_df = pd.DataFrame(monthly_rows)
        monthly_df.to_csv(out_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
        logger.info("Monthly metrics:\n%s", monthly_df.to_string())

        # Combined predictions
        if all_test_dfs:
            combined = pd.concat(all_test_dfs, ignore_index=True)
            combined.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

            # Overall metrics
            all_rt = np.concatenate([r["test_actual"] for r in results if "error" not in r])
            all_sgdf = np.concatenate([r["test_df"]["sgdfnet_pred"].values for r in results if "error" not in r])
            all_corr = np.concatenate([
                r["test_df"]["sgdfnet_pred"].values + r["test_pred"] for r in results if "error" not in r
            ])
            overall_base = smape_floor50(all_rt, all_sgdf)
            overall_corr = smape_floor50(all_rt, all_corr)
            metrics = {
                "target_months": months,
                "model_type": args.model_type,
                "auto_simplify": args.auto_simplify,
                "overall_baseline_smape": overall_base,
                "overall_corrected_smape": overall_corr,
                "overall_improvement": overall_base - overall_corr,
                "monthly": monthly_rows,
            }
            with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            logger.info("Overall: baseline=%.4f, corrected=%.4f, improvement=%.4f",
                        overall_base, overall_corr, overall_base - overall_corr)

        logger.info("Multi-month outputs written to %s", out_dir)
        return

    # Single-month mode (original logic)
    result = train_walk_forward(dataset_valid, args.target_month, config)

    # ── Compute metrics ──────────────────────────────────────────────
    test_df = result["test_df"].copy()
    test_pred = result["test_pred"]
    test_actual_residual = result["test_actual"]

    # Corrected prediction = sgdfnet_pred + residual_pred
    sgdfnet_pred_vals = test_df["sgdfnet_pred"].values
    corrected_pred = sgdfnet_pred_vals + test_pred
    rt_actual_vals = test_df["rt_actual"].values

    # Baseline sMAPE (SGDFNet only)
    baseline_smape = smape_floor50(rt_actual_vals, sgdfnet_pred_vals)
    # Corrected sMAPE
    corrected_smape = smape_floor50(rt_actual_vals, corrected_pred)
    improvement = baseline_smape - corrected_smape

    logger.info("=" * 60)
    logger.info("Results for %s:", args.target_month)
    logger.info("  9_16 baseline sMAPE: %.4f", baseline_smape)
    logger.info("  9_16 corrected sMAPE: %.4f", corrected_smape)
    logger.info("  Improvement: %.4f", improvement)

    # ── Hourly metrics ───────────────────────────────────────────────
    hourly_rows = []
    for hour in range(9, 17):
        mask = test_df["hour_business"].values == hour
        if mask.sum() == 0:
            continue
        h_base = smape_floor50(rt_actual_vals[mask], sgdfnet_pred_vals[mask])
        h_corr = smape_floor50(rt_actual_vals[mask], corrected_pred[mask])
        hourly_rows.append({
            "hour": hour,
            "count": int(mask.sum()),
            "baseline_smape": h_base,
            "corrected_smape": h_corr,
            "improvement": h_base - h_corr,
        })
    hourly_df = pd.DataFrame(hourly_rows)

    # ── Bucket metrics ───────────────────────────────────────────────
    bucket_rows = []
    for bucket_name, bucket_fn in [
        ("normal", lambda rt: (np.abs(rt) <= 500) & (rt >= 0)),
        ("spike", lambda rt: np.abs(rt) > 500),
        ("negative", lambda rt: rt < 0),
    ]:
        mask = bucket_fn(rt_actual_vals)
        if mask.sum() == 0:
            continue
        b_base = smape_floor50(rt_actual_vals[mask], sgdfnet_pred_vals[mask])
        b_corr = smape_floor50(rt_actual_vals[mask], corrected_pred[mask])
        bucket_rows.append({
            "bucket": bucket_name,
            "count": int(mask.sum()),
            "baseline_smape": b_base,
            "corrected_smape": b_corr,
            "improvement": b_base - b_corr,
        })
    bucket_df = pd.DataFrame(bucket_rows)

    # ── Feature importance ───────────────────────────────────────────
    fi_rows = []
    for fname, imp in sorted(result["feature_importance"].items(), key=lambda x: -x[1]):
        fi_rows.append({"feature": fname, "importance": imp})
    fi_df = pd.DataFrame(fi_rows)

    # ── Verdict ──────────────────────────────────────────────────────
    normal_improvement = None
    for row in bucket_rows:
        if row["bucket"] == "normal":
            normal_improvement = row["improvement"]
            break

    if improvement >= 1.0 and (normal_improvement is None or normal_improvement >= 0):
        verdict = "GO"
    elif improvement >= 0.3:
        verdict = "LOW-WEIGHT"
    else:
        verdict = "NO-GO"

    logger.info("  Verdict: %s", verdict)

    # ── Write outputs ────────────────────────────────────────────────
    # predictions.csv
    pred_df = test_df[["business_day", "hour_business", "ds", "period",
                        "rt_actual", "da_price", "sgdfnet_pred"]].copy()
    pred_df["solar916_residual_pred"] = test_pred
    pred_df["solar916_corrected_pred"] = corrected_pred
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # metrics_summary.json
    metrics = {
        "target_month": args.target_month,
        "model_type": args.model_type,
        "train_rows": result["train_rows"],
        "val_rows": result["val_rows"],
        "test_rows": result["test_rows"],
        "val_smape": result["val_smape"],
        "baseline_smape_916": baseline_smape,
        "corrected_smape_916": corrected_smape,
        "improvement": improvement,
        "normal_bucket_improvement": normal_improvement,
        "verdict": verdict,
        "dataset_samples": ds_info["n_samples"],
        "missing_features": ds_info["missing_features"],
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # hourly_metrics.csv
    hourly_df.to_csv(out_dir / "hourly_metrics.csv", index=False, encoding="utf-8-sig")

    # bucket_metrics.csv
    bucket_df.to_csv(out_dir / "bucket_metrics.csv", index=False, encoding="utf-8-sig")

    # feature_importance.csv
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # go_nogo.md
    go_nogo_lines = [
        f"# Solar916 Residual Model — Go/No-Go Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Target Month:** {args.target_month}",
        f"**Model Type:** {args.model_type}",
        f"**Verdict:** {verdict}",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| 9_16 Baseline sMAPE | {baseline_smape:.4f} |",
        f"| 9_16 Corrected sMAPE | {corrected_smape:.4f} |",
        f"| Improvement | {improvement:.4f} |",
        f"| Normal Bucket Improvement | {normal_improvement if normal_improvement is not None else 'N/A'} |",
        f"| Test Rows | {result['test_rows']} |",
        "",
        "## Verdict Criteria",
        "",
        "- GO: improvement >= 1.0 AND normal bucket not worse",
        "- LOW-WEIGHT: improvement 0.3~1.0",
        "- NO-GO: improvement < 0.3 or worse",
        "",
    ]

    # Hourly breakdown
    go_nogo_lines.extend([
        "## Hourly Breakdown",
        "",
        "| Hour | Baseline sMAPE | Corrected sMAPE | Improvement |",
        "|------|---------------|-----------------|-------------|",
    ])
    for _, row in hourly_df.iterrows():
        go_nogo_lines.append(
            f"| {int(row['hour'])} | {row['baseline_smape']:.4f} | "
            f"{row['corrected_smape']:.4f} | {row['improvement']:.4f} |"
        )

    go_nogo_lines.extend(["", "## Top Features", ""])
    for _, row in fi_df.head(10).iterrows():
        go_nogo_lines.append(f"- {row['feature']}: {row['importance']:.4f}")

    (out_dir / "go_nogo.md").write_text("\n".join(go_nogo_lines), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("All outputs written to %s", out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
