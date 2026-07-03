#!/usr/bin/env python
"""Run residual baseline lab comparing 8 models on 2026-02 with real SGDFNet.

Usage:
    python scripts/run_residual_baseline_lab.py \
      --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
      --sgdfnet-predictions reports/local/deep_final/sgdfnet_predictions/sgdfnet_consolidated_2026_01_05.csv \
      --target-month 2026-02 \
      --out-dir reports/local/deep_final/residual_baseline_lab_2026_02
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.realtime_feature_builder import (
    build_realtime_features, audit_feature_coverage,
)
from models.deep_sgdf_delta.realtime_dataset_final import (
    build_training_datasets_final, collate_fn_final,
)
from models.deep_sgdf_delta.realtime_feature_contract import get_period
from models.deep_sgdf_delta.metrics import smape_floor50, compute_full_metrics
from models.deep_sgdf_delta.residual_baselines import (
    baseline_sgdfnet_only, baseline_da_anchor_only, baseline_mean_bias,
    baseline_hour_bias, baseline_period_bias,
    HGBResidualModel, RidgeResidualModel, MLPResidualModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("residual_baseline_lab")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Residual baseline lab")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--sgdfnet-predictions", type=str, required=True)
    parser.add_argument("--target-month", type=str, default="2026-02")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────

def _compute_bucket_metrics(df, col="rt_pred"):
    buckets = {
        "normal": (df["rt_true"] >= 0) & (df["rt_true"] < 500),
        "negative": df["rt_true"] < 0,
        "spike": df["rt_true"] >= 500,
    }
    results = {}
    for name, mask in buckets.items():
        if mask.sum() > 0:
            results[name] = float(smape_floor50(
                df.loc[mask, "rt_true"].values, df.loc[mask, col].values,
            ))
    return results


def _smape(y_true, y_pred):
    return float(smape_floor50(np.asarray(y_true), np.asarray(y_pred)))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "reports" / "local" / "deep_final" / "residual_baseline_lab_2026_02"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────
    logger.info("Loading data from %s", args.data_path)
    try:
        raw = pd.read_csv(args.data_path, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError):
        raw = pd.read_csv(args.data_path, encoding="gbk")

    sgd = pd.read_csv(args.sgdfnet_predictions)
    logger.info("Loaded SGDFNet predictions: %d rows", len(sgd))

    # ── Build features ─────────────────────────────────────────────
    logger.info("Building full features...")
    df = build_realtime_features(
        raw, sgdfnet_pred_df=sgd, mode="FULL_DAY",
        # allow fallback for pre-2026 training data; test data has 100% SGDFNet coverage
        allow_sgdfnet_fallback=True,
    )

    # ── Build train/test split ─────────────────────────────────────
    train_ds, val_ds, test_ds, manifest = build_training_datasets_final(
        df, target_month=args.target_month, val_days=30, train_min_days=90,
        allow_sgdfnet_fallback=True,
    )
    logger.info("Train: %d days, Val: %d days, Test: %d days",
                train_ds.n_days, val_ds.n_days, test_ds.n_days)

    # Collect dataframes for train and test
    train_df = df[df["business_day"].isin(train_ds.business_days)].copy()
    test_df = df[df["business_day"].isin(test_ds.business_days)].copy()
    logger.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    # Test true values
    y_true = test_df["rt_actual"].values
    sgdfnet_pred = test_df["sgdfnet_pred"].values
    da_anchor = test_df["da_anchor"].values

    # ── Run baselines ──────────────────────────────────────────────
    baselines = {}
    results = []

    # A: DA anchor only
    logger.info("A: DA anchor only")
    pred = baseline_da_anchor_only(test_df)
    baselines["DA_anchor"] = {"pred": pred, "time": 0}
    results.append({
        "model": "DA_anchor", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    # B: SGDFNet only
    logger.info("B: SGDFNet only")
    pred = baseline_sgdfnet_only(test_df)
    baselines["SGDFNet"] = {"pred": pred, "time": 0}
    results.append({
        "model": "SGDFNet", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    # C: SGDFNet + global mean bias
    logger.info("C: SGDFNet + global mean bias")
    t0 = time.time()
    resid_mean = (train_df["rt_actual"] - train_df["sgdfnet_pred"]).mean()
    pred = baseline_mean_bias(test_df, residual_mean=resid_mean)
    baselines["Mean_bias"] = {"pred": pred, "time": time.time() - t0}
    results.append({
        "model": "Mean_bias", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    # D: SGDFNet + hour-wise bias
    logger.info("D: SGDFNet + hour-wise bias")
    t0 = time.time()
    train_df["_resid"] = train_df["rt_actual"] - train_df["sgdfnet_pred"]
    hour_bias = train_df.groupby("hour_business")["_resid"].mean().to_dict()
    train_df = train_df.drop(columns=["_resid"])
    pred = baseline_hour_bias(test_df, hour_bias_map=hour_bias)
    baselines["Hour_bias"] = {"pred": pred, "time": time.time() - t0}
    results.append({
        "model": "Hour_bias", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    # E: SGDFNet + period-wise bias
    logger.info("E: SGDFNet + period-wise bias")
    t0 = time.time()
    train_df["_period"] = train_df["hour_business"].apply(get_period)
    train_df["_resid2"] = train_df["rt_actual"] - train_df["sgdfnet_pred"]
    period_bias = train_df.groupby("_period")["_resid2"].mean().to_dict()
    train_df = train_df.drop(columns=["_period", "_resid2"])
    pred = baseline_period_bias(test_df, period_bias_map=period_bias)
    baselines["Period_bias"] = {"pred": pred, "time": time.time() - t0}
    results.append({
        "model": "Period_bias", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    # F: SGDFNet + Ridge residual
    logger.info("F: SGDFNet + Ridge residual")
    t0 = time.time()
    ridge = RidgeResidualModel(alpha=1.0)
    ridge.fit(train_df)
    pred = ridge.predict(test_df)
    baselines["Ridge"] = {"pred": pred, "time": time.time() - t0}
    results.append({
        "model": "Ridge", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    # G: SGDFNet + HGB residual
    logger.info("G: SGDFNet + HGB residual")
    t0 = time.time()
    hgb = HGBResidualModel(max_iter=300)
    hgb.fit(train_df, target_col="residual_target")
    pred = hgb.predict(test_df)
    baselines["HGB"] = {"pred": pred, "time": time.time() - t0}
    results.append({
        "model": "HGB", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    # H: SGDFNet + MLP residual
    logger.info("H: SGDFNet + MLP residual")
    t0 = time.time()
    mlp = MLPResidualModel(hidden=(32, 16))
    mlp.fit(train_df)
    pred = mlp.predict(test_df)
    baselines["MLP"] = {"pred": pred, "time": time.time() - t0}
    results.append({
        "model": "MLP", "overall": _smape(y_true, pred),
        "1_8": _smape(y_true[test_df["hour_business"].between(1, 8)],
                       pred[test_df["hour_business"].between(1, 8)]),
        "9_16": _smape(y_true[test_df["hour_business"].between(9, 16)],
                        pred[test_df["hour_business"].between(9, 16)]),
        "17_24": _smape(y_true[test_df["hour_business"].between(17, 24)],
                         pred[test_df["hour_business"].between(17, 24)]),
    })

    elapsed = time.time() - t_start

    # ── Build leaderboard ──────────────────────────────────────────
    lb = pd.DataFrame(results)
    lb = lb.sort_values("overall").reset_index(drop=True)
    lb.to_csv(out_dir / "residual_baseline_leaderboard.csv", index=False)

    # Best model
    best = lb.iloc[0]
    best_name = best["model"]
    best_overall = best["overall"]

    # ── Determine signal ───────────────────────────────────────────
    sgdfnet_overall = results[1]["overall"]  # SGDFNet only
    da_overall = results[0]["overall"]       # DA anchor
    best_improvement = sgdfnet_overall - best_overall

    if best_name == "HGB" and best_overall < 20:
        verdict = "HGB_STRONG"
    elif best_name == "HGB" and best_overall < 23:
        verdict = "HGB_SIGNAL"
    elif best_improvement >= 0.3 and best_overall < 25:
        verdict = "WEAK_SIGNAL"
    elif best_improvement < 0.3 or best_overall >= 25:
        verdict = "NO_RESIDUAL_SIGNAL"
    else:
        verdict = "NO_RESIDUAL_SIGNAL"

    if best_overall > sgdfnet_overall:
        verdict = "BAD_RESIDUAL"

    # ── Detailed metrics ───────────────────────────────────────────
    period_rows = []
    for name, (lo, hi) in [("1_8", (1, 8)), ("9_16", (9, 16)), ("17_24", (17, 24))]:
        m = test_df["hour_business"].between(lo, hi)
        period_rows.append({
            "period": name,
            "count": int(m.sum()),
            "DA_anchor": _smape(y_true[m], da_anchor[m]),
            "SGDFNet": _smape(y_true[m], sgdfnet_pred[m]),
            best_name: _smape(y_true[m], baselines[best_name]["pred"][m]),
        })
    pd.DataFrame(period_rows).to_csv(out_dir / "period_metrics.csv", index=False)

    hourly_rows = []
    for h in range(1, 25):
        m = test_df["hour_business"] == h
        hourly_rows.append({
            "hour": h, "count": int(m.sum()),
            "DA_anchor": _smape(y_true[m], da_anchor[m]),
            "SGDFNet": _smape(y_true[m], sgdfnet_pred[m]),
            best_name: _smape(y_true[m], baselines[best_name]["pred"][m]),
        })
    pd.DataFrame(hourly_rows).to_csv(out_dir / "hourly_metrics.csv", index=False)

    # Buckets for best model
    bucket_rows = []
    test_df["rt_true"] = y_true
    for name, pred_key in [("DA_anchor", da_anchor), ("SGDFNet", sgdfnet_pred),
                           (best_name, baselines[best_name]["pred"])]:
        bucket_result = _compute_bucket_metrics(test_df.assign(rt_pred=pred_key), col="rt_pred")
        bucket_result["model"] = name
        bucket_rows.append(bucket_result)
    pd.DataFrame(bucket_rows).to_csv(out_dir / "bucket_metrics.csv", index=False)

    # Predictions
    pred_df = pd.DataFrame({
        "ds": test_df["ds"].values if "ds" in test_df.columns else range(len(y_true)),
        "hour_business": test_df["hour_business"].values,
        "rt_true": y_true,
        "da_anchor": da_anchor,
        "sgdfnet_pred": sgdfnet_pred,
    })
    for name, data in baselines.items():
        pred_df[name] = data["pred"]
    pred_df.to_csv(out_dir / "predictions_by_model.csv", index=False)

    # ── Signal report ──────────────────────────────────────────────
    signal_report = {
        "best_model": best_name,
        "best_overall_smape": round(best_overall, 2),
        "sgdfnet_overall_smape": round(sgdfnet_overall, 2),
        "da_anchor_overall_smape": round(da_overall, 2),
        "best_vs_sgdfnet_improvement_pp": round(best_improvement, 2),
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "sgdfnet_coverage": manifest.get("sgdfnet_coverage", 0),
    }
    (out_dir / "residual_baseline_metrics.json").write_text(
        json.dumps(signal_report, indent=2), encoding="utf-8"
    )

    # ── Markdown report ────────────────────────────────────────────
    lines = [
        "# Residual Baseline Lab Report",
        f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Target month: {args.target_month}*",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Overall | 1_8 | 9_16 | 17_24 |",
        "|------|-------|---------|-----|------|-------|",
    ]
    for i, (_, row) in enumerate(lb.iterrows()):
        lines.append(
            f"| {i+1} | {row['model']} | {row['overall']:.2f}% "
            f"| {row['1_8']:.2f}% | {row['9_16']:.2f}% | {row['17_24']:.2f}% |"
        )

    lines += [
        "",
        "## Signal Verdict",
        f"- **Best model**: {best_name} ({best_overall:.2f}%)",
        f"- **SGDFNet**: {sgdfnet_overall:.2f}%",
        f"- **DA anchor**: {da_overall:.2f}%",
        f"- **Best vs SGDFNet**: {best_improvement:+.2f}pp",
        f"- **Verdict**: **{verdict}**",
        "",
    ]
    if verdict == "HGB_STRONG":
        lines.append("### HGB < 20% — Strong residual signal. Continue residual modeling.")
    elif verdict == "HGB_SIGNAL":
        lines.append("### HGB < 23% — Residual signal exists. Deep residual-only may be viable.")
    elif verdict == "WEAK_SIGNAL":
        lines.append("### Weak residual signal — improvement < 0.3pp. Not worth deep model.")
    elif verdict == "NO_RESIDUAL_SIGNAL":
        lines.append("### No residual signal. Archive TrendKnightRT deep model.")
    elif verdict == "BAD_RESIDUAL":
        lines.append("### Residual correction makes things worse. Use SGDFNet or DA anchor directly.")

    lines += [
        "",
        "## Bucket Metrics",
    ]
    for br in bucket_rows:
        lines.append(f"- **{br.get('model', '?')}**: "
                      f"normal={br.get('normal', 'N/A'):.2f}%, "
                      f"negative={br.get('negative', 'N/A'):.2f}%, "
                      f"spike={br.get('spike', 'N/A'):.2f}%")

    (out_dir / "residual_signal_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ── Print summary ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESIDUAL BASELINE LAB — COMPLETE")
    print("=" * 60)
    print(f"  Best model:      {best_name} ({best_overall:.2f}%)")
    print(f"  SGDFNet only:    {sgdfnet_overall:.2f}%")
    print(f"  DA anchor only:  {da_overall:.2f}%")
    print(f"  Improvement:     {best_improvement:+.2f}pp")
    print(f"  Verdict:         {verdict}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print("=" * 60)
    for _, row in lb.iterrows():
        marker = " << BEST" if row["model"] == best_name else ""
        print(f"  {row['model']:15s}  {row['overall']:.2f}%{marker}")

    return signal_report


if __name__ == "__main__":
    main()
