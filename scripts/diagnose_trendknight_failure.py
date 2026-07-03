#!/usr/bin/env python
"""Diagnose TrendKnightRT failure: why didn't the model learn?

Loads the trained model, runs prediction with full features, and analyzes
prediction behavior against DA anchor, SGDFNet, and ground truth.

Outputs detailed diagnosis reports.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.realtime_feature_builder import build_realtime_features
from models.deep_sgdf_delta.realtime_dataset_final import (
    build_training_datasets_final, collate_fn_final,
)
from models.deep_sgdf_delta.trendknight_rt import TrendKnightRTConfig, build_trendknight_rt
from models.deep_sgdf_delta.metrics import smape_floor50

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("diagnose_failure")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose TrendKnightRT failure")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--sgdfnet-predictions", type=str, default=None)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--target-month", type=str, default="2026-02")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "reports" / "local" / "deep_final" / "failure_diagnosis_2026_02"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data and build features ────────────────────────────────
    logger.info("Loading data from %s", args.data_path)
    try:
        raw = pd.read_csv(args.data_path, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError):
        raw = pd.read_csv(args.data_path, encoding="gbk")

    sgd_preds = None
    if args.sgdfnet_predictions:
        sgd_preds = pd.read_csv(args.sgdfnet_predictions)
        logger.info("Loaded SGDFNet predictions: %d rows", len(sgd_preds))

    logger.info("Building full features...")
    df = build_realtime_features(
        raw, sgdfnet_pred_df=sgd_preds, mode="FULL_DAY",
        allow_sgdfnet_fallback=True,
    )

    # ── Load model ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.model_dir) / "best_model.pt",
                      map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    config = TrendKnightRTConfig(**cfg_dict)
    model = build_trendknight_rt(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("Model loaded: %s params, input_dim=%d", sum(p.numel() for p in model.parameters()), config.input_dim)

    # ── Build test dataset ──────────────────────────────────────────
    _, _, test_ds, manifest = build_training_datasets_final(
        df, target_month=args.target_month, val_days=30, train_min_days=90,
        allow_sgdfnet_fallback=True,
    )
    loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                        collate_fn=collate_fn_final, num_workers=0)
    logger.info("Test dataset: %d days", test_ds.n_days)

    # ── Run prediction ──────────────────────────────────────────────
    all_rows = []
    gate_values = []
    conf_values = []

    with torch.no_grad():
        for batch in loader:
            feat = batch["features_24h"].to(device)
            seg = batch["segment_id"].to(device)
            da = batch["da_anchor_24"].to(device)
            sgd = batch["sgdfnet_pred_24"].to(device)
            hid = batch["hour_ids"].to(device)
            mask = batch["mask_24"].to(device)

            out = model(feat, seg, da, sgd, hid)

            rt_pred = out["trend_rt_pred_24"].cpu()
            delta_pred = out["delta_pred_24"].cpu()
            residual_pred = out["residual_to_sgdfnet_24"].cpu()
            confidence = out["confidence_24"].cpu()

            # Gate values
            if "gate" in out:
                gate_values.append(out["gate"].cpu().numpy().ravel())

            conf_values.append(confidence.numpy().ravel())

            rt_true = (da + batch["delta_target_24"]).cpu()
            B = feat.size(0)

            for i in range(B):
                for h in range(24):
                    if mask[i, h].item() == 0:
                        continue
                    all_rows.append({
                        "rt_pred": float(rt_pred[i, h]),
                        "rt_true": float(rt_true[i, h]),
                        "da_anchor": float(da[i, h]),
                        "sgdfnet_pred": float(sgd[i, h]),
                        "delta_pred": float(delta_pred[i, h]),
                        "residual_pred": float(residual_pred[i, h]),
                        "confidence": float(confidence[i, h]),
                        "hour_business": h + 1,
                    })

    pred_df = pd.DataFrame(all_rows)
    logger.info("Predictions: %d rows", len(pred_df))

    # ── Analysis ────────────────────────────────────────────────────
    target_var = pred_df["rt_true"].var()
    pred_var = pred_df["rt_pred"].var()
    residual_true = pred_df["rt_true"] - pred_df["sgdfnet_pred"]
    residual_pred = pred_df["residual_pred"]

    # Correlations
    corr_with_anchor = pred_df["rt_pred"].corr(pred_df["da_anchor"])
    corr_with_sgdfnet = pred_df["rt_pred"].corr(pred_df["sgdfnet_pred"])

    # DA anchor delta
    pred_da_delta = (pred_df["rt_pred"] - pred_df["da_anchor"]).abs().mean()
    anchor_smape = smape_floor50(pred_df["rt_true"].values, pred_df["da_anchor"].values)

    # Overall metrics
    overall_smape = smape_floor50(pred_df["rt_true"].values, pred_df["rt_pred"].values)
    sgdfnet_smape = smape_floor50(pred_df["rt_true"].values, pred_df["sgdfnet_pred"].values)
    residual_true_std = residual_true.std()
    residual_pred_std = residual_pred.std()

    # Period metrics
    periods = {"1_8": (1, 8), "9_16": (9, 16), "17_24": (17, 24)}
    period_metrics = {}
    for name, (lo, hi) in periods.items():
        mask = pred_df["hour_business"].between(lo, hi)
        if mask.sum() > 0:
            period_metrics[name] = {
                "smape": float(smape_floor50(
                    pred_df.loc[mask, "rt_true"].values,
                    pred_df.loc[mask, "rt_pred"].values,
                )),
                "anchor_smape": float(smape_floor50(
                    pred_df.loc[mask, "rt_true"].values,
                    pred_df.loc[mask, "da_anchor"].values,
                )),
            }

    # Hour metrics
    hour_metrics = {}
    for h in range(1, 25):
        mask = pred_df["hour_business"] == h
        if mask.sum() > 0:
            hour_metrics[f"hour_{h}"] = {
                "smape": float(smape_floor50(
                    pred_df.loc[mask, "rt_true"].values,
                    pred_df.loc[mask, "rt_pred"].values,
                )),
                "count": int(mask.sum()),
            }

    # Bucket metrics
    bucket_metrics = _compute_bucket_metrics(pred_df)

    # Worst days
    pred_df["business_day"] = pd.to_datetime(pred_df["ds"]).dt.date if "ds" in pred_df.columns else None
    # If ds not available, reconstruct from hour_business
    if "business_day" not in pred_df.columns or pred_df["business_day"].isna().all():
        # Approximate: every 24 rows is a day
        pred_df["_day_idx"] = pred_df.index // 24
        pred_df["business_day"] = pred_df.groupby("_day_idx").ngroup()

    day_smapes = pred_df.groupby("business_day").apply(
        lambda g: smape_floor50(g["rt_true"].values, g["rt_pred"].values)
    ).sort_values(ascending=False)
    worst_days = day_smapes.head(10).to_dict()

    # Gate analysis
    all_gates = np.concatenate(gate_values) if gate_values else np.array([])
    gate_mean = float(all_gates.mean()) if len(all_gates) > 0 else None

    # Confidence analysis
    all_conf = np.concatenate(conf_values) if conf_values else np.array([])
    errors = np.abs(pred_df["rt_pred"].values - pred_df["rt_true"].values)
    conf_corr = float(np.corrcoef(all_conf, -errors)[0, 1]) if len(errors) > 1 and len(all_conf) > 1 else None

    # ── Diagnoses ───────────────────────────────────────────────────
    diagnoses = []
    if corr_with_anchor > 0.98 and pred_da_delta < 5:
        diagnoses.append("COLLAPSE_TO_ANCHOR")
    if corr_with_sgdfnet > 0.98 and residual_pred_std < 1:
        diagnoses.append("COPY_SGDFNET")
    if residual_pred_std < residual_true_std * 0.1:
        diagnoses.append("NO_RESIDUAL_SIGNAL")
    if gate_mean is not None and (gate_mean < 0.05 or gate_mean > 0.95):
        diagnoses.append(f"GATE_COLLAPSE (mean={gate_mean:.3f})")
    if pred_var < target_var * 0.5:
        diagnoses.append("LOW_PREDICTION_VARIANCE")
    if overall_smape >= anchor_smape - 0.5:
        diagnoses.append("NO_BETTER_THAN_ANCHOR")
    diagnoses = diagnoses or ["NORMAL"]

    # ── Save outputs ────────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_predictions": len(pred_df),
        "overall_smape_floor50": float(overall_smape),
        "anchor_smape_floor50": float(anchor_smape),
        "sgdfnet_smape_floor50": float(sgdfnet_smape),
        "corr_with_da_anchor": float(corr_with_anchor),
        "corr_with_sgdfnet": float(corr_with_sgdfnet),
        "pred_da_delta_mean": float(pred_da_delta),
        "prediction_variance": float(pred_var),
        "target_variance": float(target_var),
        "variance_ratio": float(pred_var / target_var) if target_var > 0 else None,
        "residual_true_std": float(residual_true_std),
        "residual_pred_std": float(residual_pred_std),
        "residual_std_ratio": float(residual_pred_std / residual_true_std) if residual_true_std > 0 else None,
        "gate_mean": gate_mean,
        "confidence_error_corr": conf_corr,
        "diagnoses": diagnoses,
        "period_metrics": period_metrics,
        "best_val_smape": float(manifest.get("best_val_smape", 0)),
        "input_dim": config.input_dim,
        "model_backbone": config.backbone,
    }
    (out_dir / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save prediction distribution
    pred_df.to_csv(out_dir / "prediction_distribution.csv", index=False)

    # Hourly errors
    hourly = pred_df.groupby("hour_business").agg(
        smape=("rt_true", lambda x: smape_floor50(x.values, pred_df.loc[x.index, "rt_pred"].values)),
        count=("rt_true", "count"),
    ).reset_index()
    hourly.to_csv(out_dir / "error_by_hour.csv", index=False)

    # Period errors
    period_rows = []
    for name, (lo, hi) in periods.items():
        m = pred_df["hour_business"].between(lo, hi)
        period_rows.append({
            "period": name,
            "smape": smape_floor50(pred_df.loc[m, "rt_true"].values, pred_df.loc[m, "rt_pred"].values),
            "anchor_smape": smape_floor50(pred_df.loc[m, "rt_true"].values, pred_df.loc[m, "da_anchor"].values),
            "count": int(m.sum()),
        })
    pd.DataFrame(period_rows).to_csv(out_dir / "error_by_period.csv", index=False)

    # Bucket errors
    pd.DataFrame(bucket_metrics).to_csv(out_dir / "error_by_bucket.csv", index=False)

    # Worst days
    pd.DataFrame(list(worst_days.items()), columns=["day", "smape"]).to_csv(
        out_dir / "worst_days.csv", index=False
    )

    # Anchor comparison
    pred_df["trend_vs_anchor"] = pred_df["rt_pred"] - pred_df["da_anchor"]
    pred_df["trend_vs_sgdfnet"] = pred_df["rt_pred"] - pred_df["sgdfnet_pred"]
    comp = pred_df[["rt_true", "rt_pred", "da_anchor", "sgdfnet_pred",
                    "trend_vs_anchor", "trend_vs_sgdfnet", "hour_business"]].head(500)
    comp.to_csv(out_dir / "anchor_comparison.csv", index=False)

    # Gate analysis
    gate_df = pd.DataFrame({"gate": all_gates}) if len(all_gates) > 0 else pd.DataFrame()
    if not gate_df.empty:
        gate_df.to_csv(out_dir / "gate_analysis.csv", index=False)

    # Residual distribution
    resid_df = pd.DataFrame({
        "residual_true": residual_true,
        "residual_pred": residual_pred,
        "residual_error": residual_true - residual_pred,
    })
    resid_df.to_csv(out_dir / "residual_distribution.csv", index=False)

    # ── Diagnosis report ────────────────────────────────────────────
    lines = [
        "# TrendKnightRT Failure Diagnosis Report",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary",
        f"- **Overall sMAPE**: {overall_smape:.4f}",
        f"- **Anchor sMAPE**: {anchor_smape:.4f}",
        f"- **SGDFNet sMAPE**: {sgdfnet_smape:.4f}",
        f"- **Corr with DA anchor**: {corr_with_anchor:.4f}",
        f"- **Corr with SGDFNet**: {corr_with_sgdfnet:.4f}",
        "",
        "## Diagnoses",
    ]
    for d in diagnoses:
        lines.append(f"- **{d}**")

    lines += [
        "",
        "## Prediction Behavior",
        f"- **Pred - DA anchor (mean abs)**: {pred_da_delta:.2f}",
        f"- **Prediction variance**: {pred_var:.2f} (target: {target_var:.2f}, ratio: {pred_var/target_var:.2%})",
        f"- **Residual true std**: {residual_true_std:.2f}",
        f"- **Residual pred std**: {residual_pred_std:.2f}",
        f"- **Residual std ratio**: {residual_pred_std/residual_true_std:.2%}" if residual_true_std > 0 else "",
        f"- **Gate mean**: {gate_mean}" if gate_mean else "- **Gate**: N/A (not tracked)",
        "",
        "## Period Metrics",
    ]
    for name, m in period_metrics.items():
        lines.append(f"- **{name}**: model={m['smape']:.2f}%, anchor={m['anchor_smape']:.2f}%")

    lines += [
        "",
        "## Hour Metrics (worst 5)",
    ]
    worst_hours = sorted(hour_metrics.items(), key=lambda x: x[1]["smape"], reverse=True)[:5]
    for h, m in worst_hours:
        lines.append(f"- **{h}**: {m['smape']:.2f}% (n={m['count']})")

    lines += [
        "",
        "## Bucket Metrics",
    ]
    for b in bucket_metrics[:5]:
        lines.append(f"- **{b.get('bucket', '?')}**: smape={b.get('smape', 0):.2f}%")

    lines += [
        "",
        "## Gate Analysis",
    ]
    if gate_mean is not None:
        lines.append(f"- **Gate mean**: {gate_mean:.4f}")
        lines.append(f"- **Gate std**: {float(np.std(all_gates)):.4f}" if len(all_gates) > 1 else "")
    else:
        lines.append("- Gate not tracked in model output")

    lines += [
        "",
        "## Confidence Analysis",
    ]
    if conf_corr is not None:
        lines.append(f"- **Corr(-error, confidence)**: {conf_corr:.4f}")
        if conf_corr < -0.3:
            lines.append("- Confidence has useful negative correlation with error")
        else:
            lines.append("- Confidence is NOT correlated with error")
    else:
        lines.append("- Confidence not available")

    (out_dir / "diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ── Print summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  TRENDKNIGHTRT FAILURE DIAGNOSIS")
    print("=" * 60)
    print(f"  Overall sMAPE:       {overall_smape:.4f}")
    print(f"  Anchor sMAPE:        {anchor_smape:.4f}")
    print(f"  SGDFNet sMAPE:       {sgdfnet_smape:.4f}")
    print(f"  Corr w/ anchor:      {corr_with_anchor:.4f}")
    print(f"  Corr w/ SGDFNet:     {corr_with_sgdfnet:.4f}")
    print(f"  Residual std ratio:  {residual_pred_std/residual_true_std:.2%}" if residual_true_std > 0 else "")
    print(f"  Diagnoses:           {', '.join(diagnoses)}")
    print("=" * 60)


def _compute_bucket_metrics(df: pd.DataFrame) -> list[dict]:
    """Compute metrics by price bucket."""
    buckets = {
        "normal": (df["rt_true"] >= 0) & (df["rt_true"] < 500),
        "negative": df["rt_true"] < 0,
        "spike": df["rt_true"] >= 500,
    }
    results = []
    for name, mask in buckets.items():
        if mask.sum() > 0:
            results.append({
                "bucket": name,
                "smape": float(smape_floor50(
                    df.loc[mask, "rt_true"].values,
                    df.loc[mask, "rt_pred"].values,
                )),
                "count": int(mask.sum()),
            })
    return results


if __name__ == "__main__":
    main()
