#!/usr/bin/env python
"""Train NegativeRisk classification module.

Usage:
    python scripts/train_negative_risk_module.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-month 2026-02 \
        --out-dir artifacts/negative_risk/exp_2026_02
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.negative_risk_targets import (
    NegativeRiskThresholds,
    compute_negative_risk_targets,
)
from models.deep_sgdf_delta.negative_risk_features import (
    build_negative_risk_features,
)
from models.deep_sgdf_delta.negative_risk_model import (
    NegativeRiskConfig,
    NegativeRiskModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train NegativeRisk module")
    p.add_argument("--data-path", required=True, help="Path to shandong_pmos_hourly.csv")
    p.add_argument("--target-month", required=True, help="Target month YYYY-MM")
    p.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--threshold-negative", type=float, default=0.0)
    p.add_argument("--threshold-deep-negative", type=float, default=-100.0)
    p.add_argument("--threshold-relative-down", type=float, default=-200.0)
    p.add_argument("--mode", choices=["FULL_DAY", "INTRADAY"], default="FULL_DAY")
    p.add_argument("--fast-dev-run", action="store_true", help="Use small subset for debugging")
    return p.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV with encoding fallback."""
    for enc in ["utf-8", "gb18030", "gbk", "latin-1"]:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            logger.info("Loaded %d rows with encoding %s", len(df), enc)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot read {data_path} with any known encoding")


def split_train_test(
    df: pd.DataFrame,
    target_month: str,
    val_days: int = 30,
) -> tuple:
    """Split into train/val/test by target_month.

    train: all data before target_month minus val_days
    val: last val_days before target_month
    test: target_month
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    target_start = pd.Timestamp(target_month + "-01")
    target_end = target_start + pd.tseries.offsets.MonthEnd(1) + pd.Timedelta(days=1)

    test_mask = (df["ds"] >= target_start) & (df["ds"] < target_end)
    before_target = df["ds"] < target_start

    if val_days > 0:
        val_start = target_start - pd.Timedelta(days=val_days)
        train_mask = before_target & (df["ds"] < val_start)
        val_mask = before_target & (df["ds"] >= val_start)
    else:
        train_mask = before_target
        val_mask = pd.Series(False, index=df.index)

    return train_mask, val_mask, test_mask


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    raw_df = load_data(args.data_path)

    # Rename Chinese columns to English
    from models.deep_sgdf_delta.realtime_column_mapping import rename_chinese_columns
    raw_df = rename_chinese_columns(raw_df)

    # Date filtering
    if args.start_date:
        raw_df = raw_df[pd.to_datetime(raw_df.iloc[:, 0]) >= args.start_date]
    if args.end_date:
        raw_df = raw_df[pd.to_datetime(raw_df.iloc[:, 0]) <= args.end_date]

    if args.fast_dev_run:
        raw_df = raw_df.head(24 * 120)  # ~120 days
        logger.info("Fast dev run: using %d rows", len(raw_df))

    # Step 1: Compute targets
    thresholds = NegativeRiskThresholds(
        negative=args.threshold_negative,
        deep_negative=args.threshold_deep_negative,
        relative_down=args.threshold_relative_down,
    )
    target_result = compute_negative_risk_targets(raw_df, thresholds=thresholds)
    logger.info("Targets computed: %d rows, %d valid", target_result.n_rows, target_result.n_valid)
    logger.info("Negative rate: %.3f, Deep negative rate: %.3f, Relative down rate: %.3f",
                target_result.negative_rate, target_result.deep_negative_rate, target_result.relative_down_rate)

    # Step 2: Build features
    feature_result = build_negative_risk_features(raw_df, mode=args.mode)
    logger.info("Features built: %d features, verdict=%s",
                feature_result.audit.n_features, feature_result.audit.verdict)
    logger.info("Derived coverage: %.2f, Lag coverage: %.2f, Calendar coverage: %.2f",
                feature_result.audit.derived_feature_coverage,
                feature_result.audit.lag_feature_coverage,
                feature_result.audit.calendar_feature_coverage)

    # Check feature audit
    if feature_result.audit.verdict == "NOT_READY":
        logger.warning("Feature audit is NOT_READY. Will still train but metrics are experimental.")

    # Step 3: Merge targets and features
    work = feature_result.df.copy()

    # Merge target columns
    target_cols = ["negative_label", "deep_negative_label", "relative_down_label"]
    for col in target_cols:
        if col in target_result.df.columns and col not in work.columns:
            work[col] = target_result.df[col].values

    # Step 4: Train/test split
    train_mask, val_mask, test_mask = split_train_test(work, args.target_month)

    feature_cols = feature_result.feature_columns
    # Filter to columns that exist in work
    feature_cols = [c for c in feature_cols if c in work.columns]

    logger.info("Train: %d, Val: %d, Test: %d",
                train_mask.sum(), val_mask.sum(), test_mask.sum())

    if train_mask.sum() < 10:
        raise ValueError(f"Not enough training samples: {train_mask.sum()}")
    if test_mask.sum() < 10:
        raise ValueError(f"Not enough test samples: {test_mask.sum()}")

    # Prepare training data
    X_train = work.loc[train_mask, feature_cols].copy()
    y_negative_train = work.loc[train_mask, "negative_label"].values
    y_deep_negative_train = work.loc[train_mask, "deep_negative_label"].values
    y_relative_down_train = work.loc[train_mask, "relative_down_label"].values

    # Fill NaN in features with 0 for model training
    X_train = X_train.fillna(0)

    # Step 5: Train model
    config = NegativeRiskConfig()
    model = NegativeRiskModel(config)
    model.fit(X_train, y_negative_train, y_deep_negative_train, y_relative_down_train)
    logger.info("Model trained successfully.")

    # Step 6: Predict on test set
    X_test = work.loc[test_mask, feature_cols].copy().fillna(0)
    test_pred = model.predict(X_test)

    # Add actual labels and price info to predictions
    pred_df = test_pred.df.copy()
    pred_df["ds"] = work.loc[test_mask, "ds"].values
    pred_df["business_day"] = work.loc[test_mask, "business_day"].values
    pred_df["hour_business"] = work.loc[test_mask, "hour_business"].values
    pred_df["period"] = work.loc[test_mask, "period"].values
    pred_df["da_anchor"] = work.loc[test_mask, "da_anchor"].values
    pred_df["rt_actual"] = work.loc[test_mask, "rt_actual"].values
    pred_df["negative_label"] = work.loc[test_mask, "negative_label"].values
    pred_df["deep_negative_label"] = work.loc[test_mask, "deep_negative_label"].values
    pred_df["relative_down_label"] = work.loc[test_mask, "relative_down_label"].values

    # Step 7: Save outputs
    # model.pkl
    import pickle
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", model_path)

    # config.yaml
    config_dict = {
        "thresholds": {
            "negative": args.threshold_negative,
            "deep_negative": args.threshold_deep_negative,
            "relative_down": args.threshold_relative_down,
        },
        "mode": args.mode,
        "prob_threshold": config.prob_threshold,
        "hgb_max_iter": config.hgb_max_iter,
        "hgb_max_depth": config.hgb_max_depth,
        "hgb_learning_rate": config.hgb_learning_rate,
        "risk_score_weights": config.risk_score_weights,
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # feature_manifest.json
    feature_manifest = {
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "missing_features": feature_result.audit.missing_features,
        "derived_feature_coverage": feature_result.audit.derived_feature_coverage,
        "lag_feature_coverage": feature_result.audit.lag_feature_coverage,
        "calendar_feature_coverage": feature_result.audit.calendar_feature_coverage,
        "leakage_check": feature_result.audit.leakage_check,
        "verdict": feature_result.audit.verdict,
    }
    with open(out_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, ensure_ascii=False, indent=2)

    # train_manifest.json
    train_manifest = {
        "target_month": args.target_month,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "missing_features": feature_result.audit.missing_features,
        "feature_audit_verdict": feature_result.audit.verdict,
        "formal_metric": feature_result.audit.formal_ready,
        "thresholds": {
            "negative": args.threshold_negative,
            "deep_negative": args.threshold_deep_negative,
            "relative_down": args.threshold_relative_down,
        },
        "negative_rate_train": float(y_negative_train[y_negative_train >= 0].mean()) if (y_negative_train >= 0).any() else 0,
        "deep_negative_rate_train": float(y_deep_negative_train[y_deep_negative_train >= 0].mean()) if (y_deep_negative_train >= 0).any() else 0,
        "relative_down_rate_train": float(y_relative_down_train[y_relative_down_train >= 0].mean()) if (y_relative_down_train >= 0).any() else 0,
        "mean_rt_train": float(target_result.mean_rt),
        "std_rt_train": float(target_result.std_rt),
        "created_at": datetime.now().isoformat(),
        "mode": args.mode,
    }
    with open(out_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, ensure_ascii=False, indent=2)

    # predictions.csv (test set)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    logger.info("Saved predictions to %s (%d rows)", out_dir / "predictions.csv", len(pred_df))

    # feature_importance.csv
    fi = model.get_feature_importance()
    fi.to_csv(out_dir / "feature_importance.csv", index=False)
    logger.info("Saved feature importance to %s", out_dir / "feature_importance.csv")

    # metrics_summary.json (classification metrics + canonical sMAPE)
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
    )
    from models.deep_sgdf_delta.metrics import smape_floor50 as _smape_floor50

    # Test set metrics
    test_labels_negative = work.loc[test_mask, "negative_label"].values
    test_labels_deep = work.loc[test_mask, "deep_negative_label"].values
    test_labels_relative = work.loc[test_mask, "relative_down_label"].values

    valid_negative = test_labels_negative >= 0
    valid_deep = test_labels_deep >= 0
    valid_relative = test_labels_relative >= 0

    metrics = {}

    if valid_negative.sum() > 0 and len(np.unique(test_labels_negative[valid_negative])) > 1:
        neg_pred = (pred_df.loc[valid_negative, "negative_prob"].values >= 0.5).astype(int)
        metrics["negative"] = {
            "precision": float(precision_score(test_labels_negative[valid_negative], neg_pred, zero_division=0)),
            "recall": float(recall_score(test_labels_negative[valid_negative], neg_pred, zero_division=0)),
            "f1": float(f1_score(test_labels_negative[valid_negative], neg_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_labels_negative[valid_negative], pred_df.loc[valid_negative, "negative_prob"].values)),
        }

    if valid_deep.sum() > 0 and len(np.unique(test_labels_deep[valid_deep])) > 1:
        deep_pred = (pred_df.loc[valid_deep, "deep_negative_prob"].values >= 0.5).astype(int)
        metrics["deep_negative"] = {
            "precision": float(precision_score(test_labels_deep[valid_deep], deep_pred, zero_division=0)),
            "recall": float(recall_score(test_labels_deep[valid_deep], deep_pred, zero_division=0)),
            "f1": float(f1_score(test_labels_deep[valid_deep], deep_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_labels_deep[valid_deep], pred_df.loc[valid_deep, "deep_negative_prob"].values)),
        }

    if valid_relative.sum() > 0 and len(np.unique(test_labels_relative[valid_relative])) > 1:
        rel_pred = (pred_df.loc[valid_relative, "relative_down_prob"].values >= 0.5).astype(int)
        metrics["relative_down"] = {
            "precision": float(precision_score(test_labels_relative[valid_relative], rel_pred, zero_division=0)),
            "recall": float(recall_score(test_labels_relative[valid_relative], rel_pred, zero_division=0)),
            "f1": float(f1_score(test_labels_relative[valid_relative], rel_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_labels_relative[valid_relative], pred_df.loc[valid_relative, "relative_down_prob"].values)),
        }

    # DA anchor sMAPE on test set (using canonical formula)
    da_vals = work.loc[test_mask, "da_anchor"].values
    rt_vals = work.loc[test_mask, "rt_actual"].values
    valid_price = ~(np.isnan(da_vals) | np.isnan(rt_vals))
    if valid_price.sum() > 0:
        da_v = da_vals[valid_price]
        rt_v = rt_vals[valid_price]
        metrics["da_anchor_smape_floor50"] = _smape_floor50(rt_v, da_v)

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Training complete. All outputs saved to %s", out_dir)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
