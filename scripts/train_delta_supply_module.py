#!/usr/bin/env python
"""Train DeltaSupply deviation risk module.

Usage:
    python scripts/train_delta_supply_module.py \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --target-month 2026-02 \
        --out-dir artifacts/delta_supply/exp_2026_02
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

from models.deep_sgdf_delta.delta_supply_targets import (
    DeltaSupplyThresholds,
    compute_delta_supply_targets,
)
from models.deep_sgdf_delta.delta_supply_features import (
    build_delta_supply_features,
)
from models.deep_sgdf_delta.delta_supply_model import (
    DeltaSupplyConfig,
    DeltaSupplyModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train DeltaSupply module")
    p.add_argument("--data-path", required=True, help="Path to shandong_pmos_hourly.csv")
    p.add_argument("--target-month", required=True, help="Target month YYYY-MM")
    p.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--threshold-upward", type=float, default=100.0)
    p.add_argument("--threshold-downward", type=float, default=-100.0)
    p.add_argument("--threshold-abs-large", type=float, default=150.0)
    p.add_argument("--clip", type=float, default=500.0)
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
    thresholds = DeltaSupplyThresholds(
        upward=args.threshold_upward,
        downward=args.threshold_downward,
        abs_large=args.threshold_abs_large,
        clip=args.clip,
    )
    target_result = compute_delta_supply_targets(raw_df, thresholds=thresholds)
    logger.info("Targets computed: %d rows, %d valid", target_result.n_rows, target_result.n_valid)
    logger.info("Upward rate: %.3f, Downward rate: %.3f, Large abs rate: %.3f",
                target_result.upward_rate, target_result.downward_rate, target_result.large_abs_rate)

    # Step 2: Build features
    feature_result = build_delta_supply_features(raw_df, mode=args.mode)
    logger.info("Features built: %d features, verdict=%s",
                feature_result.audit.n_features, feature_result.audit.verdict)
    logger.info("Forecast coverage: %.2f, Lag coverage: %.2f, Calendar coverage: %.2f",
                feature_result.audit.forecast_feature_coverage,
                feature_result.audit.lag_feature_coverage,
                feature_result.audit.calendar_feature_coverage)

    # Check feature audit
    if feature_result.audit.verdict == "NOT_READY":
        logger.warning("Feature audit is NOT_READY. Will still train but metrics are experimental.")

    # Step 3: Merge targets and features
    work = feature_result.df.copy()

    # Merge target columns
    target_cols = ["price_delta", "upward_deviation_label", "downward_deviation_label",
                   "large_abs_deviation_label", "deviation_magnitude_target"]
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
    y_up_train = work.loc[train_mask, "upward_deviation_label"].values
    y_down_train = work.loc[train_mask, "downward_deviation_label"].values
    y_large_train = work.loc[train_mask, "large_abs_deviation_label"].values
    y_mag_train = work.loc[train_mask, "deviation_magnitude_target"].values

    # Fill NaN in features with 0 for model training
    X_train = X_train.fillna(0)

    # Step 5: Train model
    config = DeltaSupplyConfig()
    model = DeltaSupplyModel(config)
    model.fit(X_train, y_up_train, y_down_train, y_large_train, y_mag_train)
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
    pred_df["price_delta"] = work.loc[test_mask, "price_delta"].values
    pred_df["upward_deviation_label"] = work.loc[test_mask, "upward_deviation_label"].values
    pred_df["downward_deviation_label"] = work.loc[test_mask, "downward_deviation_label"].values
    pred_df["large_abs_deviation_label"] = work.loc[test_mask, "large_abs_deviation_label"].values
    pred_df["deviation_magnitude_target"] = work.loc[test_mask, "deviation_magnitude_target"].values

    # Also predict on full dataset for comprehensive output
    X_all = work[feature_cols].fillna(0)
    all_pred = model.predict(X_all)
    all_pred_df = all_pred.df.copy()
    all_pred_df["ds"] = work["ds"].values
    all_pred_df["da_anchor"] = work["da_anchor"].values if "da_anchor" in work.columns else np.nan
    all_pred_df["rt_actual"] = work["rt_actual"].values if "rt_actual" in work.columns else np.nan
    all_pred_df["price_delta"] = work["price_delta"].values if "price_delta" in work.columns else np.nan

    # Step 7: Save outputs
    # model.pkl (using joblib via sklearn)
    import pickle
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", model_path)

    # config.yaml
    config_dict = {
        "thresholds": {
            "upward": args.threshold_upward,
            "downward": args.threshold_downward,
            "abs_large": args.threshold_abs_large,
            "clip": args.clip,
        },
        "mode": args.mode,
        "prob_threshold": config.prob_threshold,
        "hgb_max_iter": config.hgb_max_iter,
        "hgb_max_depth": config.hgb_max_depth,
        "hgb_learning_rate": config.hgb_learning_rate,
        "ridge_alpha": config.ridge_alpha,
        "magnitude_scale": config.magnitude_scale,
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # feature_manifest.json
    feature_manifest = {
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "missing_features": feature_result.audit.missing_features,
        "forecast_feature_coverage": feature_result.audit.forecast_feature_coverage,
        "lag_feature_coverage": feature_result.audit.lag_feature_coverage,
        "calendar_feature_coverage": feature_result.audit.calendar_feature_coverage,
        "sgdfnet_available": feature_result.audit.sgdfnet_available,
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
            "upward": args.threshold_upward,
            "downward": args.threshold_downward,
            "abs_large": args.threshold_abs_large,
            "clip": args.clip,
        },
        "upward_rate_train": float(y_up_train[y_up_train >= 0].mean()) if (y_up_train >= 0).any() else 0,
        "downward_rate_train": float(y_down_train[y_down_train >= 0].mean()) if (y_down_train >= 0).any() else 0,
        "large_abs_rate_train": float(y_large_train[y_large_train >= 0].mean()) if (y_large_train >= 0).any() else 0,
        "mean_delta_train": float(target_result.mean_delta),
        "std_delta_train": float(target_result.std_delta),
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

    # metrics_summary.json (basic training metrics)
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        mean_absolute_error, mean_squared_error,
    )

    # Test set metrics
    test_labels_up = work.loc[test_mask, "upward_deviation_label"].values
    test_labels_down = work.loc[test_mask, "downward_deviation_label"].values
    test_labels_large = work.loc[test_mask, "large_abs_deviation_label"].values
    test_mag = work.loc[test_mask, "deviation_magnitude_target"].values

    valid_up = test_labels_up >= 0
    valid_down = test_labels_down >= 0
    valid_large = test_labels_large >= 0
    valid_mag = ~np.isnan(test_mag)

    metrics = {}
    if valid_up.sum() > 0 and len(np.unique(test_labels_up[valid_up])) > 1:
        up_pred = (pred_df.loc[valid_up, "upward_deviation_prob"].values >= 0.5).astype(int)
        metrics["upward"] = {
            "precision": float(precision_score(test_labels_up[valid_up], up_pred, zero_division=0)),
            "recall": float(recall_score(test_labels_up[valid_up], up_pred, zero_division=0)),
            "f1": float(f1_score(test_labels_up[valid_up], up_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_labels_up[valid_up], pred_df.loc[valid_up, "upward_deviation_prob"].values)),
        }

    if valid_down.sum() > 0 and len(np.unique(test_labels_down[valid_down])) > 1:
        down_pred = (pred_df.loc[valid_down, "downward_deviation_prob"].values >= 0.5).astype(int)
        metrics["downward"] = {
            "precision": float(precision_score(test_labels_down[valid_down], down_pred, zero_division=0)),
            "recall": float(recall_score(test_labels_down[valid_down], down_pred, zero_division=0)),
            "f1": float(f1_score(test_labels_down[valid_down], down_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_labels_down[valid_down], pred_df.loc[valid_down, "downward_deviation_prob"].values)),
        }

    if valid_large.sum() > 0 and len(np.unique(test_labels_large[valid_large])) > 1:
        large_pred = (pred_df.loc[valid_large, "large_abs_deviation_prob"].values >= 0.5).astype(int)
        metrics["large_abs"] = {
            "precision": float(precision_score(test_labels_large[valid_large], large_pred, zero_division=0)),
            "recall": float(recall_score(test_labels_large[valid_large], large_pred, zero_division=0)),
            "f1": float(f1_score(test_labels_large[valid_large], large_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_labels_large[valid_large], pred_df.loc[valid_large, "large_abs_deviation_prob"].values)),
        }

    if valid_mag.sum() > 0:
        mag_pred = pred_df.loc[valid_mag, "deviation_magnitude_pred"].values
        metrics["magnitude"] = {
            "mae": float(mean_absolute_error(test_mag[valid_mag], mag_pred)),
            "rmse": float(np.sqrt(mean_squared_error(test_mag[valid_mag], mag_pred))),
        }

    # DA anchor sMAPE on test set
    da_vals = work.loc[test_mask, "da_anchor"].values
    rt_vals = work.loc[test_mask, "rt_actual"].values
    valid_price = ~(np.isnan(da_vals) | np.isnan(rt_vals))
    if valid_price.sum() > 0:
        da_v = da_vals[valid_price]
        rt_v = rt_vals[valid_price]
        smape = np.mean(2 * np.abs(rt_v - da_v) / (np.maximum(np.abs(rt_v), 50) + np.maximum(np.abs(da_v), 50)))
        metrics["da_anchor_smape_floor50"] = float(smape)

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Training complete. All outputs saved to %s", out_dir)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
