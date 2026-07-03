"""Solar916 residual correction model.

Tree-based models for predicting SGDFNet residual in 9_16 hours.
Model priority:
  1. HistGradientBoostingRegressor
  2. CatBoostRegressor (if available)
  3. LGBMRegressor (if available)

Walk-forward training:
  - Train on data before target month
  - Validate on last 30 days of training period
  - Test on target month
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Solar916Config:
    """Configuration for Solar916 model."""
    model_type: str = "hist_gradient_boosting"  # or "catboost", "lightgbm"
    max_iter: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_samples_leaf: int = 20
    random_state: int = 42
    feature_columns: list[str] = field(default_factory=list)
    # For small datasets, use simpler model
    auto_simplify: bool = False


# ── Feature columns for the model ────────────────────────────────────

DEFAULT_FEATURE_COLS = [
    "hour_business",
    "weekday",
    "month",
    "da_anchor",
    "sgdfnet_pred",
    "forecast_load",
    "forecast_wind",
    "forecast_solar",
    "forecast_new_energy",
    "bidding_space",
    "net_load",
    "renewable_share",
    "delta_lag_24",
    "delta_lag_168",
    "residual_lag_24",
    "residual_lag_168",
    "rolling_residual_mean_7d",
    "rolling_residual_std_7d",
    "same_hour_residual_mean_7d",
    "same_hour_residual_std_7d",
]


def get_available_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return feature columns that exist in df and have non-NaN values."""
    available = []
    for col in feature_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                available.append(col)
            else:
                logger.warning("Feature %s has all NaN — skipping", col)
    return available


def build_model(config: Solar916Config):
    """Create a model instance based on config."""
    max_iter = config.max_iter
    max_depth = config.max_depth
    min_samples_leaf = config.min_samples_leaf
    learning_rate = config.learning_rate

    if config.auto_simplify:
        # Simpler model for small datasets to avoid overfitting
        max_iter = min(max_iter, 100)
        max_depth = min(max_depth, 3)
        min_samples_leaf = max(min_samples_leaf, 10)
        learning_rate = max(learning_rate, 0.1)
        logger.info("Auto-simplified model: max_iter=%d, max_depth=%d, min_samples_leaf=%d, lr=%f",
                     max_iter, max_depth, min_samples_leaf, learning_rate)

    if config.model_type == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=config.random_state,
        )
    elif config.model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=config.max_iter,
                learning_rate=config.learning_rate,
                depth=config.max_depth,
                random_seed=config.random_state,
                verbose=0,
            )
        except ImportError:
            logger.warning("CatBoost not available, falling back to HistGBR")
            return build_model(Solar916Config(model_type="hist_gradient_boosting"))
    elif config.model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=config.max_iter,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                min_child_samples=config.min_samples_leaf,
                random_state=config.random_state,
                verbose=-1,
            )
        except ImportError:
            logger.warning("LightGBM not available, falling back to HistGBR")
            return build_model(Solar916Config(model_type="hist_gradient_boosting"))
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 50.0) -> float:
    yt = np.clip(np.abs(y_true), floor, None)
    yp = np.clip(np.abs(y_pred), floor, None)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "sgdfnet_residual",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare X, y arrays from DataFrame, dropping NaN rows.

    Returns (X, y, actual_feature_cols_used).
    """
    available = get_available_features(df, feature_cols)
    subset = df[available + [target_col]].dropna()

    if len(subset) == 0:
        raise ValueError("No valid training rows after dropping NaN")

    X = subset[available].values.astype(np.float64)
    y = subset[target_col].values.astype(np.float64)

    logger.info("Training data: %d rows, %d features", X.shape[0], X.shape[1])
    return X, y, available


def train_walk_forward(
    dataset: pd.DataFrame,
    target_month: str,
    config: Solar916Config,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """Walk-forward training for a single target month.

    - Train: all data before target_month
    - Validate: last 30 days before target_month
    - Test: target_month

    Returns dict with model, predictions, metrics.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    dataset = dataset.copy()
    dataset["business_day"] = pd.to_datetime(dataset["business_day"])
    target_start = pd.Timestamp(target_month)
    target_end = target_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)

    # Split
    train_mask = dataset["business_day"] < target_start
    test_mask = (dataset["business_day"] >= target_start) & (dataset["business_day"] < target_end)

    train_df = dataset[train_mask].copy()
    test_df = dataset[test_mask].copy()

    if len(train_df) < 100:
        raise ValueError(f"Insufficient training data: {len(train_df)} rows")
    if len(test_df) == 0:
        raise ValueError(f"No test data for {target_month}")

    # Validation set: last 30 days of training
    val_start = train_df["business_day"].max() - pd.Timedelta(days=30)
    val_mask = train_df["business_day"] >= val_start
    train_proper_mask = ~val_mask

    # If train_proper would be too small (< 30 rows), reduce val window
    if train_proper_mask.sum() < 30:
        # Use 70/30 split of training data instead
        split_idx = int(len(train_df) * 0.7)
        train_proper_df = train_df.iloc[:split_idx].copy()
        val_df = train_df.iloc[split_idx:].copy()
        logger.info("  (Using 70/30 split — insufficient data for 30-day val window)")
    else:
        val_df = train_df[val_mask].copy()
        train_proper_df = train_df[train_proper_mask].copy()

    logger.info("Walk-forward split for %s:", target_month)
    logger.info("  Train: %d rows (before %s)", len(train_proper_df), val_start.date())
    logger.info("  Val:   %d rows (%s to %s)", len(val_df), val_start.date(), train_df["business_day"].max().date())
    logger.info("  Test:  %d rows (%s)", len(test_df), target_month)

    # Prepare features
    available = get_available_features(train_proper_df, feature_cols)
    config.feature_columns = available

    X_train, y_train, _ = prepare_training_data(train_proper_df, feature_cols)
    X_val, y_val, _ = prepare_training_data(val_df, feature_cols)
    X_test, y_test, actual_features = prepare_training_data(test_df, feature_cols)

    # Train
    model = build_model(config)
    model.fit(X_train, y_train)

    # Validate
    val_pred = model.predict(X_val)
    val_smape = smape_floor50(y_val, val_pred)
    logger.info("  Val residual sMAPE: %.4f", val_smape)

    # Test predictions
    test_pred = model.predict(X_test)

    # Feature importance
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        for i, fname in enumerate(actual_features):
            feature_importance[fname] = float(imp[i])
    elif hasattr(model, "get_feature_importance"):
        imp = model.get_feature_importance()
        for i, fname in enumerate(actual_features):
            feature_importance[fname] = float(imp[i])

    return {
        "model": model,
        "feature_columns": actual_features,
        "feature_importance": feature_importance,
        "train_rows": len(train_proper_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "val_smape": val_smape,
        "test_pred": test_pred,
        "test_actual": y_test,
        "test_df": test_df,
    }


def train_multi_month_walk_forward(
    dataset: pd.DataFrame,
    target_months: list[str],
    config: Solar916Config,
    feature_cols: Optional[list[str]] = None,
) -> list[dict]:
    """Train separate walk-forward models for each target month.

    Each month uses only data before that month for training.
    Returns list of results dicts (one per month).
    """
    results = []
    for month in target_months:
        logger.info("=" * 60)
        logger.info("Training for target month: %s", month)
        logger.info("=" * 60)
        try:
            result = train_walk_forward(dataset, month, config, feature_cols)
            result["target_month"] = month
            results.append(result)
        except ValueError as e:
            logger.warning("Skipping %s: %s", month, e)
            results.append({"target_month": month, "error": str(e)})
    return results
