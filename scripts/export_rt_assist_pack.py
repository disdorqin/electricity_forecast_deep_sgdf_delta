"""
Export RT-Assist model pack.

Trains residual model (RandomForest) on historical data
and exports to exported_models/rt_assist_pack/.

Usage:
    python scripts/export_rt_assist_pack.py \
        --data-path data/preprocessed_data.csv \
        --output-dir exported_models/rt_assist_pack \
        --train-end 2025-12-31
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export RT-Assist model pack")
    parser.add_argument("--data-path", type=str, required=True, help="Path to preprocessed data CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for model pack")
    parser.add_argument("--train-end", type=str, default="2025-12-31", help="Train on data up to this date")
    parser.add_argument("--alpha", type=float, default=1.0, help="Correction alpha (default: 1.0)")
    parser.add_argument("--clip", type=float, default=0.0, help="Correction clip (default: 0 = no clip)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_path, parse_dates=["times"])
    df = df.sort_values("times").reset_index(drop=True)
    logger.info(f"Data loaded: {len(df)} rows, {df['times'].min()} to {df['times'].max()}")

    # Filter to training data (up to train-end)
    train_df = df[df["times"] <= pd.Timestamp(args.train_end)].copy()
    logger.info(f"Training data: {len(train_df)} rows (up to {args.train_end})")

    # Add business_time columns
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    train_df = add_business_time_columns(train_df, timestamp_col="times")

    # Add residual column
    train_df["residual"] = train_df["rt_price"] - train_df["da_price"]

    # Feature engineering (matching Phase 2-5)
    train_df = _add_features(train_df)

    # Feature columns
    feature_cols = _get_feature_columns()
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

    # Drop NaN
    train_valid = train_df.dropna(subset=feature_cols + ["residual"]).reset_index(drop=True)
    logger.info(f"Training samples after dropna: {len(train_valid)}")

    if len(train_valid) < 1000:
        logger.error(f"Not enough training samples ({len(train_valid)})!")
        return

    # Train residual model
    from sklearn.ensemble import RandomForestRegressor
    X = train_valid[feature_cols].values.astype(float)
    y = train_valid["residual"].values.astype(float)

    logger.info(f"Training RandomForest on {len(X)} samples...")
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)
    logger.info("Training done.")

    # Save model pack
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save residual model
    import pickle
    with open(output_path / "residual_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Residual model saved to {output_path / 'residual_model.pkl'}")

    # Save manifest
    import json
    from datetime import datetime
    manifest = {
        "model_version": "RT-Assist-1",
        "export_date": datetime.now().isoformat(),
        "train_end": args.train_end,
        "n_train_samples": len(train_valid),
        "feature_columns": feature_cols,
        "alpha": args.alpha,
        "clip": args.clip,
        "enable_safe_correction": True,
        "model_type": "RandomForestRegressor",
        "notes": "RT-Assist-1: residual regression with alpha=1.0, no clip.",
    }
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest saved to {output_path / 'manifest.json'}")

    # Save feature columns separately (for prediction)
    with open(output_path / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Feature columns saved to {output_path / 'feature_columns.json'}")

    logger.info(f"\n✅ Model pack exported to {args.output_dir}")
    logger.info(f"   To predict: python scripts/predict_rt_assist_pack.py --model-dir {args.output_dir} ...")


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features matching Phase 2-5."""
    df = df.copy()

    # Calendar features
    df["hour"] = df["times"].dt.hour + 1
    df["is_weekend"] = (df["times"].dt.dayofweek >= 5).astype(int)
    df["month"] = df["times"].dt.month

    # Period buckets
    df["period"] = pd.cut(df["hour"], bins=[0, 8, 16, 24], labels=["p1_8", "p9_16", "p17_24"], include_lowest=True)

    # Bucket features
    df["da_price_level"] = pd.cut(df["da_price"], bins=[-np.inf, 0, 100, 500, np.inf], labels=["negative", "low", "mid", "high"])
    df["abs_residual_bucket"] = pd.cut(np.abs(df["residual"]), bins=[0, 50, 150, 500, np.inf], labels=["small", "medium", "large", "extreme"])

    # Encode categorical as numeric codes
    cat_mappings = {
        "da_price_level": {"negative": 0, "low": 1, "mid": 2, "high": 3},
        "abs_residual_bucket": {"small": 0, "medium": 1, "large": 2, "extreme": 3},
        "period": {"p1_8": 0, "p9_16": 1, "p17_24": 2},
    }
    for col, mapping in cat_mappings.items():
        if col in df.columns:
            code_col = col + "_code"
            df[code_col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

    # Lags
    for lag in [24, 48, 72, 168]:
        df[f"da_lag_{lag}h"] = df["da_price"].shift(lag)
        df[f"rt_lag_{lag}h"] = df["rt_price"].shift(lag)

    # Rolling
    for window in [24, 48, 168]:
        df[f"da_roll_mean_{window}h"] = df["da_price"].rolling(window, min_periods=1).mean()
        df[f"rt_roll_mean_{window}h"] = df["rt_price"].rolling(window, min_periods=1).mean()

    return df


def _get_feature_columns() -> list:
    """Feature columns (matching test_2025_full_year.py)."""
    return [
        "da_price", "hour", "is_weekend", "month",
        "da_price_level_code", "abs_residual_bucket_code", "period_code",
        "da_lag_24h", "da_lag_48h", "da_lag_72h", "da_lag_168h",
        "rt_lag_24h", "rt_lag_48h", "rt_lag_72h", "rt_lag_168h",
        "da_roll_mean_24h", "da_roll_mean_48h", "da_roll_mean_168h",
        "rt_roll_mean_24h", "rt_roll_mean_48h", "rt_roll_mean_168h",
    ]


if __name__ == "__main__":
    main()
