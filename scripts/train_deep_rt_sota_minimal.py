"""Minimal working training script for DeepRT-SOTA v2.

This script:
1. Loads real data
2. Builds features
3. Trains a simple model
4. Evaluates on test set
5. Reports metrics

Usage:
    conda run -n epf-2 python scripts/train_deep_rt_sota_minimal.py
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features
from models.deep_sgdf_delta.deep_rt_sota_model import DeepRTSOTAModel, DeepRTSOTAModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess real data.

    Args:
        data_path: Path to CSV file.

    Returns:
        Preprocessed DataFrame.
    """
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path, encoding='gbk')
    logger.info(f"Loaded {len(df)} rows")

    # Rename Chinese columns to English
    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })

    # Convert ds to datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Add business time columns
    df = add_business_time_columns(df, timestamp_col='ds')

    logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    logger.info(f"Unique days: {df['business_day'].nunique()}")

    return df


def create_simple_dataset(df: pd.DataFrame, seq_len_days: int = 7):
    """Create a simple dataset for training (hourly prediction).

    Args:
        df: Preprocessed DataFrame.
        seq_len_days: Sequence length in days (not used for MLP).

    Returns:
        X, y arrays.
    """
    # Build features
    df, feature_manifest = build_deep_rt_sota_features(
        df,
        risk_features=False,
        forecast_features=False,
    )

    # Get feature columns
    feature_cols = [
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'previous_day_rt_mean', 'previous_day_rt_std',
        'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
        'da_anchor', 'forecast_price', 'anchor_spread',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'period_id',
    ]

    # Check which features are available
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available features: {len(available_features)}")

    # Create hourly samples (not day-level)
    X = []
    y = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Skip if any feature is NaN
        if row[available_features].isna().any():
            continue

        # Skip if rt_actual is NaN
        if pd.isna(row['rt_actual']):
            continue

        X.append(row[available_features].values)
        y.append(row['rt_actual'])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Created dataset: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

    return X, y, available_features


def train_model(X_train, y_train, X_val, y_val, config: DeepRTSOTAModelConfig, epochs: int = 50):
    """Train model.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        config: Model configuration.
        epochs: Number of epochs.

    Returns:
        Trained model.
    """
    # Create model
    config.n_features = X_train.shape[1]  # For MLP: n_features = number of features
    model = DeepRTSOTAModel(config)

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info(f"Training on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loss
    criterion = nn.HuberLoss()

    # Convert to tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()

        # Forward pass
        if config.model_profile == "deep_rt_mlp":
            pred, _ = model(X_train_tensor)
        else:
            # Reshape for sequence models
            X_seq = X_train_tensor.reshape(X_train_tensor.shape[0], config.seq_len_days * 24, -1)
            pred, _ = model(X_seq)

        loss = criterion(pred, y_train_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def evaluate_model(model, X_test, y_test, config):
    """Evaluate model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
        config: Model configuration.

    Returns:
        Metrics dictionary.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        if config.model_profile == "deep_rt_mlp":
            pred, _ = model(X_test_tensor)
        else:
            X_seq = X_test_tensor.reshape(X_test_tensor.shape[0], config.seq_len_days * 24, -1)
            pred, _ = model(X_seq)

        pred = pred.cpu().numpy()

    # Compute metrics
    y_true_flat = y_test.flatten()
    y_pred_flat = pred.flatten()

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    # Canonical sMAPE_floor50 (MUST use this, not raw sMAPE)
    floor = 50.0
    eps = 1e-8
    yt = np.where(y_true_flat < floor, floor, y_true_flat)
    yp = np.where(y_pred_flat < floor, floor, y_pred_flat)
    denom = np.abs(yt) + np.abs(yp) + eps
    smape = float(np.mean(200.0 * np.abs(yp - yt) / denom))

    return {
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
    }


def main():
    """Main training function."""
    logger.info("Starting DeepRT-SOTA v2 minimal training...")

    # 1. Load data
    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    df = load_and_preprocess_data(data_path)

    # 2. Split data (target: 2026-02)
    target_month = '2026-02'
    target_year, target_month_num = 2026, 2

    test_mask = (
        (df['business_day'].dt.year == target_year) &
        (df['business_day'].dt.month == target_month_num)
    )
    test_df = df[test_mask].copy()

    train_val_df = df[~test_mask].copy()

    # Split train/val
    train_val_dates = sorted(train_val_df['business_day'].unique())
    val_start = train_val_dates[-30]

    val_df = train_val_df[train_val_df['business_day'] >= val_start].copy()
    train_df = train_val_df[train_val_df['business_day'] < val_start].copy()

    logger.info(f"Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    logger.info(f"Val: {len(val_df)} rows, {val_df['business_day'].nunique()} days")
    logger.info(f"Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

    # 3. Create datasets
    logger.info("Creating datasets...")
    X_train, y_train, feature_cols = create_simple_dataset(train_df)
    X_val, y_val, _ = create_simple_dataset(val_df)
    X_test, y_test, _ = create_simple_dataset(test_df)

    # 4. Create model config
    config = DeepRTSOTAModelConfig(
        model_profile='deep_rt_mlp',  # Start with simple MLP
        seq_len_days=7,
        target_mode='direct',
        n_features=len(feature_cols),
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        output_dim=1,  # Hourly prediction: output 1 value
    )

    # 5. Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train, X_val, y_val, config, epochs=50)

    # 6. Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, config)

    logger.info("=" * 50)
    logger.info("Results:")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  sMAPE: {metrics['smape']:.4f}")
    logger.info("=" * 50)

    # 7. Save results
    results = {
        'model_profile': config.model_profile,
        'target_mode': config.target_mode,
        'target_month': target_month,
        'metrics': {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'smape': float(metrics['smape']),
        },
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
    }

    out_path = Path('artifacts/deep_rt_sota/minimal_exp')
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {out_path / 'results.json'}")

    return metrics


if __name__ == '__main__':
    main()
