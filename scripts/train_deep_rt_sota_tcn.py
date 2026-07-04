"""Improved training script for DeepRT-SOTA v2 with TCN model.

This script:
1. Uses TCN model for sequence modeling
2. Implements day-level prediction (24-hour vector output)
3. Uses business_time.py for proper time alignment
4. Trains on real data and evaluates on 2026-02

Usage:
    conda run -n epf-2 python scripts/train_deep_rt_sota_tcn.py
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
    """Load and preprocess real data."""
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path, encoding='gbk')
    logger.info(f"Loaded {len(df)} rows")

    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })

    df['ds'] = pd.to_datetime(df['ds'])
    df = add_business_time_columns(df, timestamp_col='ds')

    logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    logger.info(f"Unique days: {df['business_day'].nunique()}")

    return df


def create_day_level_dataset(df: pd.DataFrame, seq_len_days: int = 14, feature_cols: list = None):
    """Create day-level dataset for sequence models.

    Args:
        df: Preprocessed DataFrame (with features already built).
        seq_len_days: Number of past days to use as sequence.
        feature_cols: List of feature columns.

    Returns:
        X_seq, y arrays.
    """
    if feature_cols is None:
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

    if len(available_features) == 0:
        raise ValueError("No features available! Make sure to build features before creating dataset.")

    # Create day-level samples
    X = []
    y = []

    unique_days = sorted(df['business_day'].unique())

    for i, day in enumerate(unique_days):
        if i < seq_len_days:
            continue

        # Get target day data
        day_idx = df['business_day'] == day
        day_data = df[day_idx].sort_values('hour_business')

        if len(day_data) != 24:
            continue

        # Get sequence (past seq_len_days)
        seq_start_idx = i - seq_len_days
        seq_end_idx = i

        seq_features = []
        valid = True

        for seq_day_idx in range(seq_start_idx, seq_end_idx):
            seq_day = unique_days[seq_day_idx]
            seq_day_data = df[df['business_day'] == seq_day].sort_values('hour_business')

            if len(seq_day_data) != 24:
                valid = False
                break

            seq_features.append(seq_day_data[available_features].values)

        if not valid or len(seq_features) != seq_len_days:
            continue

        # X_seq: (seq_len_days * 24, n_features)
        X_day = np.array(seq_features).reshape(-1, len(available_features))
        X.append(X_day)

        # y: (24,)
        y.append(day_data['rt_actual'].values)

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Created day-level dataset: X shape {X.shape}, y shape {y.shape}")

    return X, y, available_features


def train_sequence_model(X_train, y_train, config: DeepRTSOTAModelConfig, epochs: int = 80):
    """Train sequence model (TCN/GRU/Transformer).

    Args:
        X_train: Training sequences (n_samples, seq_len * 24, n_features).
        y_train: Training targets (n_samples, 24).
        config: Model configuration.
        epochs: Number of epochs.

    Returns:
        Trained model.
    """
    # Create model
    config.n_features = X_train.shape[2]
    config.output_dim = 24  # 24-hour prediction
    model = DeepRTSOTAModel(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info(f"Training on {device}")
    logger.info(f"Model: {config.model_profile}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    for epoch in range(epochs):
        model.train()

        pred, _ = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def evaluate_sequence_model(model, X_test, y_test, config):
    """Evaluate sequence model using canonical smape_floor50."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        pred, _ = model(X_test_tensor)
        pred = pred.cpu().numpy()

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
    smape_floor50 = float(np.mean(200.0 * np.abs(yp - yt) / denom))

    return {'mae': mae, 'rmse': rmse, 'smape': smape_floor50}


def main():
    """Main training function."""
    logger.info("Starting DeepRT-SOTA v2 TCN training...")

    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    df = load_and_preprocess_data(data_path)

    # Split data
    target_month = '2026-02'
    test_mask = (df['business_day'].dt.year == 2026) & (df['business_day'].dt.month == 2)
    test_df = df[test_mask].copy()

    train_val_df = df[~test_mask].copy()
    train_val_dates = sorted(train_val_df['business_day'].unique())
    val_start = train_val_dates[-30]

    val_df = train_val_df[train_val_df['business_day'] >= val_start].copy()
    train_df = train_val_df[train_val_df['business_day'] < val_start].copy()

    logger.info(f"Train: {train_df['business_day'].nunique()} days")
    logger.info(f"Val: {val_df['business_day'].nunique()} days")
    logger.info(f"Test: {test_df['business_day'].nunique()} days")

    # Build features BEFORE creating datasets
    # CRITICAL FIX: Merge train+test before building features to avoid NaN in test set
    logger.info("Building features (merged train+test to avoid NaN in test)...")
    
    # Load synthetic risk features if available
    risk_csv = Path('artifacts/deep_rt_sota/synthetic_risk_features.csv')
    risk_df = None
    if risk_csv.exists():
        logger.info(f"Loading risk features from {risk_csv}...")
        risk_df = pd.read_csv(risk_csv)
        risk_df['ds'] = pd.to_datetime(risk_df['ds'])
        risk_df = add_business_time_columns(risk_df, timestamp_col='ds')
    
    # Merge train and test for feature building
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f"  Merged: {len(merged_df)} rows")
    
    # Build features on merged data
    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df, risk_features=True, forecast_features=False, risk_df=risk_df
    )
    logger.info(f"Feature manifest: {feature_manifest}")
    
    # Split back to train and test
    train_df = merged_df[merged_df['business_day'] < target_month].copy()
    test_df = merged_df[(merged_df['business_day'] >= target_month) & (merged_df['business_day'] < '2026-03')].copy()
    
    logger.info(f"  Split back: train={len(train_df)}, test={len(test_df)}")

    # Drop rows with NaN features
    feature_cols = [
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'previous_day_rt_mean', 'previous_day_rt_std',
        'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
        'da_anchor', 'forecast_price', 'anchor_spread',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'period_id',
    ]
    available_features = [col for col in feature_cols if col in train_df.columns]

    logger.info(f"Dropping rows with NaN features...")
    logger.info(f"  Train before: {len(train_df)}")
    train_df = train_df.dropna(subset=available_features)
    logger.info(f"  Train after: {len(train_df)}")

    logger.info(f"  Test before: {len(test_df)}")
    test_df = test_df.dropna(subset=available_features)
    logger.info(f"  Test after: {len(test_df)}")

    # Create datasets
    logger.info("Creating day-level datasets...")
    feature_cols = [
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'previous_day_rt_mean', 'previous_day_rt_std',
        'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
        'da_anchor', 'forecast_price', 'anchor_spread',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'period_id',
    ]

    X_train, y_train, _ = create_day_level_dataset(train_df, seq_len_days=7, feature_cols=feature_cols)
    X_test, y_test, _ = create_day_level_dataset(test_df, seq_len_days=7, feature_cols=feature_cols)

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train TCN model
    config = DeepRTSOTAModelConfig(
        model_profile='deep_rt_tcn',
        seq_len_days=14,
        target_mode='direct',
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        output_dim=24,
    )

    logger.info("Training TCN model...")
    model = train_sequence_model(X_train, y_train, config, epochs=80)

    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_sequence_model(model, X_test, y_test, config)

    logger.info("=" * 50)
    logger.info("Results (TCN, day-level prediction):")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  sMAPE: {metrics['smape']:.4f}")
    logger.info("=" * 50)

    # Save results
    results = {
        'model_profile': config.model_profile,
        'target_month': target_month,
        'seq_len_days': config.seq_len_days,
        'metrics': {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'smape': float(metrics['smape']),
        },
        'n_features': X_train.shape[2],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
    }

    out_path = Path('artifacts/deep_rt_sota/tcn_exp')
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {out_path / 'results.json'}")

    return metrics


if __name__ == '__main__':
    main()
