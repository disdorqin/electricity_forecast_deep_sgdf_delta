"""
DeepRT-SOTA v2 - Day-level prediction (24-hour vector output).

Uses CORRECT smape_floor50 metric.
Implements day-level sequence modeling for better performance.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import logging

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features
from models.deep_sgdf_delta.deep_rt_sota_model import (
    DeepRTSOTAModel,
    DeepRTSOTAModelConfig,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def smape_floor50(y_true, y_pred, floor=50.0, eps=1e-8):
    """Canonical sMAPE_floor50."""
    yt = np.where(y_true < floor, floor, y_true)
    yp = np.where(y_pred < floor, floor, y_pred)
    denom = np.abs(yt) + np.abs(yp) + eps
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


def load_data(data_path):
    """Load and preprocess data."""
    df = pd.read_csv(data_path, encoding='gbk')
    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })
    df['ds'] = pd.to_datetime(df['ds'])
    df = add_business_time_columns(df, timestamp_col='ds')
    return df


def create_day_level_dataset(df, seq_len_days=14, risk_features=False):
    """Create day-level dataset for sequence models.
    
    Returns:
        X_seq: (n_samples, seq_len_days * 24, n_features)
        y: (n_samples, 24)
    """
    # Build features
    df, _ = build_deep_rt_sota_features(
        df,
        risk_features=risk_features,
        forecast_features=False,
    )
    
    feature_cols = [
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'previous_day_rt_mean', 'previous_day_rt_std',
        'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
        'da_anchor', 'anchor_spread',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'period_id',
    ]
    
    if risk_features:
        risk_cols = ['negative_risk_score', 'spike_risk_score']
        feature_cols.extend([col for col in risk_cols if col in df.columns])
    
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available features: {len(available_features)}")
    
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
        seq_features = []
        valid_seq = True
        
        for seq_offset in range(1, seq_len_days + 1):
            seq_day = unique_days[i - seq_offset]
            seq_day_data = df[df['business_day'] == seq_day].sort_values('hour_business')
            
            if len(seq_day_data) != 24:
                valid_seq = False
                break
            
            # Check for NaN
            if seq_day_data[available_features].isna().any().any():
                valid_seq = False
                break
            
            seq_features.append(seq_day_data[available_features].values)
        
        if not valid_seq:
            continue
        
        # Check if target day has valid rt_actual
        if day_data['rt_actual'].isna().any():
            continue
        
        # X_seq: (seq_len_days * 24, n_features)
        X_day = np.array(seq_features).reshape(-1, len(available_features))
        X.append(X_day)
        
        # y: (24,)
        y.append(day_data['rt_actual'].values)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    logger.info(f"Created day-level dataset: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, available_features


def train_and_evaluate_day_level(X_train, y_train, X_test, y_test, model_profile='deep_rt_tcn', seq_len_days=14):
    """Train and evaluate day-level model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standardize
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    
    # Create model
    config = DeepRTSOTAModelConfig(
        model_profile=model_profile,
        target_mode='direct',
        n_features=X_train.shape[2],
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        output_dim=24,
    )
    
    model = DeepRTSOTAModel(config).to(device)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        
        pred, _ = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/80], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        pred, _ = model(X_test_tensor)
        pred = pred.cpu().numpy()
    
    y_true_flat = y_test.flatten()
    y_pred_flat = pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    smape = smape_floor50(y_true_flat, y_pred_flat)
    
    return {'mae': mae, 'rmse': rmse, 'smape': smape}


def main():
    """Run day-level prediction experiment."""
    logger.info("Starting DeepRT-SOTA v2: Day-level prediction")
    
    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    df = load_data(data_path)
    
    # Split: use 2026-02 as test
    test_mask = (df['business_day'].dt.year == 2026) & (df['business_day'].dt.month == 2)
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()
    
    logger.info(f"Train days: {len(train_df['business_day'].unique())}")
    logger.info(f"Test days: {len(test_df['business_day'].unique())}")
    
    # Create day-level datasets
    logger.info("Creating day-level datasets...")
    X_train, y_train, feature_cols = create_day_level_dataset(train_df, seq_len_days=14, risk_features=False)
    X_test, y_test, _ = create_day_level_dataset(test_df, seq_len_days=14, risk_features=False)
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train and evaluate
    model_profile = 'deep_rt_tcn'  # Options: deep_rt_mlp, deep_rt_tcn, deep_rt_gru, deep_rt_transformer
    logger.info(f"\nTraining {model_profile} (day-level, seq_len_days=14)...")
    metrics = train_and_evaluate_day_level(X_train, y_train, X_test, y_test, model_profile='deep_rt_tcn', seq_len_days=14)
    
    logger.info("="*80)
    logger.info("Results (day-level prediction):")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  sMAPE_floor50: {metrics['smape']:.4f}")
    logger.info("="*80)
    
    # Save results
    results = {
        'model_profile': 'deep_rt_tcn',
        'prediction_mode': 'day_level',
        'seq_len_days': 14,
        'metrics': metrics,
    }
    
    out_path = Path('artifacts/deep_rt_sota/day_level_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
