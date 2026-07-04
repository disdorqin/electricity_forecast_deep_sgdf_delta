"""
Quick experiment: Add risk features and retrain.

Tests whether adding negative_risk_score and spike_risk_score improves performance.
Uses CORRECT smape_floor50 metric.
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


def create_hourly_dataset(df, risk_features=False):
    """Create simple hourly dataset."""
    # Build features
    df, feature_manifest = build_deep_rt_sota_features(
        df,
        risk_features=risk_features,
        forecast_features=False,
    )
    
    logger.info(f"Feature manifest: {feature_manifest.get('risk_features_used', False)}")
    
    # Select features
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
        for col in risk_cols:
            if col in df.columns:
                feature_cols.append(col)
                logger.info(f"Added risk feature: {col}")
    
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available features: {len(available_features)}: {available_features}")
    
    # Create samples (skip rows with NaN)
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
    
    return X, y, available_features


def train_and_evaluate(X_train, y_train, X_test, y_test, model_profile='deep_rt_tcn'):
    """Train and evaluate model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    config = DeepRTSOTAModelConfig(
        model_profile=model_profile,
        target_mode='direct',
        n_features=X_train.shape[1],
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        output_dim=1,
    )
    
    model = DeepRTSOTAModel(config).to(device)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    
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
    """Run experiment with/without risk features."""
    logger.info("Starting experiment: Effect of risk features")
    
    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    df = load_data(data_path)
    
    # Split: use 2026-02 as test
    test_mask = (df['business_day'].dt.year == 2026) & (df['business_day'].dt.month == 2)
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()
    
    results = []
    
    # Experiment 1: Without risk features (use MLP for now, TCN needs 3D input)
    logger.info("\n" + "="*80)
    logger.info("Experiment 1: WITHOUT risk features (MLP)")
    logger.info("="*80)
    
    train_df_no_risk, _ = build_deep_rt_sota_features(train_df, risk_features=False, forecast_features=False)
    test_df_no_risk, _ = build_deep_rt_sota_features(test_df, risk_features=False, forecast_features=False)
    
    X_train, y_train, _ = create_hourly_dataset(train_df_no_risk, risk_features=False)
    X_test, y_test, _ = create_hourly_dataset(test_df_no_risk, risk_features=False)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    metrics_no_risk = train_and_evaluate(X_train, y_train, X_test, y_test, model_profile='deep_rt_mlp')
    results.append({
        'experiment': 'without_risk_features',
        'metrics': metrics_no_risk,
    })
    
    logger.info(f"Results (without risk): sMAPE={metrics_no_risk['smape']:.2f}")
    
    # Experiment 2: With risk features (if available)
    logger.info("\n" + "="*80)
    logger.info("Experiment 2: WITH risk features")
    logger.info("="*80)
    
    train_df_with_risk, manifest = build_deep_rt_sota_features(train_df, risk_features=True, forecast_features=False)
    test_df_with_risk, _ = build_deep_rt_sota_features(test_df, risk_features=True, forecast_features=False)
    
    logger.info(f"Risk features used: {manifest.get('risk_features_used', False)}")
    
    X_train_risk, y_train_risk, feature_cols_risk = create_hourly_dataset(train_df_with_risk, risk_features=True)
    X_test_risk, y_test_risk, _ = create_hourly_dataset(test_df_with_risk, risk_features=True)
    
    logger.info(f"Train: {len(X_train_risk)}, Test: {len(X_test_risk)}")
    logger.info(f"Features: {len(feature_cols_risk)}")
    
    if len(feature_cols_risk) > 18:  # More than without risk features
        metrics_with_risk = train_and_evaluate(X_train_risk, y_train_risk, X_test_risk, y_test_risk, model_profile='deep_rt_tcn')
        results.append({
            'experiment': 'with_risk_features',
            'metrics': metrics_with_risk,
        })
        
        logger.info(f"Results (with risk): sMAPE={metrics_with_risk['smape']:.2f}")
        
        # Compare
        improvement = metrics_no_risk['smape'] - metrics_with_risk['smape']
        logger.info("\n" + "="*80)
        logger.info(f"Comparison: Risk features {'IMPROVE' if improvement > 0 else 'WORSEN'} by {abs(improvement):.2f} pp")
        logger.info("="*80)
    else:
        logger.info("Risk features not available or not used. Skipping Experiment 2.")
    
    # Save results
    out_path = Path('artifacts/deep_rt_sota/risk_feature_experiment.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
