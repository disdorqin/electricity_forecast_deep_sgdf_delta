"""
Quick test: Train MLP model without risk features to verify pipeline.

Target: direct prediction (not residual).
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features
from models.deep_sgdf_delta.deep_rt_sota_model import DeepRTSOTAModel, DeepRTSOTAModelConfig
from models.deep_sgdf_delta.metrics import smape_floor50

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def create_hourly_dataset_simple(df: pd.DataFrame):
    """Create hourly dataset (simplified, no risk features)."""
    # Build features (without risk features)
    df, feature_manifest = build_deep_rt_sota_features(
        df,
        risk_features=False,
        forecast_features=False,
    )
    
    # Check available features
    feature_cols = [
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'previous_day_rt_mean', 'previous_day_rt_std',
        'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
        'da_anchor', 'forecast_price', 'anchor_spread',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'period_id',
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available features: {len(available_features)}")
    logger.info(f"Feature columns: {available_features}")
    
    # Check NaN before creating samples
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Rows with all features non-NaN: {df[available_features].notna().all(axis=1).sum()}")
    
    # Create samples
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
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created dataset: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, available_features


def main():
    """Main function."""
    logger.info("Starting quick test (no risk features)...")
    
    # Load data
    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    df = pd.read_csv(data_path, encoding='gbk')
    logger.info(f"Loaded {len(df)} rows")
    
    # Rename and parse
    df = df.rename(columns={'时刻': 'ds', '日前电价': 'da_anchor', '实时电价': 'rt_actual'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = add_business_time_columns(df, timestamp_col='ds')
    
    # Split
    target_month = '2026-02'
    train_mask = df['business_day'] < target_month
    test_mask = (df['business_day'] >= target_month) & (df['business_day'] < '2026-03')
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # Create datasets
    X_train, y_train, feature_cols = create_hourly_dataset_simple(train_df)
    X_test, y_test, _ = create_hourly_dataset_simple(test_df)
    
    logger.info(f"Final: Train {len(X_train)}, Test {len(X_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("No valid samples! Check feature building.")
        return
    
    # Train MLP model
    config = DeepRTSOTAModelConfig(
        model_profile='deep_rt_mlp',
        seq_len_days=7,
        target_mode='direct',
        n_features=len(feature_cols),
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        output_dim=1,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = DeepRTSOTAModel(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    logger.info("Training...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        pred, _ = model(X_test_tensor)
        pred = pred.cpu().numpy().flatten()
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    smape = smape_floor50(y_test, pred)
    
    logger.info("="*50)
    logger.info("Results (MLP, no risk features):")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  sMAPE_floor50: {smape:.4f}")
    logger.info("="*50)
    
    # Compare with DA anchor
    da_test = test_df.drop_duplicates(['business_day', 'hour_business']).sort_values(['business_day', 'hour_business'])['da_anchor'].values[:len(y_test)]
    
    if len(da_test) == len(y_test):
        da_smape = smape_floor50(y_test, da_test)
        da_mae = mean_absolute_error(y_test, da_test)
        
        logger.info("\nDA Anchor Comparison:")
        logger.info(f"  DA sMAPE: {da_smape:.4f}")
        logger.info(f"  DA MAE: {da_mae:.4f}")
        
        if smape < da_smape:
            logger.info("\n✅ Model BEATS DA anchor!")
        else:
            logger.info(f"\n❌ Model does NOT beat DA anchor (gap: {smape - da_smape:.4f})")


if __name__ == '__main__':
    main()
