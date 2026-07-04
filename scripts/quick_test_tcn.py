"""
Quick test: Train TCN model (hourly prediction).

Compare with MLP baseline.
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


def create_sequence_dataset(df: pd.DataFrame, seq_len_hours: int = 168):
    """Create sequence dataset for TCN.
    
    Args:
        df: Preprocessed DataFrame (sorted by business_day, hour_business).
        seq_len_hours: Sequence length in hours (default 168 = 7 days).
        
    Returns:
        X_seq, y arrays.
    """
    # Build features
    df, _ = build_deep_rt_sota_features(df, risk_features=False, forecast_features=False)
    
    feature_cols = [
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'previous_day_rt_mean', 'previous_day_rt_std',
        'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
        'da_anchor', 'forecast_price', 'anchor_spread',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'period_id',
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Sort by time
    df_sorted = df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    
    X = []
    y = []
    
    for i in range(seq_len_hours, len(df_sorted)):
        # Skip if target is NaN
        if pd.isna(df_sorted.loc[i, 'rt_actual']):
            continue
        
        # Get sequence
        seq_data = df_sorted.iloc[i-seq_len_hours:i][available_features]
        
        # Skip if any NaN in sequence
        if seq_data.isna().any().any():
            continue
        
        X.append(seq_data.values)
        y.append(df_sorted.loc[i, 'rt_actual'])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created sequence dataset: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, available_features


def main():
    """Main function."""
    logger.info("Starting TCN quick test...")
    
    # Load data
    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    df = pd.read_csv(data_path, encoding='gbk')
    logger.info(f"Loaded {len(df)} rows")
    
    # Preprocess
    df = df.rename(columns={'时刻': 'ds', '日前电价': 'da_anchor', '实时电价': 'rt_actual'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = add_business_time_columns(df, timestamp_col='ds')
    
    # Sort
    df = df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    
    # Split (use last 672 rows = 28 days as test)
    test_size = 672
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # Create datasets
    logger.info("Creating sequence datasets...")
    X_train, y_train, feature_cols = create_sequence_dataset(train_df, seq_len_hours=168)
    X_test, y_test, _ = create_sequence_dataset(test_df, seq_len_hours=168)
    
    logger.info(f"Final: Train {len(X_train)}, Test {len(X_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("No valid samples!")
        return
    
    # Train TCN model
    config = DeepRTSOTAModelConfig(
        model_profile='deep_rt_tcn',
        seq_len_days=7,  # 7 days = 168 hours
        target_mode='direct',
        n_features=len(feature_cols),
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        output_dim=1,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale features (per sample)
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    scaler = StandardScaler()
    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler.transform(X_test_2d)
    
    X_train_scaled = X_train_2d_scaled.reshape(X_train_shape)
    X_test_scaled = X_test_2d_scaled.reshape(X_test_shape)
    
    # Create model
    model = DeepRTSOTAModel(config).to(device)
    logger.info(f"TCN parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    logger.info("Training TCN...")
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
    logger.info("Results (TCN, hourly, seq_len=168h):")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  sMAPE_floor50: {smape:.4f}")
    logger.info("="*50)
    
    # Compare with DA anchor
    # Get DA anchor for test samples
    test_df_sorted = test_df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    da_test_list = []
    valid_indices = []
    
    for i in range(168, len(test_df_sorted)):
        if pd.notna(test_df_sorted.loc[i, 'rt_actual']) and not pd.isna(test_df_sorted.loc[i, 'da_anchor']):
            seq_data = test_df_sorted.iloc[i-168:i][['da_anchor']]  # Simplified check
            if not seq_data.isna().any().any():
                da_test_list.append(test_df_sorted.loc[i, 'da_anchor'])
                valid_indices.append(i)
    
    if len(da_test_list) == len(y_test):
        da_smape = smape_floor50(y_test, np.array(da_test_list))
        logger.info(f"\nDA anchor sMAPE: {da_smape:.4f}")
        
        if smape < da_smape:
            logger.info("✅ TCN BEATS DA anchor!")
        else:
            logger.info(f"❌ TCN does NOT beat DA anchor (gap: {smape - da_smape:.4f})")


if __name__ == '__main__':
    main()
