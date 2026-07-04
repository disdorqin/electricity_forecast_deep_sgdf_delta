"""
Train DeepRT-SOTA v2 with residual-to-DA target.

Target: predict (rt_actual - da_anchor) instead of rt_actual directly.
This might work better because DA anchor is already a strong baseline.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features
from models.deep_sgdf_delta.deep_rt_sota_model import DeepRTSOTAModel, DeepRTSOTAModelConfig
from models.deep_sgdf_delta.metrics import smape_floor50

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def create_hourly_dataset_residual(df: pd.DataFrame, risk_features: bool = False):
    """Create hourly dataset for residual prediction.
    
    Args:
        df: Preprocessed DataFrame.
        risk_features: Whether to include risk features.
        
    Returns:
        X, y_residual, y_actual arrays.
    """
    # Build features
    df, feature_manifest = build_deep_rt_sota_features(
        df,
        risk_features=risk_features,
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
    
    if risk_features:
        feature_cols.extend(['negative_risk_score', 'spike_risk_score'])
    
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"Available features: {len(available_features)}")
    
    # Create hourly samples
    X = []
    y_residual = []
    y_actual = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Skip if any feature is NaN
        if row[available_features].isna().any():
            continue
        
        # Skip if rt_actual or da_anchor is NaN
        if pd.isna(row['rt_actual']) or pd.isna(row['da_anchor']):
            continue
        
        X.append(row[available_features].values)
        y_residual.append(row['rt_actual'] - row['da_anchor'])  # Residual
        y_actual.append(row['rt_actual'])
    
    X = np.array(X)
    y_residual = np.array(y_residual)
    y_actual = np.array(y_actual)
    
    logger.info(f"Created dataset: X shape {X.shape}")
    
    return X, y_residual, y_actual, available_features


def train_residual_model(X_train, y_train_residual, X_test, y_test_actual, da_test, config, epochs: int = 80):
    """Train residual model.
    
    Args:
        X_train: Training features.
        y_train_residual: Training residuals (rt_actual - da_anchor).
        X_test: Test features.
        y_test_actual: Test actual values.
        da_test: DA anchor for test set.
        config: Model configuration.
        epochs: Number of epochs.
        
    Returns:
        Metrics dictionary.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    config.n_features = X_train.shape[1]
    model = DeepRTSOTAModel(config).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model_profile}")
    logger.info(f"Parameters: {n_params:,}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_residual).unsqueeze(1).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if config.model_profile == "deep_rt_mlp":
            pred, _ = model(X_train_tensor)
        else:
            # Reshape for sequence models
            X_seq = X_train_tensor.reshape(X_train_tensor.shape[0], config.seq_len_days * 24, -1)
            pred, _ = model(X_seq)
        
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        if config.model_profile == "deep_rt_mlp":
            pred_residual, _ = model(X_test_tensor)
        else:
            X_seq_test = X_test_tensor.reshape(X_test_tensor.shape[0], config.seq_len_days * 24, -1)
            pred_residual, _ = model(X_seq_test)
        
        pred_residual = pred_residual.cpu().numpy().flatten()
    
    # Convert residual to actual prediction
    # pred_actual = da_test + pred_residual
    pred_actual = da_test + pred_residual
    
    # Compute metrics
    mae = mean_absolute_error(y_test_actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
    smape = smape_floor50(y_test_actual, pred_actual)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
        'target_mode': 'residual_to_da',
    }
    
    logger.info("="*50)
    logger.info(f"Results (Residual-to-DA, {config.model_profile}):")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  sMAPE: {smape:.4f}")
    logger.info("="*50)
    
    return metrics


def main():
    """Main function."""
    logger.info("Starting DeepRT-SOTA v2 residual-to-DA training...")
    
    # Load data
    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, encoding='gbk')
    logger.info(f"Loaded {len(df)} rows")
    
    # Rename columns
    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })
    
    # Parse timestamp
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Add business time columns
    df = add_business_time_columns(df, timestamp_col='ds')
    
    # Split train/test (target month: 2026-02)
    target_month = '2026-02'
    train_mask = df['business_day'] < target_month
    test_mask = (df['business_day'] >= target_month) & (df['business_day'] < '2026-03')
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train: {train_df['business_day'].nunique()} days")
    logger.info(f"Test: {test_df['business_day'].nunique()} days")
    
    # Load risk features
    risk_path = Path('artifacts/deep_rt_sota/synthetic_risk_features.csv')
    if risk_path.exists():
        logger.info(f"Loading risk features from {risk_path}...")
        risk_df = pd.read_csv(risk_path)
        risk_df['ds'] = pd.to_datetime(risk_df['ds'])
        risk_df = add_business_time_columns(risk_df, timestamp_col='ds')
        
        # Merge risk features
        train_df = train_df.merge(
            risk_df[['business_day', 'hour_business', 'negative_risk_score', 'spike_risk_score']],
            on=['business_day', 'hour_business'],
            how='left'
        )
        test_df = test_df.merge(
            risk_df[['business_day', 'hour_business', 'negative_risk_score', 'spike_risk_score']],
            on=['business_day', 'hour_business'],
            how='left'
        )
    
    # Create datasets (residual target)
    logger.info("Creating datasets (residual target)...")
    X_train, y_train_residual, y_train_actual, feature_cols = create_hourly_dataset_residual(train_df, risk_features=True)
    X_test, y_test_residual, y_test_actual, _ = create_hourly_dataset_residual(test_df, risk_features=True)
    
    # Get DA anchor for test set
    da_test = test_df.drop_duplicates(['business_day', 'hour_business']).sort_values(['business_day', 'hour_business'])['da_anchor'].values[:len(X_test)]
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model (try both MLP and TCN)
    results = {}
    
    for model_profile in ['deep_rt_mlp', 'deep_rt_tcn']:
        logger.info(f"\nTraining {model_profile} (residual-to-DA)...")
        
        config = DeepRTSOTAModelConfig(
            model_profile=model_profile,
            seq_len_days=14,
            target_mode='residual_to_da',
            n_features=len(feature_cols),
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
            output_dim=1,
        )
        
        metrics = train_residual_model(
            X_train, y_train_residual, X_test, y_test_actual, da_test, config, epochs=80
        )
        
        results[model_profile] = metrics
    
    # Save results
    import json
    
    output_path = Path('artifacts/deep_rt_sota/residual_exp/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_results = {}
    for key, val in results.items():
        serializable_results[key] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in val.items()}
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print comparison
    print("\n" + "="*80)
    print("RESIDUAL-TO-DA RESULTS (2026-02)")
    print("="*80)
    
    for model_profile, metrics in results.items():
        print(f"\n{model_profile} (residual-to-DA):")
        print(f"  sMAPE_floor50: {metrics['smape']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    print(f"\nDA anchor baseline: 26.70")
    print("="*80)


if __name__ == '__main__':
    main()
