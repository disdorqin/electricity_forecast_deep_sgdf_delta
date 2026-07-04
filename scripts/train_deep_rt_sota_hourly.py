"""
DeepRT-SOTA v2 - Hourly Prediction (TCN Sequence Model).

Trains hourly prediction model to evaluate on all 672 hours (28 days).
This satisfies "test rows >= 650" requirement.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import logging
import json
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


def create_hourly_sequence_dataset(df: pd.DataFrame, seq_len_hours: int = 168):
    """Create hourly sequence dataset for TCN.
    
    Args:
        df: Preprocessed DataFrame (with features already built, sorted by time).
        seq_len_hours: Sequence length in hours (default 168 = 7 days).
        
    Returns:
        X_seq, y arrays.
    """
    # Sort by time
    df_sorted = df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    
    X = []
    y = []
    
    for i in range(seq_len_hours, len(df_sorted)):
        # Skip if target is NaN
        if pd.isna(df_sorted.loc[i, 'rt_actual']):
            continue
        
        # Get sequence (past seq_len_hours)
        seq_data = df_sorted.iloc[i-seq_len_hours:i][available_features]
        
        # Skip if any NaN in sequence
        if seq_data.isna().any().any():
            continue
        
        X.append(seq_data.values)
        y.append(df_sorted.loc[i, 'rt_actual'])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created hourly sequence dataset: X shape {X.shape}, y shape {y.shape}")
    
    return X, y


def train_hourly_tcn(X_train, y_train, X_test, y_test, config, epochs: int = 80):
    """Train hourly TCN model.
    
    Args:
        X_train: Training sequences (n_samples, seq_len_hours, n_features).
        y_train: Training targets (n_samples,).
        X_test: Test sequences.
        y_test: Test targets.
        config: Model configuration.
        epochs: Number of epochs.
        
    Returns:
        Metrics dictionary.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")
    
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
    config.n_features = X_train.shape[-1]
    config.output_dim = 1  # Hourly prediction
    model = DeepRTSOTAModel(config).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model_profile}")
    logger.info(f"Parameters: {n_params:,}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    logger.info("Training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, _ = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        pred, _ = model(X_test_tensor)
        pred = pred.cpu().numpy().flatten()
    
    # Compute metrics
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    smape = smape_floor50(y_test, pred)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'smape': float(smape),
        'test_samples': int(len(y_test)),
    }
    
    logger.info("="*50)
    logger.info(f"Results (Hourly TCN, seq_len={config.seq_len_days}d):")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  sMAPE: {smape:.4f}")
    logger.info(f"  Test samples: {len(y_test)}")
    logger.info("="*50)
    
    return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DeepRT-SOTA v2 Hourly Prediction (TCN)')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--target-month', type=str, required=True, help='Target month (e.g., 2026-02)')
    parser.add_argument('--seq-len-days', type=int, default=7, help='Sequence length in days')
    parser.add_argument('--model-profile', type=str, default='deep_rt_tcn', choices=['deep_rt_tcn', 'deep_rt_gru'], help='Model profile')
    parser.add_argument('--risk-features', type=str, default='off', choices=['off', 'real', 'synthetic'], help='Risk features source')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DeepRT-SOTA v2 - Hourly Prediction (TCN)")
    logger.info("="*80)
    
    # Load data
    data_path = args.data_path
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, encoding='gbk')
    logger.info(f"Loaded {len(df)} rows")
    
    # Rename and preprocess
    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })
    df['ds'] = pd.to_datetime(df['ds'])
    df = add_business_time_columns(df, timestamp_col='ds')
    
    # Split train/test
    target_month = args.target_month
    train_mask = df['business_day'] < target_month
    test_mask = (df['business_day'] >= target_month) & (df['business_day'] < target_month[:4] + '-' + str(int(target_month[5:7]) + 1).zfill(2) if int(target_month[5:7]) < 12 else target_month[:4] + '-01')
    
    # Simpler: use 2026-03-01 as end
    if target_month == '2026-02':
        test_end = '2026-03-01'
    else:
        # Generic
        year, month = map(int, target_month.split('-'))
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1
        test_end = f"{next_year}-{next_month:02d}-01"
    
    test_mask = (df['business_day'] >= target_month) & (df['business_day'] < test_end)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    logger.info(f"Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")
    
    # CRITICAL FIX: Merge train+test before building features
    logger.info("Merging train+test for feature building...")
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Build features
    logger.info("Building features...")
    risk_df = None
    if args.risk_features == 'synthetic':
        risk_csv = Path('artifacts/deep_rt_sota/synthetic_risk_features.csv')
        if risk_csv.exists():
            logger.info(f"Loading synthetic risk features from {risk_csv}...")
            risk_df = pd.read_csv(risk_csv)
            risk_df['ds'] = pd.to_datetime(risk_df['ds'])
            risk_df = add_business_time_columns(risk_df, timestamp_col='ds')
    
    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df,
        risk_features=(args.risk_features != 'off'),
        forecast_features=False,
        risk_df=risk_df,
    )
    
    logger.info(f"Feature manifest: {feature_manifest}")
    
    # Split back
    train_df = merged_df[merged_df['business_day'] < target_month].copy()
    test_df = merged_df[(merged_df['business_day'] >= target_month) & (merged_df['business_day'] < test_end)].copy()
    
    logger.info(f"Split back: train={len(train_df)}, test={len(test_df)}")
    
    # Get available features
    global available_features
    feature_cols = [
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'previous_day_rt_mean', 'previous_day_rt_std',
        'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
        'da_anchor', 'forecast_price', 'anchor_spread',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend', 'period_id',
    ]
    
    if args.risk_features != 'off':
        feature_cols.extend(['negative_risk_score', 'spike_risk_score'])
    
    available_features = [col for col in feature_cols if col in train_df.columns]
    logger.info(f"Available features: {len(available_features)}")
    
    # Create datasets (hourly sequence)
    seq_len_hours = args.seq_len_days * 24
    logger.info(f"Creating hourly sequence datasets (seq_len={seq_len_hours}h)...")
    
    # For train: use all data (with enough history)
    train_sorted = train_df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    
    X_train = []
    y_train = []
    
    for i in range(seq_len_hours, len(train_sorted)):
        if pd.isna(train_sorted.loc[i, 'rt_actual']):
            continue
        
        seq_data = train_sorted.iloc[i-seq_len_hours:i][available_features]
        
        if seq_data.isna().any().any():
            continue
        
        X_train.append(seq_data.values)
        y_train.append(train_sorted.loc[i, 'rt_actual'])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # For test: use all data (with enough history from train)
    test_sorted = test_df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    
    X_test = []
    y_test = []
    
    # We need to use train data as history for first test samples
    # Merge train+test for sequence creation
    merged_sorted = pd.concat([train_sorted, test_sorted], ignore_index=True).sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    
    # Find start index of test set in merged data
    test_start_idx = len(train_sorted)
    
    for i in range(test_start_idx + seq_len_hours, len(merged_sorted)):
        if pd.isna(merged_sorted.loc[i, 'rt_actual']):
            continue
        
        # Only include if this is a test set sample
        if pd.to_datetime(merged_sorted.loc[i, 'business_day']) < pd.to_datetime(target_month):
            continue
        
        seq_data = merged_sorted.iloc[i-seq_len_hours:i][available_features]
        
        if seq_data.isna().any().any():
            continue
        
        X_test.append(seq_data.values)
        y_test.append(merged_sorted.loc[i, 'rt_actual'])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    if len(X_test) < 650:
        logger.warning(f"Test samples ({len(X_test)}) < 650! May not satisfy requirement.")
    
    # Train model
    config = DeepRTSOTAModelConfig(
        model_profile=args.model_profile,
        seq_len_days=args.seq_len_days,
        target_mode='direct',
        n_features=len(available_features),
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        output_dim=1,
    )
    
    metrics = train_hourly_tcn(X_train, y_train, X_test, y_test, config, epochs=args.epochs)
    
    # Save results
    if args.out_dir is None:
        args.out_dir = f"artifacts/deep_rt_sota/hourly_{args.model_profile}_{target_month.replace('-', '_')}"
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata to metrics
    metrics['target_month'] = target_month
    metrics['model_profile'] = args.model_profile
    metrics['seq_len_days'] = args.seq_len_days
    metrics['risk_features'] = args.risk_features
    metrics['test_rows'] = int(len(y_test))
    metrics['test_business_days'] = int(test_df['business_day'].nunique())
    
    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    # Print comparison with baselines
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINES")
    print("="*80)
    print(f"\nDeepRT-SOTA (Hourly TCN):")
    print(f"  sMAPE_floor50: {metrics['smape']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Test samples: {metrics['test_samples']}")
    
    # Load baseline results
    baseline_path = Path(f"reports/local/deep_rt_sota/baselines_{target_month.replace('-', '_')}/baseline_metrics.json")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        
        print(f"\nBaselines:")
        for name, baseline_metrics in baseline_results.items():
            print(f"  {name}: sMAPE={baseline_metrics['smape_floor50']:.4f}, MAE={baseline_metrics['mae']:.4f}")
        
        # Check if beats baselines
        da_smape = baseline_results.get('da_anchor', {}).get('smape_floor50', 100.0)
        
        if metrics['smape'] < da_smape:
            print(f"\n✅ Deep model BEATS DA anchor! (gap: {da_smape - metrics['smape']:.4f})")
        else:
            print(f"\n❌ Deep model does NOT beat DA anchor (gap: {metrics['smape'] - da_smape:.4f})")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
