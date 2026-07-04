"""
Phase 1: Comprehensive Feature Engineering.

Build derived features, test feature importance, design experiments.
Goal: Push day-level sMAPE below 20.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def build_comprehensive_features(df, online_mode=True):
    """
    Build comprehensive feature set.
    
    Args:
        df: DataFrame with columns from shandong_pmos_hourly.csv
        online_mode: If True, only use features available online (no future data)
    
    Returns:
        DataFrame with all features
    """
    
    df = df.copy()
    
    # Ensure time column is datetime
    if '时刻' in df.columns:
        df['times'] = pd.to_datetime(df['时刻'])
    elif 'times' not in df.columns:
        raise ValueError("No time column found")
    
    df = df.sort_values('times').reset_index(drop=True)
    
    # === TARGET COLUMNS ===
    # These are targets, not features
    target_cols = ['rt_price', 'da_price', 'residual']
    
    if '实时电价' in df.columns:
        df['rt_price'] = df['实时电价']
    if '日前电价' in df.columns:
        df['da_price'] = df['日前电价']
    if 'rt_price' in df.columns and 'da_price' in df.columns:
        df['residual'] = df['rt_price'] - df['da_price']
    
    # === ONLINE FEATURES (available for prediction) ===
    
    # 1. Calendar features
    df['hour_business'] = df['times'].dt.hour + 1
    df['period'] = pd.cut(df['hour_business'], bins=[0, 8, 16, 24], 
                           labels=['1_8', '9_16', '17_24'], include_lowest=True)
    df['day_of_week'] = df['times'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['times'].dt.month
    df['day_of_month'] = df['times'].dt.day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_business'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_business'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 2. DA anchor features
    df['da_anchor'] = df['da_price']
    df['da_negative'] = (df['da_price'] < 0).astype(int)
    df['da_high'] = (df['da_price'] > 500).astype(int)
    df['da_low'] = (df['da_price'] < 100).astype(int)
    
    # DA quantiles (rolling 30d)
    # Use percentile rank instead of qcut (qcut doesn't work with rolling)
    df['da_pct_rank_7d'] = df['da_price'].rolling(7*24, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    ).fillna(0.5)
    df['da_pct_rank_30d'] = df['da_price'].rolling(30*24, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    ).fillna(0.5)
    
    # 3. Forecast features (ONLY forecasts, no actuals, to avoid leakage)
    forecast_cols = [
        '地方电厂总加预测值',
        '联络线受电负荷预测值',
        '风电总加预测值',
        '光伏总加预测值',
        '核电总加预测值',
        '自备机组总加预测值',
        '试验机组总加预测值',
        '直调负荷预测值',
        '竞价空间预测值',
        '新能源总加预测值'
    ]
    
    for col in forecast_cols:
        if col in df.columns:
            # Normalize column name
            feature_name = col.replace('总加预测值', '').replace('预测值', '').replace(' ', '_')
            df[feature_name + '_forecast'] = df[col]
    
    # 4. Historical RT/lags (only up to D-1, no future data)
    if 'rt_price' in df.columns:
        # RT lags (only use up to D-1 for online prediction)
        for lag in [24, 48, 72, 168]:  # 1d, 2d, 3d, 7d
            df[f'rt_lag_{lag}h'] = df['rt_price'].shift(lag)
        
        # RT rolling stats (trailing, no future)
        for window in [24, 48, 168]:
            df[f'rt_roll_mean_{window}h'] = df['rt_price'].rolling(window, min_periods=1).mean().shift(24)  # Only use up to D-1
            df[f'rt_roll_std_{window}h'] = df['rt_price'].rolling(window, min_periods=1).std().shift(24)
    
    # 5. Historical residual lags (only up to D-1)
    if 'residual' in df.columns:
        for lag in [24, 48, 72, 168]:
            df[f'residual_lag_{lag}h'] = df['residual'].shift(lag)
        
        # Residual rolling stats
        for window in [24, 48, 168]:
            df[f'residual_roll_mean_{window}h'] = df['residual'].rolling(window, min_periods=1).mean().shift(24)
            df[f'residual_roll_std_{window}h'] = df['residual'].rolling(window, min_periods=1).std().shift(24)
    
    # 6. DA lags and rolling stats
    if 'da_price' in df.columns:
        for lag in [24, 48, 72, 168]:
            df[f'da_lag_{lag}h'] = df['da_price'].shift(lag)
        
        for window in [24, 48, 168]:
            df[f'da_roll_mean_{window}h'] = df['da_price'].rolling(window, min_periods=1).mean().shift(24)
            df[f'da_roll_std_{window}h'] = df['da_price'].rolling(window, min_periods=1).std().shift(24)
    
    # 7. Interaction features (DA * forecast features)
    if 'da_price' in df.columns:
        for col in forecast_cols:
            if col in df.columns:
                feature_name = col.replace('总加预测值', '').replace('预测值', '').replace(' ', '_')
                df[f'da_x_{feature_name}'] = df['da_price'] * df[col]
    
    # 8. Time since last negative DA / spike
    if 'da_price' in df.columns:
        df['hours_since_negative_da'] = np.nan
        negative_mask = df['da_price'] < 0
        last_negative_idx = -999
        for i in range(len(df)):
            if negative_mask.iloc[i]:
                last_negative_idx = i
            if last_negative_idx == -999:
                df.loc[i, 'hours_since_negative_da'] = 9999
            else:
                df.loc[i, 'hours_since_negative_da'] = i - last_negative_idx
    
    # === LEAKAGE-EXCLUDED COLUMNS ===
    # These should NOT be used as features
    leakage_cols = [
        '实时电价', 'rt_price',  # Target
        '地方电厂总加实际值', '联络线受电负荷实际值', '风电总加实际值',
        '光伏总加实际值', '核电总加实际值', '自备机组总加实际值',
        '试验机组总加实际值', '直调负荷实际值', '竞价空间实际值',
        '新能源总加实际值'
    ]
    
    # Mark leakage columns
    for col in leakage_cols:
        if col in df.columns:
            df[f'{col}_LEAKAGE'] = df[col]  # Mark as leakage, don't use for training
    
    return df

def get_feature_columns(df):
    """Get list of valid feature columns (no leakage, no targets)."""
    
    # Columns to exclude (targets, leakage, time)
    exclude_patterns = [
        'rt_price', 'da_price', 'residual',  # Targets
        '实时电价', '日前电价',  # Chinese target names
        '实际值',  # Actual values (leakage)
        'LEAKAGE',  # Marked leakage columns
        'times', '时刻',  # Time columns
        'period',  # Categorical
    ]
    
    feature_cols = []
    for col in df.columns:
        # Skip if matches exclude pattern
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Skip non-numeric
        if df[col].dtype in ['object', 'category']:
            continue
        # Skip all-NaN
        if df[col].isna().all():
            continue
        feature_cols.append(col)
    
    return feature_cols

def compute_smape_floor50(y_true, y_pred, floor=50.0):
    """Compute sMAPE with floor=50."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, floor)
    return 100 * np.mean(np.abs(y_true - y_pred) / denom)

def compute_day_level_smape(y_true, y_pred, timestamps):
    """Compute day-level sMAPE."""
    df = pd.DataFrame({
        'timestamp': timestamps,
        'y_true': y_true,
        'y_pred': y_pred
    })
    df['date'] = df['timestamp'].dt.date
    
    daily_true = df.groupby('date')['y_true'].mean()
    daily_pred = df.groupby('date')['y_pred'].mean()
    
    return compute_smape_floor50(daily_true.values, daily_pred.values)

def run_feature_experiment(data_path, out_dir):
    """Run feature engineering experiment."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(data_path, encoding='gbk', parse_dates=['时刻'])
    df = df.sort_values('时刻').reset_index(drop=True)
    
    print(f"=== Phase 1: Feature Engineering ===\n")
    print(f"Raw data: {df.shape}")
    print(f"Time range: {df['时刻'].min()} to {df['时刻'].max()}")
    
    # Build comprehensive features
    print(f"\nBuilding comprehensive features...")
    df_features = build_comprehensive_features(df, online_mode=True)
    
    print(f"DataFrame shape after feature engineering: {df_features.shape}")
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features)
    print(f"\nTotal features: {len(feature_cols)}")
    
    # Save feature inventory
    feature_inventory = pd.DataFrame({
        'feature': feature_cols,
        'dtype': [str(df_features[col].dtype) for col in feature_cols],
        'nan_pct': [df_features[col].isna().sum() / len(df_features) for col in feature_cols]
    })
    feature_inventory.to_csv(out_dir / 'feature_inventory.csv', index=False, encoding='utf-8-sig')
    print(f"\nFeature inventory saved to {out_dir / 'feature_inventory.csv'}")
    
    # === Experiment: Test feature importance ===
    print(f"\n=== Experiment: Feature Importance ===")
    
    # Use 2024-05 to 2025-12 for training/validation
    df_recent = df_features[(df_features['times'] >= '2024-05-01') & (df_features['times'] < '2026-01-01')].copy()
    
    print(f"Recent data (for experiment): {len(df_recent)} rows")
    
    # Prepare data
    X = df_recent[feature_cols].fillna(0).values
    y = (df_recent['rt_price'] - df_recent['da_price']).values  # Residual as target
    da_anchor = df_recent['da_price'].values
    
    # Split: use last 3 months as validation
    val_start = '2025-10-01'
    train_mask = df_recent['times'] < val_start
    val_mask = df_recent['times'] >= val_start
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    da_val = da_anchor[val_mask]
    rt_val = df_recent.loc[val_mask, 'rt_price'].values
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train model to predict residual
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    
    # Model 1: Ridge (for feature importance)
    print(f"\nTraining Ridge model...")
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)
    
    # Feature importance (abs(coefficients))
    ridge_importance = np.abs(model_ridge.coef_)
    ridge_top_features = np.argsort(ridge_importance)[::-1][:20]
    
    print(f"\nTop 20 features (Ridge):")
    for i, idx in enumerate(ridge_top_features, 1):
        print(f"{i:2d}. {feature_cols[idx]}: {ridge_importance[idx]:.4f}")
    
    # Model 2: HGB (for residual prediction)
    print(f"\nTraining HGB model...")
    model_hgb = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model_hgb.fit(X_train, y_train)
    
    # HGB feature importance
    hgb_importance = model_hgb.feature_importances_
    hgb_top_features = np.argsort(hgb_importance)[::-1][:20]
    
    print(f"\nTop 20 features (HGB):")
    for i, idx in enumerate(hgb_top_features, 1):
        print(f"{i:2d}. {feature_cols[idx]}: {hgb_importance[idx]:.4f}")
    
    # === Evaluate: Does residual prediction improve sMAPE? ===
    print(f"\n=== Evaluation: Residual Prediction ===")
    
    # Predict residual
    residual_pred = model_hgb.predict(X_val)
    
    # Apply correction with different alpha/clip
    results = []
    
    for alpha in [0.0, 0.02, 0.05, 0.10, 0.20]:
        for clip in [0, 5, 10, 20, 30, 50]:
            
            if alpha == 0.0:
                final_pred = da_val
            else:
                if clip > 0:
                    residual_clipped = np.clip(residual_pred, -clip, clip)
                else:
                    residual_clipped = residual_pred
                final_pred = da_val + alpha * residual_clipped
            
            # Compute sMAPE
            smape = compute_day_level_smape(rt_val, final_pred, df_recent.loc[val_mask, 'times'].values)
            da_smape = compute_day_level_smape(rt_val, da_val, df_recent.loc[val_mask, 'times'].values)
            
            results.append({
                'alpha': alpha,
                'clip': clip,
                'model_smape': smape,
                'da_smape': da_smape,
                'improvement': da_smape - smape
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / 'residual_prediction_results.csv', index=False, encoding='utf-8-sig')
    
    print(f"\nResults (top 5):")
    print(results_df.sort_values('improvement', ascending=False).head(10))
    
    # Check if any improvement
    best_improvement = results_df['improvement'].max()
    print(f"\nBest improvement: {best_improvement:.4f}")
    
    if best_improvement > 0.1:
        print(f"\n✅ Feature engineering shows promise!")
        print(f"   Best config: alpha={results_df.loc[results_df['improvement'].idxmax(), 'alpha']}, clip={results_df.loc[results_df['improvement'].idxmax(), 'clip']}")
    else:
        print(f"\n❌ Feature engineering does not improve sMAPE significantly.")
        print(f"   Residual is still unpredictable with these features.")
    
    # Generate report
    report_path = out_dir / 'feature_engineering_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1: Feature Engineering Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Total features built: {len(feature_cols)}\n")
        f.write(f"Best improvement: {best_improvement:.4f}pp\n\n")
        
        f.write("## Top 20 Features (Ridge)\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for i, idx in enumerate(ridge_top_features, 1):
            f.write(f"| {i} | {feature_cols[idx]} | {ridge_importance[idx]:.4f} |\n")
        f.write("\n")
        
        f.write("## Top 20 Features (HGB)\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for i, idx in enumerate(hgb_top_features, 1):
            f.write(f"| {i} | {feature_cols[idx]} | {hgb_importance[idx]:.4f} |\n")
        f.write("\n")
        
        f.write("## Results\n\n")
        f.write(results_df.sort_values('improvement', ascending=False).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Conclusion\n\n")
        if best_improvement > 0.1:
            f.write("✅ Feature engineering shows promise. Continue to Phase 2.\n")
        else:
            f.write("❌ No significant improvement. Need different features or approach.\n")
    
    print(f"\nReport saved to {report_path}")
    
    return results_df, feature_cols

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    run_feature_experiment(args.data_path, args.out_dir)

if __name__ == '__main__':
    main()
