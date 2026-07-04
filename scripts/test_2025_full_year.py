"""
Test RT-Assist-1 (alpha=1.0) on full year 2025.

Walk-forward backtest:
  For each month in 2025 (Jan~Dec):
    - Train on all data before this month
    - Test on this month
    - Compute day-level sMAPE

Output: monthly sMAPE + average.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# sMAPE
# =============================================================================
def compute_smape_floor50(y_true, y_pred, floor=50.0):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, floor)
    return 100 * np.mean(np.abs(y_true - y_pred) / denom)

def compute_day_level_smape(y_true, y_pred, timestamps):
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'y_true': y_true,
        'y_pred': y_pred
    })
    df['date'] = df['timestamp'].dt.date
    daily_true = df.groupby('date')['y_true'].mean()
    daily_pred = df.groupby('date')['y_pred'].mean()
    return compute_smape_floor50(daily_true.values, daily_pred.values)

# =============================================================================
# Feature Engineering (same as Phase 2+3+alpha1.0)
# =============================================================================
def add_features(df):
    df = df.copy()
    
    # Ensure times is datetime
    df['times'] = pd.to_datetime(df['times'])
    
    # Residual
    df['residual'] = df['rt_price'] - df['da_price']
    
    # Calendar
    df['hour'] = df['times'].dt.hour + 1  # 1-24
    df['dayofweek'] = df['times'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['month'] = df['times'].dt.month
    
    # Period buckets (Phase 3)
    df['period'] = pd.cut(df['hour'], bins=[0, 8, 16, 24],
                           labels=['p1_8', 'p9_16', 'p17_24'], include_lowest=True)
    
    # Bucket features (Phase 2)
    df['da_price_level'] = pd.cut(df['da_price'],
                                   bins=[-np.inf, 0, 100, 500, np.inf],
                                   labels=['negative', 'low', 'mid', 'high'])
    df['abs_residual_bucket'] = pd.cut(np.abs(df['residual']),
                                         bins=[0, 50, 150, 500, np.inf],
                                         labels=['small', 'medium', 'large', 'extreme'])
    
    # DA anchor features
    df['da_anchor'] = df['da_price']
    df['da_negative'] = (df['da_price'] < 0).astype(int)
    df['da_high'] = (df['da_price'] > 500).astype(int)
    
    # Lags (D-1 only, no future)
    for lag in [24, 48, 72, 168]:
        df[f'da_lag_{lag}h'] = df['da_price'].shift(lag)
        df[f'rt_lag_{lag}h'] = df['rt_price'].shift(lag)
    
    # Rolling
    for window in [24, 48, 168]:
        df[f'da_roll_mean_{window}h'] = df['da_price'].rolling(window, min_periods=1).mean()
        df[f'rt_roll_mean_{window}h'] = df['rt_price'].rolling(window, min_periods=1).mean()
    
    # Encode categorical features as numeric codes
    df = encode_categorical_features(df)
    
    return df

def get_feature_columns():
    """Feature columns (no leakage)."""
    return [
        'da_price', 'hour', 'is_weekend', 'month',
        'da_price_level_code', 'abs_residual_bucket_code',
        'period_code',
        'da_negative', 'da_high',
        'da_lag_24h', 'da_lag_48h', 'da_lag_72h', 'da_lag_168h',
        'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
        'da_roll_mean_24h', 'da_roll_mean_48h', 'da_roll_mean_168h',
        'rt_roll_mean_24h', 'rt_roll_mean_48h', 'rt_roll_mean_168h',
    ]

def encode_categorical_features(df):
    """Encode categorical features as numeric codes."""
    df = df.copy()
    
    # Encode categorical columns
    cat_mappings = {
        'da_price_level': {'negative': 0, 'low': 1, 'mid': 2, 'high': 3},
        'abs_residual_bucket': {'small': 0, 'medium': 1, 'large': 2, 'extreme': 3},
        'period': {'p1_8': 0, 'p9_16': 1, 'p17_24': 2},
    }
    
    for col, mapping in cat_mappings.items():
        if col in df.columns:
            code_col = col + '_code'
            # Convert to string first to avoid Categorical issues
            df[code_col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    
    return df

# =============================================================================
# Walk-forward test on 2025
# =============================================================================
def run_2025_test():
    print("=" * 60)
    print("RT-Assist-1 (alpha=1.0) | 2025 Full Year Test")
    print("=" * 60)
    
    # Load data
    data_path = Path('data/preprocessed_data.csv')
    df = pd.read_csv(data_path, parse_dates=['times'])
    df = df.sort_values('times').reset_index(drop=True)
    
    # Add features
    df = add_features(df)
    
    feature_cols = get_feature_columns()
    
    # Filter to rows with non-null rt_price (training targets)
    df_valid = df.dropna(subset=['rt_price'])
    
    # 2025 months
    test_months = [(2025, m) for m in range(1, 13)]
    
    results = []
    
    for year, month in test_months:
        print(f"\n📅 Testing {year}-{month:02d}...")
        
        # Train: all data before this month
        train_mask = (df_valid['times'].dt.year < year) | \
                     ((df_valid['times'].dt.year == year) & (df_valid['times'].dt.month < month))
        train_df = df_valid[train_mask].dropna(subset=feature_cols)
        
        # Test: this month
        test_mask = (df_valid['times'].dt.year == year) & (df_valid['times'].dt.month == month)
        test_df = df_valid[test_mask]
        
        if len(train_df) < 1000:
            print(f"   ⚠️  Not enough training data ({len(train_df)} rows), skipping")
            continue
        
        if len(test_df) == 0:
            print(f"   ⚠️  No test data for {year}-{month:02d}, skipping")
            continue
        
        # Prepare training data
        feature_cols = get_feature_columns()
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df['residual'].values  # predict residual
        
        # Prepare test data
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df['rt_price'].values
        da_anchor = test_df['da_price'].values
        
        # Train model (RandomForest, same as Phase 2-5)
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict residual
        residual_pred = model.predict(X_test)
        
        # Apply alpha=1.0 (full correction, no clip)
        alpha = 1.0
        rt_pred = da_anchor + alpha * residual_pred
        
        # Compute sMAPE
        # Hourly sMAPE
        hourly_smape = compute_smape_floor50(y_test, rt_pred)
        
        # Day-level sMAPE (CRITICAL)
        day_smape = compute_day_level_smape(
            y_test, rt_pred, test_df['times'].values
        )
        
        # DA-only baseline (day-level)
        da_day_smape = compute_day_level_smape(
            y_test, da_anchor, test_df['times'].values
        )
        
        improvement = da_day_smape - day_smape
        
        print(f"   DA-only day sMAPE:  {da_day_smape:.2f}")
        print(f"   RT-Assist day sMAPE: {day_smape:.2f} (improvement: {improvement:+.2f}pp)")
        
        results.append({
            'month': f'{year}-{month:02d}',
            'da_smape': da_day_smape,
            'rt_smape': day_smape,
            'improvement_pp': improvement,
            'n_days': test_df['times'].dt.date.nunique(),
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: 2025 Full Year (Month-level day sMAPE)")
    print("=" * 60)
    
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    
    print(f"\n📊 Average monthly day sMAPE:")
    print(f"   DA-only:  {result_df['da_smape'].mean():.2f}")
    print(f"   RT-Assist: {result_df['rt_smape'].mean():.2f}")
    print(f"   Improvement: {result_df['improvement_pp'].mean():+.2f}pp")
    
    print(f"\n📊 Worst month (day sMAPE):")
    worst = result_df.loc[result_df['rt_smape'].idxmax()]
    print(f"   {worst['month']}: {worst['rt_smape']:.2f}")
    
    print(f"\n📊 Best month (day sMAPE):")
    best = result_df.loc[result_df['rt_smape'].idxmin()]
    print(f"   {best['month']}: {best['rt_smape']:.2f}")
    
    # Check target
    if result_df['rt_smape'].max() < 20:
        print(f"\n✅ TARGET MET: All months < 20 (worst = {result_df['rt_smape'].max():.2f})")
    else:
        print(f"\n⚠️  TARGET NOT MET: Worst month = {result_df['rt_smape'].max():.2f} >= 20")
        over_20 = result_df[result_df['rt_smape'] >= 20]
        print(f"   Months over 20: {len(over_20)}")
        print(f"   {over_20[['month', 'rt_smape']].to_string(index=False)}")
    
    return result_df

if __name__ == '__main__':
    run_2025_test()
