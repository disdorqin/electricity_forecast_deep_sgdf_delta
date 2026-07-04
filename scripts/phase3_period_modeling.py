"""
Phase 3: Time-Period Modeling.

Train 3 separate models for 3 periods: 1-8, 9-16, 17-24.
Goal: Improve sMAPE by handling period-specific patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

def prepare_data(data_path):
    """Load and prepare data."""
    
    df = pd.read_csv(data_path, encoding='gbk', parse_dates=['时刻'])
    df = df.sort_values('时刻').reset_index(drop=True)
    
    # Add standard columns
    df['times'] = df['时刻']
    df['rt_price'] = df['实时电价']
    df['da_price'] = df['日前电价']
    df['residual'] = df['rt_price'] - df['da_price']
    
    # Add calendar features
    df['hour'] = df['times'].dt.hour + 1
    df['period'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], 
                           labels=['1_8', '9_16', '17_24'], include_lowest=True)
    df['day_of_week'] = df['times'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['times'].dt.month
    
    # Add DA anchor
    df['da_anchor'] = df['da_price']
    df['da_negative'] = (df['da_price'] < 0).astype(int)
    df['da_high'] = (df['da_price'] > 500).astype(int)
    
    # Add forecast features
    forecast_cols = [
        '地方电厂总加预测值', '联络线受电负荷预测值', '风电总加预测值',
        '光伏总加预测值', '核电总加预测值', '自备机组总加预测值',
        '试验机组总加预测值', '直调负荷预测值', '竞价空间预测值', '新能源总加预测值'
    ]
    
    for col in forecast_cols:
        if col in df.columns:
            feature_name = col.replace('总加预测值', '').replace('预测值', '').strip()
            df[feature_name] = df[col]
    
    # Add bucket features (from Phase 2)
    df['bucket_negative_da'] = (df['da_price'] < 0).astype(int)
    df['bucket_large_residual'] = (np.abs(df['residual']) >= 100).astype(int)
    df['bucket_spike'] = (df['rt_price'] > 500).astype(int)
    
    return df

def get_feature_columns(df):
    """Get feature columns (exclude targets and leakage)."""
    
    exclude = ['rt_price', 'da_price', 'residual', '实时电价', '日前电价']
    exclude.extend([col for col in df.columns if '实际值' in col])
    exclude.extend(['times', '时刻', 'period'])  # period will be one-hot encoded
    
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in ['object', 'category']:
            continue
        if df[col].isna().all():
            continue
        feature_cols.append(col)
    
    return feature_cols

def train_period_model(df_train, period, feature_cols):
    """Train model for a specific period."""
    
    # Filter to period
    df_period = df_train[df_train['period'] == period].copy()
    
    if len(df_period) < 100:
        print(f"  Warning: Only {len(df_period)} rows for period {period}")
        return None
    
    X = df_period[feature_cols].fillna(0).values
    y = (df_period['rt_price'] - df_period['da_price']).values
    
    # Train HGB
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X, y)
    
    return model

def predict_period(model, df_test, period, feature_cols):
    """Predict for a specific period."""
    
    if model is None:
        # No model - return 0 correction
        return np.zeros(len(df_test))
    
    # Filter to period
    df_period = df_test[df_test['period'] == period].copy()
    
    if len(df_period) == 0:
        return np.array([])
    
    X = df_period[feature_cols].fillna(0).values
    residual_pred = model.predict(X)
    
    return residual_pred

def run_phase3_experiment(data_path, out_dir):
    """Run Phase 3 experiment: time-period modeling."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = prepare_data(data_path)
    
    print("=== Phase 3: Time-Period Modeling ===\n")
    print(f"Data: {df.shape}")
    print(f"Time range: {df['times'].min()} to {df['times'].max()}\n")
    
    # Define periods
    periods = ['1_8', '9_16', '17_24']
    
    print(f"Period distribution:")
    for period in periods:
        count = (df['period'] == period).sum()
        print(f"  {period}: {count} rows ({count/len(df)*100:.1f}%)")
    print()
    
    # Define test months
    test_months = ['2026-02', '2026-03', '2026-04', '2026-05']
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Feature columns: {len(feature_cols)}\n")
    
    # === Experiment: Compare single model vs period-specific models ===
    print("=== Experiment: Single vs Period-specific Models ===\n")
    
    results = []
    
    for test_month in test_months:
        print(f"Testing {test_month}...")
        
        # Define periods
        test_start = test_month
        test_end = (pd.Timestamp(test_month) + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        train_end = test_month
        
        df_train = df[df['times'] < train_end].copy()
        df_test = df[(df['times'] >= test_start) & (df['times'] < test_end)].copy()
        
        if len(df_test) == 0:
            print(f"  Skipping (no test data)")
            continue
        
        # Approach 1: Single model (trained on all periods)
        print(f"  Training single model...")
        X_train = df_train[feature_cols].fillna(0).values
        y_train = (df_train['rt_price'] - df_train['da_price']).values
        
        from sklearn.ensemble import HistGradientBoostingRegressor
        single_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        single_model.fit(X_train, y_train)
        
        # Predict with single model
        X_test = df_test[feature_cols].fillna(0).values
        residual_pred_single = single_model.predict(X_test)
        
        # Apply correction (alpha=0.1, clip=20)
        alpha = 0.10
        clip = 20
        correction_single = alpha * np.clip(residual_pred_single, -clip, clip)
        pred_single = df_test['da_price'].values + correction_single
        
        smape_single = compute_day_level_smape(df_test['rt_price'], pred_single, df_test['times'])
        da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
        
        # Approach 2: Period-specific models
        print(f"  Training period-specific models...")
        period_models = {}
        for period in periods:
            model = train_period_model(df_train, period, feature_cols)
            period_models[period] = model
        
        # Predict with period-specific models
        residual_pred_period = np.zeros(len(df_test))
        for period in periods:
            mask = df_test['period'] == period
            if mask.sum() > 0:
                pred = predict_period(period_models[period], df_test, period, feature_cols)
                if len(pred) > 0:
                    residual_pred_period[mask] = pred
        
        # Apply correction
        correction_period = alpha * np.clip(residual_pred_period, -clip, clip)
        pred_period = df_test['da_price'].values + correction_period
        
        smape_period = compute_day_level_smape(df_test['rt_price'], pred_period, df_test['times'])
        
        # DA-only (for reference)
        smape_da = da_smape
        
        results.append({
            'test_month': test_month,
            'da_smape': smape_da,
            'single_model_smape': smape_single,
            'period_model_smape': smape_period,
            'single_improvement': smape_da - smape_single,
            'period_improvement': smape_da - smape_period
        })
        
        print(f"  DA sMAPE: {smape_da:.2f}")
        print(f"  Single model sMAPE: {smape_single:.2f} (improvement: {smape_da - smape_single:.2f})")
        print(f"  Period model sMAPE: {smape_period:.2f} (improvement: {smape_da - smape_period:.2f})")
        print()
    
    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / 'phase3_results.csv', index=False, encoding='utf-8-sig')
    
    print("=== Summary ===")
    print(f"Mean DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Mean single model sMAPE: {results_df['single_model_smape'].mean():.2f}")
    print(f"Mean period model sMAPE: {results_df['period_model_smape'].mean():.2f}")
    print(f"Mean single improvement: {results_df['single_improvement'].mean():.2f}")
    print(f"Mean period improvement: {results_df['period_improvement'].mean():.2f}")
    
    # Check if period-specific models help
    period_improvement = results_df['period_improvement'].mean()
    single_improvement = results_df['single_improvement'].mean()
    
    print(f"\n=== Comparison ===")
    print(f"Single model improvement: {single_improvement:.2f}pp")
    print(f"Period model improvement: {period_improvement:.2f}pp")
    
    if period_improvement > single_improvement + 0.05:
        verdict = "✅ KEEP period-specific models"
        print(f"\n{verdict}")
    elif period_improvement > 0.1:
        verdict = "✅ KEEP period-specific models (moderate improvement)"
        print(f"\n{verdict}")
    else:
        verdict = "❌ KILL period-specific models (no significant improvement)"
        print(f"\n{verdict}")
    
    # Generate report
    report_path = out_dir / 'phase3_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3: Time-Period Modeling Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Test months: {len(results_df)}\n")
        f.write(f"Mean DA sMAPE: {results_df['da_smape'].mean():.2f}\n")
        f.write(f"Mean single model sMAPE: {results_df['single_model_smape'].mean():.2f}\n")
        f.write(f"Mean period model sMAPE: {results_df['period_model_smape'].mean():.2f}\n\n")
        
        f.write("## Results\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        
        if 'KEEP' in verdict:
            f.write("Period-specific models show improvement. Keep them.\n")
        else:
            f.write("Period-specific models do not help significantly. Use single model.\n")
    
    print(f"\nReport saved to {report_path}")
    
    return results_df, verdict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    run_phase3_experiment(args.data_path, args.out_dir)

if __name__ == '__main__':
    main()
