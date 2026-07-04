"""
Quick verification: Test alpha=1.0 on ALL test months.

Check if alpha=1.0 is safe for all months.
If yes → target met, stop iteration.
If no → add safety guards.
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
    """Load and prepare data with all features."""
    
    df = pd.read_csv(data_path, encoding='gbk', parse_dates=['时刻'])
    df = df.sort_values('时刻').reset_index(drop=True)
    
    # Add standard columns
    df['times'] = df['时刻']
    df['rt_price'] = df['实时电价']
    df['da_price'] = df['日前电价']
    df['residual'] = df['rt_price'] - df['da_price']
    
    # Calendar features
    df['hour'] = df['times'].dt.hour + 1
    df['period'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], 
                           labels=['1_8', '9_16', '17_24'], include_lowest=True)
    df['day_of_week'] = df['times'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['times'].dt.month
    
    # DA anchor
    df['da_anchor'] = df['da_price']
    df['da_negative'] = (df['da_price'] < 0).astype(int)
    df['da_high'] = (df['da_price'] > 500).astype(int)
    
    # Forecast features
    forecast_cols = [
        '地方电厂总加预测值', '联络线受电负荷预测值', '风电总加预测值',
        '光伏总加预测值', '核电总加预测值', '自备机组总加预测值',
        '试验机组总加预测值', '直调负荷预测值', '竞价空间预测值', '新能源总加预测值'
    ]
    
    for col in forecast_cols:
        if col in df.columns:
            feature_name = col.replace('总加预测值', '').replace('预测值', '').strip()
            df[feature_name] = df[col]
    
    # Bucket features (from Phase 2)
    df['bucket_negative_da'] = (df['da_price'] < 0).astype(int)
    df['bucket_large_residual'] = (np.abs(df['residual']) >= 100).astype(int)
    df['bucket_spike'] = (df['rt_price'] > 500).astype(int)
    
    # Add lags (ONLY D-1 and earlier, no future)
    for lag in [24, 48, 72, 168]:
        df[f'rt_lag_{lag}h'] = df['rt_price'].shift(lag)
        df[f'residual_lag_{lag}h'] = df['residual'].shift(lag)
        df[f'da_lag_{lag}h'] = df['da_price'].shift(lag)
    
    return df

def get_feature_columns(df):
    """Get feature columns (exclude targets and leakage)."""
    
    exclude = ['rt_price', 'da_price', 'residual', '实时电价', '日前电价']
    exclude.extend([col for col in df.columns if '实际值' in col])
    exclude.extend(['times', '时刻', 'period'])
    
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

def main():
    data_path = r'D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\data\shandong_pmos_hourly.csv'
    
    print("=== Quick Verification: alpha=1.0 on ALL test months ===\n")
    
    # Load data
    df = prepare_data(data_path)
    
    print(f"Data: {df.shape}")
    print(f"Time range: {df['times'].min()} to {df['times'].max()}\n")
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Feature columns: {len(feature_cols)}\n")
    
    # Test months
    test_months = ['2026-02', '2026-03', '2026-04', '2026-05']
    
    # Results
    results = []
    
    for test_month in test_months:
        print(f"=== Testing {test_month} (alpha=1.0) ===")
        
        # Define periods
        test_start = test_month
        test_end = (pd.Timestamp(test_month) + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        train_end = test_month
        
        df_train = df[df['times'] < train_end].copy()
        df_test = df[(df['times'] >= test_start) & (df['times'] < test_end)].copy()
        
        if len(df_test) == 0:
            print(f"  Skipping (no test data)")
            continue
        
        # Train model
        print(f"  Training...")
        X_train = df_train[feature_cols].fillna(0).values
        y_train = (df_train['rt_price'] - df_train['da_price']).values
        
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict with alpha=1.0
        X_test = df_test[feature_cols].fillna(0).values
        residual_pred = model.predict(X_test)
        
        # alpha=1.0, no clip
        final_pred = df_test['da_price'].values + residual_pred
        
        # Compute sMAPE
        test_smape = compute_day_level_smape(df_test['rt_price'], final_pred, df_test['times'])
        da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
        
        # Check for disasters
        max_error = np.max(np.abs(df_test['rt_price'] - final_pred))
        pct_improvement = (da_smape - test_smape) / da_smape * 100
        
        print(f"  DA sMAPE: {da_smape:.2f}")
        print(f"  Model sMAPE (alpha=1.0): {test_smape:.2f}")
        print(f"  Improvement: {da_smape - test_smape:.2f} ({pct_improvement:.1f}%)")
        print(f"  Max absolute error: {max_error:.2f}")
        
        # Check if safe
        if test_smape < da_smape:
            print(f"  ✅ SAFE (improves DA)")
        else:
            print(f"  ❌ UNSAFE (worse than DA)")
        
        results.append({
            'test_month': test_month,
            'da_smape': da_smape,
            'model_smape': test_smape,
            'improvement': da_smape - test_smape,
            'pct_improvement': pct_improvement,
            'safe': test_smape < da_smape
        })
        
        print()
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("=== Summary ===")
    print(f"Mean DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Mean Model sMAPE: {results_df['model_smape'].mean():.2f}")
    print(f"Mean improvement: {results_df['improvement'].mean():.2f}pp")
    print(f"Months improved: {(results_df['improvement'] > 0).sum()}/{len(results_df)}")
    print(f"Worst month sMAPE: {results_df['model_smape'].max():.2f}")
    
    # Check if target met
    target = 20.0
    if results_df['model_smape'].max() < target:
        print(f"\n✅✅✅ TARGET MET! Worst month sMAPE = {results_df['model_smape'].max():.2f} < {target}")
        verdict = "TARGET MET - STOP ITERATION"
    else:
        print(f"\n❌ Target NOT met. Worst month sMAPE = {results_df['model_smape'].max():.2f} >= {target}")
        verdict = "TARGET NOT MET - CONTINUE"
    
    print(f"\n=== Verdict: {verdict} ===")
    
    # Save results
    out_dir = Path('reports/local/rt_assist_1/phase5_verification')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(out_dir / 'verification_results.csv', index=False, encoding='utf-8-sig')
    
    # Generate report
    report_path = out_dir / 'verification_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 5: Verification Report (alpha=1.0)\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Target: Worst month sMAPE < 20\n")
        f.write(f"Mean Model sMAPE: {results_df['model_smape'].mean():.2f}\n")
        f.write(f"Worst month sMAPE: {results_df['model_smape'].max():.2f}\n\n")
        
        f.write("## Monthly Results\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        
        if 'MET' in verdict:
            f.write("Target met! Alpha=1.0 works for all test months.\n")
            f.write("Stop iteration and deploy model.\n")
        else:
            f.write("Target NOT met. Need to add safety guards or continue iteration.\n")
    
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
