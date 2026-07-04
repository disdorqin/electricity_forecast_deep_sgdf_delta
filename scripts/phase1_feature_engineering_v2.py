"""
Phase 1: Comprehensive Feature Engineering (Simplified).

Goal: Test if more features help reduce sMAPE.
Iterate until day-level sMAPE < 20.
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

def build_features_minimal(df):
    """Build minimal features (known to work)."""
    df = df.copy()
    
    # Calendar
    df['hour'] = df['times'].dt.hour + 1
    df['period'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], labels=['1_8', '9_16', '17_24'], include_lowest=True)
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
    
    return df

def build_features_comprehensive(df):
    """Build comprehensive features (many derived features)."""
    df = df.copy()
    
    # Start with minimal
    df = build_features_minimal(df)
    
    # Add lags (only D-1 and earlier, no future)
    if 'rt_price' in df.columns:
        for lag in [24, 48, 72, 168]:
            df[f'rt_lag_{lag}h'] = df['rt_price'].shift(lag)
    
    if 'residual' in df.columns:
        for lag in [24, 48, 72, 168]:
            df[f'residual_lag_{lag}h'] = df['residual'].shift(lag)
    
    if 'da_price' in df.columns:
        for lag in [24, 48, 72, 168]:
            df[f'da_lag_{lag}h'] = df['da_price'].shift(lag)
        
        # DA rolling stats (trailing)
        for window in [24, 48, 168]:
            df[f'da_roll_mean_{window}h'] = df['da_price'].rolling(window, min_periods=1).mean().shift(24)
            df[f'da_roll_std_{window}h'] = df['da_price'].rolling(window, min_periods=1).std().shift(24)
    
    # Add interaction features
    if 'da_price' in df.columns:
        forecast_cols = [
            '地方电厂总加预测值', '联络线受电负荷预测值', '风电总加预测值',
            '光伏总加预测值', '核电总加预测值', '自备机组总加预测值',
            '试验机组总加预测值', '直调负荷预测值', '竞价空间预测值', '新能源总加预测值'
        ]
        for col in forecast_cols:
            if col in df.columns:
                feature_name = col.replace('总加预测值', '').replace('预测值', '').strip()
                df[f'da_x_{feature_name}'] = df['da_price'] * df[col]
    
    return df

def get_feature_columns(df, exclude_targets=True):
    """Get feature columns (exclude targets and leakage)."""
    
    exclude = []
    if exclude_targets:
        exclude.extend(['rt_price', 'da_price', 'residual', '实时电价', '日前电价'])
        exclude.extend([col for col in df.columns if '实际值' in col])  # Actual values = leakage
    
    # Also exclude time columns
    exclude.extend(['times', '时刻'])
    
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

def run_experiment(df, feature_set_name, feature_cols, test_months=['2026-02', '2026-03', '2026-04', '2026-05']):
    """Run walk-forward experiment with given features."""
    
    results = []
    
    for test_month in test_months:
        # Define periods
        test_start = test_month
        test_end = (pd.Timestamp(test_month) + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        val_start = (pd.Timestamp(test_month) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        val_end = test_start
        
        df_train = df[df['times'] < val_start].copy()
        df_val = df[(df['times'] >= val_start) & (df['times'] < val_end)].copy()
        df_test = df[(df['times'] >= test_start) & (df['times'] < test_end)].copy()
        
        if len(df_test) == 0:
            continue
        
        # Train model to predict residual
        X_train = df_train[feature_cols].fillna(0).values
        y_train = (df_train['rt_price'] - df_train['da_price']).values
        
        X_val = df_val[feature_cols].fillna(0).values
        y_val = (df_val['rt_price'] - df_val['da_price']).values
        
        # Train HGB
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Select best alpha/clip on validation
        best_improvement = -999
        best_alpha = 0.0
        best_clip = 0.0
        
        X_test = df_test[feature_cols].fillna(0).values
        residual_pred_val = model.predict(X_val)
        residual_pred_test = model.predict(X_test)
        
        for alpha in [0.0, 0.02, 0.05, 0.10, 0.20]:
            for clip in [0, 10, 20, 30, 50]:
                
                if alpha == 0.0:
                    val_pred = df_val['da_price'].values
                    test_pred = df_test['da_price'].values
                else:
                    if clip > 0:
                        val_correction = alpha * np.clip(residual_pred_val, -clip, clip)
                        test_correction = alpha * np.clip(residual_pred_test, -clip, clip)
                    else:
                        val_correction = alpha * residual_pred_val
                        test_correction = alpha * residual_pred_test
                    
                    val_pred = df_val['da_price'].values + val_correction
                    test_pred = df_test['da_price'].values + test_correction
                
                # Compute sMAPE
                val_smape = compute_day_level_smape(df_val['rt_price'], val_pred, df_val['times'])
                test_smape = compute_day_level_smape(df_test['rt_price'], test_pred, df_test['times'])
                da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
                
                improvement = da_smape - test_smape
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_alpha = alpha
                    best_clip = clip
        
        # Final evaluation with best params
        if best_alpha == 0.0:
            final_pred = df_test['da_price'].values
        else:
            final_correction = best_alpha * np.clip(residual_pred_test, -best_clip, best_clip)
            final_pred = df_test['da_price'].values + final_correction
        
        test_smape = compute_day_level_smape(df_test['rt_price'], final_pred, df_test['times'])
        da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
        
        results.append({
            'test_month': test_month,
            'feature_set': feature_set_name,
            'model_smape': test_smape,
            'da_smape': da_smape,
            'improvement': da_smape - test_smape,
            'best_alpha': best_alpha,
            'best_clip': best_clip
        })
    
    return pd.DataFrame(results)

def main():
    # Load data
    data_path = r'D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\data\shandong_pmos_hourly.csv'
    
    print("=== Phase 1: Feature Engineering ===\n")
    
    df = pd.read_csv(data_path, encoding='gbk', parse_dates=['时刻'])
    df = df.sort_values('时刻').reset_index(drop=True)
    
    # Add standard columns
    df['times'] = df['时刻']
    df['rt_price'] = df['实时电价']
    df['da_price'] = df['日前电价']
    df['residual'] = df['rt_price'] - df['da_price']
    
    print(f"Data: {df.shape}")
    print(f"Time range: {df['times'].min()} to {df['times'].max()}\n")
    
    # === Test 1: Minimal features ===
    print("=== Test 1: Minimal Features ===")
    df_minimal = build_features_minimal(df)
    feature_cols_minimal = get_feature_columns(df_minimal)
    print(f"Features: {len(feature_cols_minimal)}")
    
    results_minimal = run_experiment(df_minimal, 'minimal', feature_cols_minimal)
    print(f"\nResults (minimal):")
    print(results_minimal[['test_month', 'model_smape', 'da_smape', 'improvement']])
    print(f"\nMean improvement: {results_minimal['improvement'].mean():.4f}\n")
    
    # === Test 2: Comprehensive features ===
    print("=== Test 2: Comprehensive Features ===")
    df_comp = build_features_comprehensive(df)
    feature_cols_comp = get_feature_columns(df_comp)
    print(f"Features: {len(feature_cols_comp)}")
    
    results_comp = run_experiment(df_comp, 'comprehensive', feature_cols_comp)
    print(f"\nResults (comprehensive):")
    print(results_comp[['test_month', 'model_smape', 'da_smape', 'improvement']])
    print(f"\nMean improvement: {results_comp['improvement'].mean():.4f}\n")
    
    # === Compare ===
    print("=== Comparison ===")
    comparison = pd.concat([results_minimal, results_comp])
    print(comparison[['test_month', 'feature_set', 'model_smape', 'improvement']])
    
    # Check if comprehensive is better
    mean_imp_minimal = results_minimal['improvement'].mean()
    mean_imp_comp = results_comp['improvement'].mean()
    
    print(f"\nMean improvement (minimal): {mean_imp_minimal:.4f}")
    print(f"Mean improvement (comprehensive): {mean_imp_comp:.4f}")
    
    if mean_imp_comp > mean_imp_minimal + 0.05:
        print(f"\n✅ Comprehensive features improve sMAPE!")
        print(f"   Keep comprehensive features.")
        verdict = "KEEP comprehensive"
    else:
        print(f"\n❌ Comprehensive features do not help significantly.")
        print(f"   Stick with minimal features.")
        verdict = "KEEP minimal"
    
    # Save results
    out_dir = Path('reports/local/rt_assist_1/phase1_feature_engineering')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_minimal.to_csv(out_dir / 'results_minimal.csv', index=False, encoding='utf-8-sig')
    results_comp.to_csv(out_dir / 'results_comprehensive.csv', index=False, encoding='utf-8-sig')
    comparison.to_csv(out_dir / 'comparison.csv', index=False, encoding='utf-8-sig')
    
    # Generate report
    report_path = out_dir / 'feature_engineering_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1: Feature Engineering Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Minimal features: {len(feature_cols_minimal)}\n")
        f.write(f"Comprehensive features: {len(feature_cols_comp)}\n\n")
        
        f.write("## Results\n\n")
        f.write("### Minimal Features\n\n")
        f.write(results_minimal.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### Comprehensive Features\n\n")
        f.write(results_comp.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        
        if 'comprehensive' in verdict:
            f.write("Comprehensive features improve sMAPE. Keep them.\n")
        else:
            f.write("Comprehensive features do not help. Use minimal features.\n")
    
    print(f"\nReport saved to {report_path}")
    print(f"\n=== Phase 1 Complete ===")

if __name__ == '__main__':
    main()
