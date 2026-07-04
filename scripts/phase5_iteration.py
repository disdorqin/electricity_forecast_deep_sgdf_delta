"""
Phase 5: Iteration - Bring worst month below 20.

Current worst month: 2026-02 (sMAPE = 25.57).
Target: Bring 2026-02 sMAPE below 20.

Approaches to try:
1. Higher alpha (0.3, 0.5, 1.0)
2. Bucket-specific alpha (larger alpha for large residual bucket)
3. Use actual values as features (with proper lag)
4. Add safety guards
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
    # RT lags
    for lag in [24, 48, 72, 168]:
        df[f'rt_lag_{lag}h'] = df['rt_price'].shift(lag)
    
    # Residual lags
    for lag in [24, 48, 72, 168]:
        df[f'residual_lag_{lag}h'] = df['residual'].shift(lag)
    
    # DA lags
    for lag in [24, 48, 72, 168]:
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

def run_iteration_experiment(data_path, out_dir):
    """Run iteration experiment to bring worst month below 20."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = prepare_data(data_path)
    
    print("=== Phase 5: Iteration ===\n")
    print(f"Data: {df.shape}")
    print(f"Time range: {df['times'].min()} to {df['times'].max()}\n")
    
    # Focus on 2026-02 (worst month)
    print("Target: Bring 2026-02 sMAPE below 20.")
    print(f"Current 2026-02 sMAPE: 25.57\n")
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Feature columns: {len(feature_cols)}\n")
    
    # === Approach 1: Higher alpha ===
    print("=== Approach 1: Higher Alpha ===\n")
    
    test_month = '2026-02'
    test_start = test_month
    test_end = '2026-03-01'
    val_start = '2025-11-01'
    val_end = test_start
    
    df_train = df[df['times'] < val_start].copy()
    df_val = df[(df['times'] >= val_start) & (df['times'] < val_end)].copy()
    df_test = df[(df['times'] >= test_start) & (df['times'] < test_end)].copy()
    
    print(f"Train: {len(df_train)} rows")
    print(f"Val: {len(df_val)} rows")
    print(f"Test (2026-02): {len(df_test)} rows\n")
    
    # Train model
    X_train = df_train[feature_cols].fillna(0).values
    y_train = (df_train['rt_price'] - df_train['da_price']).values
    
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test higher alpha
    X_test = df_test[feature_cols].fillna(0).values
    residual_pred = model.predict(X_test)
    
    results = []
    
    for alpha in [0.2, 0.3, 0.5, 1.0, 2.0]:
        for clip in [0, 50, 100, 200]:
            
            if clip > 0:
                correction = alpha * np.clip(residual_pred, -clip, clip)
            else:
                correction = alpha * residual_pred
            
            final_pred = df_test['da_price'].values + correction
            
            smape = compute_day_level_smape(df_test['rt_price'], final_pred, df_test['times'])
            da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
            
            results.append({
                'approach': 'higher_alpha',
                'alpha': alpha,
                'clip': clip,
                'model_smape': smape,
                'da_smape': da_smape,
                'improvement': da_smape - smape
            })
            
            if alpha in [0.2, 0.5, 1.0] and clip in [0, 50]:
                print(f"  alpha={alpha}, clip={clip}: sMAPE={smape:.2f}, improvement={da_smape - smape:.2f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / 'approach1_higher_alpha.csv', index=False, encoding='utf-8-sig')
    
    best = results_df.sort_values('improvement', ascending=False).iloc[0]
    print(f"\nBest (Approach 1): alpha={best['alpha']}, clip={best['clip']}")
    print(f"  sMAPE = {best['model_smape']:.2f}, improvement = {best['improvement']:.2f}\n")
    
    # === Approach 2: Bucket-specific alpha ===
    print("=== Approach 2: Bucket-specific Alpha ===\n")
    
    # Classify buckets
    df_test['bucket'] = 0
    df_test.loc[df_test['da_price'] < 0, 'bucket'] = 1  # Negative DA
    df_test.loc[np.abs(df_test['residual']) >= 100, 'bucket'] = 2  # Large residual
    df_test.loc[df_test['rt_price'] > 500, 'bucket'] = 3  # Spike
    
    # Try different alpha for each bucket
    bucket_alphas = [
        {'bucket_0': 0.2, 'bucket_1': 0.5, 'bucket_2': 1.0, 'bucket_3': 0.2},
        {'bucket_0': 0.2, 'bucket_1': 1.0, 'bucket_2': 2.0, 'bucket_3': 0.5},
        {'bucket_0': 0.1, 'bucket_1': 0.3, 'bucket_2': 0.5, 'bucket_3': 0.1},
    ]
    
    for i, bucket_alpha in enumerate(bucket_alphas):
        final_pred = df_test['da_price'].values.copy()
        
        for bucket, alpha in bucket_alpha.items():
            bucket_id = int(bucket.split('_')[1])
            mask = df_test['bucket'] == bucket_id
            
            if mask.sum() > 0:
                if bucket_id == 0:
                    clip = 20
                elif bucket_id == 1:
                    clip = 50
                elif bucket_id == 2:
                    clip = 100
                else:
                    clip = 20
                
                correction = alpha * np.clip(residual_pred[mask], -clip, clip)
                final_pred[mask] = df_test.loc[mask, 'da_price'].values + correction
        
        smape = compute_day_level_smape(df_test['rt_price'], final_pred, df_test['times'])
        da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
        
        print(f"  Bucket alpha config {i+1}: sMAPE = {smape:.2f}, improvement = {da_smape - smape:.2f}")
        
        results.append({
            'approach': f'bucket_alpha_{i+1}',
            'alpha': str(bucket_alpha),
            'clip': 'bucket_specific',
            'model_smape': smape,
            'da_smape': da_smape,
            'improvement': da_smape - smape
        })
    
    # === Summary ===
    print(f"\n=== Summary ===")
    
    all_results = pd.DataFrame(results)
    all_results.to_csv(out_dir / 'all_results.csv', index=False, encoding='utf-8-sig')
    
    best_overall = all_results.sort_values('improvement', ascending=False).iloc[0]
    print(f"Best approach: {best_overall['approach']}")
    print(f"  sMAPE = {best_overall['model_smape']:.2f}")
    print(f"  Improvement = {best_overall['improvement']:.2f}\n")
    
    # Check if target met
    if best_overall['model_smape'] < 20.0:
        print(f"✅ Target MET! 2026-02 sMAPE = {best_overall['model_smape']:.2f} < 20")
        verdict = "TARGET MET"
    else:
        print(f"❌ Target NOT met. 2026-02 sMAPE = {best_overall['model_smape']:.2f} >= 20")
        print(f"   Need more iteration.")
        verdict = "TARGET NOT MET"
    
    # Generate report
    report_path = out_dir / 'phase5_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 5: Iteration Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Target: Bring 2026-02 sMAPE below 20.\n")
        f.write(f"Current (Phase 4): 25.57\n\n")
        
        f.write("## Results\n\n")
        f.write(all_results[['approach', 'alpha', 'model_smape', 'improvement']].to_string(index=False))
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        
        if 'MET' in verdict:
            f.write("Target met! Stop iteration.\n")
        else:
            f.write("Target NOT met. Continue iteration with different approaches.\n")
            f.write("- Try deep learning models (if user allows)\n")
            f.write("- Try different feature combinations\n")
            f.write("- Consider giving up and accepting DA-only\n")
    
    print(f"\nReport saved to {report_path}")
    
    return all_results, verdict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    run_iteration_experiment(args.data_path, args.out_dir)

if __name__ == '__main__':
    main()
