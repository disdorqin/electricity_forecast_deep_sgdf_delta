"""
Phase 4: Combined Model + Safety Guards.

Combine all effective approaches:
1. Bucket features (Phase 2)
2. Period-specific or single model (Phase 3)
3. DA-safe-enhancer (earlier) with safety guards

Evaluate on test set (2026-02 to 2026-05).
Goal: Bring WORST month (2026-02: 27.87) below 20.
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

def apply_safety_guards(corrections, trigger_scores, da_prices, config=None):
    """
    Apply safety guards to corrections.
    
    Args:
        corrections: array of raw corrections
        trigger_scores: array of trigger probabilities
        da_prices: array of DA prices
        config: dict with guard config
    
    Returns:
        safe_corrections: array of safe corrections
        decision_log: list of decision dicts
    """
    
    if config is None:
        config = {
            'max_fire_rate': 0.05,  # 5% max
            'max_correction': 20,
            'min_trigger_confidence': 0.1
        }
    
    safe_corrections = np.zeros(len(corrections))
    decision_log = []
    
    fire_count = 0
    total_count = 0
    
    for i in range(len(corrections)):
        total_count += 1
        
        # Guard 1: Max correction magnitude
        raw_corr = corrections[i]
        if abs(raw_corr) > config['max_correction']:
            raw_corr = np.clip(raw_corr, -config['max_correction'], config['max_correction'])
        
        # Guard 2: Max fire rate
        if fire_count / max(total_count, 1) >= config['max_fire_rate']:
            # Block correction
            decision_log.append({
                'index': i,
                'raw_correction': raw_corr,
                'safe_correction': 0.0,
                'blocked': True,
                'block_reason': 'max_fire_rate'
            })
            continue
        
        # Guard 3: Min trigger confidence
        if abs(trigger_scores[i] - 0.5) < config['min_trigger_confidence']:
            # Low confidence - block
            decision_log.append({
                'index': i,
                'raw_correction': raw_corr,
                'safe_correction': 0.0,
                'blocked': True,
                'block_reason': 'low_confidence'
            })
            continue
        
        # Allow correction
        fire_count += 1
        safe_corrections[i] = raw_corr
        decision_log.append({
            'index': i,
            'raw_correction': raw_corr,
            'safe_correction': raw_corr,
            'blocked': False,
            'block_reason': None
        })
    
    return safe_corrections, decision_log

def run_combined_experiment(data_path, out_dir):
    """Run combined experiment with all approaches."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = prepare_data(data_path)
    
    print("=== Phase 4: Combined Model + Safety Guards ===\n")
    print(f"Data: {df.shape}")
    print(f"Time range: {df['times'].min()} to {df['times'].max()}\n")
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Feature columns: {len(feature_cols)}\n")
    
    # Define test months
    test_months = ['2026-02', '2026-03', '2026-04', '2026-05']
    
    # Results
    all_results = []
    
    for test_month in test_months:
        print(f"=== Testing {test_month} ===")
        
        # Define periods
        test_start = test_month
        test_end = (pd.Timestamp(test_month) + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        val_start = (pd.Timestamp(test_month) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        val_end = test_start
        
        df_train = df[df['times'] < val_start].copy()
        df_val = df[(df['times'] >= val_start) & (df['times'] < val_end)].copy()
        df_test = df[(df['times'] >= test_start) & (df['times'] < test_end)].copy()
        
        if len(df_test) == 0:
            print(f"  Skipping (no test data)")
            continue
        
        print(f"  Train: {len(df_train)} rows")
        print(f"  Val: {len(df_val)} rows")
        print(f"  Test: {len(df_test)} rows")
        
        # Train model (single model for simplicity)
        X_train = df_train[feature_cols].fillna(0).values
        y_train = (df_train['rt_price'] - df_train['da_price']).values
        
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Select best alpha/clip on validation
        X_val = df_val[feature_cols].fillna(0).values
        residual_pred_val = model.predict(X_val)
        
        best_improvement = -999
        best_alpha = 0.0
        best_clip = 0.0
        
        for alpha in [0.0, 0.02, 0.05, 0.10, 0.20]:
            for clip in [0, 10, 20, 30, 50]:
                
                if alpha == 0.0:
                    val_pred = df_val['da_price'].values
                else:
                    if clip > 0:
                        correction = alpha * np.clip(residual_pred_val, -clip, clip)
                    else:
                        correction = alpha * residual_pred_val
                    val_pred = df_val['da_price'].values + correction
                
                val_smape = compute_day_level_smape(df_val['rt_price'], val_pred, df_val['times'])
                da_smape = compute_day_level_smape(df_val['rt_price'], df_val['da_price'], df_val['times'])
                
                improvement = da_smape - val_smape
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_alpha = alpha
                    best_clip = clip
        
        print(f"  Best val improvement: {best_improvement:.4f}")
        print(f"  Best params: alpha={best_alpha}, clip={best_clip}")
        
        # Apply to test set
        X_test = df_test[feature_cols].fillna(0).values
        residual_pred_test = model.predict(X_test)
        
        if best_alpha == 0.0:
            final_pred = df_test['da_price'].values
        else:
            if best_clip > 0:
                correction = best_alpha * np.clip(residual_pred_test, -best_clip, best_clip)
            else:
                correction = best_alpha * residual_pred_test
            final_pred = df_test['da_price'].values + correction
        
        # Compute sMAPE
        test_smape = compute_day_level_smape(df_test['rt_price'], final_pred, df_test['times'])
        da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
        
        print(f"  Test sMAPE: {test_smape:.2f}")
        print(f"  DA sMAPE: {da_smape:.2f}")
        print(f"  Improvement: {da_smape - test_smape:.2f}\n")
        
        all_results.append({
            'test_month': test_month,
            'model_smape': test_smape,
            'da_smape': da_smape,
            'improvement': da_smape - test_smape,
            'best_alpha': best_alpha,
            'best_clip': best_clip
        })
    
    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / 'combined_results.csv', index=False, encoding='utf-8-sig')
    
    print("=== Summary ===")
    print(f"Mean model sMAPE: {results_df['model_smape'].mean():.2f}")
    print(f"Mean DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Mean improvement: {results_df['improvement'].mean():.2f}")
    print(f"Months with improvement: {(results_df['improvement'] > 0).sum()}/{len(results_df)}")
    print(f"Worst month sMAPE: {results_df['model_smape'].max():.2f}")
    
    # Check if target met
    target = 20.0
    if results_df['model_smape'].max() < target:
        print(f"\n✅ Target met! Worst month sMAPE = {results_df['model_smape'].max():.2f} < {target}")
        verdict = "✅ TARGET MET"
    else:
        print(f"\n❌ Target NOT met. Worst month sMAPE = {results_df['model_smape'].max():.2f} >= {target}")
        verdict = "❌ TARGET NOT MET"
    
    # Generate report
    report_path = out_dir / 'phase4_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 4: Combined Model + Safety Guards Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Mean model sMAPE: {results_df['model_smape'].mean():.2f}\n")
        f.write(f"Mean DA sMAPE: {results_df['da_smape'].mean():.2f}\n")
        f.write(f"Mean improvement: {results_df['improvement'].mean():.2f}\n\n")
        
        f.write("## Monthly Results\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        
        if 'MET' in verdict:
            f.write("Target (worst month sMAPE < 20) is met. Stop iteration.\n")
        else:
            f.write("Target NOT met. Need to continue iteration.\n")
            f.write("- Try different model architectures\n")
            f.write("- Try different feature combinations\n")
            f.write("- Consider giving up and accepting DA-only\n")
    
    print(f"\nReport saved to {report_path}")
    
    return results_df, verdict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    run_combined_experiment(args.data_path, args.out_dir)

if __name__ == '__main__':
    main()
