"""
Phase 2: Data Classification + Bucket-specific Handling.

Classify data into: normal vs extreme (spikes, negative prices).
Handle different buckets separately.

Goal: Improve sMAPE by handling extreme cases differently.
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

def classify_buckets(df):
    """
    Classify data into buckets.
    
    Buckets:
        0: Normal (DA >= 0, |residual| < 50)
        1: Negative DA (DA < 0)
        2: Large residual (|residual| >= 100)
        3: Spike (RT > 500)
    """
    
    df = df.copy()
    
    # Ensure required columns
    if 'rt_price' not in df.columns:
        df['rt_price'] = df['实时电价']
    if 'da_price' not in df.columns:
        df['da_price'] = df['日前电价']
    if 'residual' not in df.columns:
        df['residual'] = df['rt_price'] - df['da_price']
    
    # Initialize bucket = 0 (normal)
    df['bucket'] = 0
    
    # Bucket 1: Negative DA
    df.loc[df['da_price'] < 0, 'bucket'] = 1
    
    # Bucket 2: Large residual
    df.loc[np.abs(df['residual']) >= 100, 'bucket'] = 2
    
    # Bucket 3: Spike
    df.loc[df['rt_price'] > 500, 'bucket'] = 3
    
    return df

def get_feature_columns(df):
    """Get feature columns (exclude targets and leakage)."""
    
    exclude = ['rt_price', 'da_price', 'residual', '实时电价', '日前电价']
    exclude.extend([col for col in df.columns if '实际值' in col])
    exclude.extend(['times', '时刻', 'bucket'])  # bucket will be one-hot encoded
    
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

def run_phase2_experiment(data_path, out_dir):
    """Run Phase 2 experiment."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path, encoding='gbk', parse_dates=['时刻'])
    df = df.sort_values('时刻').reset_index(drop=True)
    
    # Add standard columns
    df['times'] = df['时刻']
    df['rt_price'] = df['实时电价']
    df['da_price'] = df['日前电价']
    df['residual'] = df['rt_price'] - df['da_price']
    
    print("=== Phase 2: Data Classification + Bucket-specific Handling ===\n")
    print(f"Data: {df.shape}")
    print(f"Time range: {df['times'].min()} to {df['times'].max()}\n")
    
    # Classify buckets
    print("Classifying buckets...")
    df = classify_buckets(df)
    
    print(f"\nBucket distribution:")
    print(df['bucket'].value_counts().sort_index())
    
    bucket_names = {0: 'Normal', 1: 'Negative DA', 2: 'Large residual', 3: 'Spike'}
    
    # Analyze DA error by bucket
    print(f"\n=== DA Error by Bucket ===")
    for bucket in [0, 1, 2, 3]:
        mask = df['bucket'] == bucket
        if mask.sum() > 0:
            df_bucket = df[mask].copy()
            da_smape = compute_day_level_smape(
                df_bucket['rt_price'], 
                df_bucket['da_price'], 
                df_bucket['times']
            )
            print(f"Bucket {bucket} ({bucket_names[bucket]}): {mask.sum()} rows, DA sMAPE = {da_smape:.2f}")
    
    # === Experiment: Bucket-specific correction ===
    print(f"\n=== Experiment: Bucket-specific Correction ===")
    
    # Use 2024-05 to 2025-12 for experiment
    df_exp = df[(df['times'] >= '2024-05-01') & (df['times'] < '2026-01-01')].copy()
    
    print(f"Experiment data: {len(df_exp)} rows\n")
    
    # Approach 1: Bucket as feature (one-hot encode)
    print("Approach 1: Bucket as feature...")
    
    # One-hot encode bucket
    for bucket in [0, 1, 2, 3]:
        df_exp[f'bucket_{bucket}'] = (df_exp['bucket'] == bucket).astype(int)
    
    feature_cols = get_feature_columns(df_exp)
    print(f"  Features (including bucket): {len(feature_cols)}")
    
    # Train model with bucket features
    from sklearn.ensemble import HistGradientBoostingRegressor
    
    X = df_exp[feature_cols].fillna(0).values
    y = df_exp['residual'].values
    
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X, y)
    
    # Evaluate on validation (last 3 months)
    df_val = df_exp[df_exp['times'] >= '2025-10-01'].copy()
    X_val = df_val[feature_cols].fillna(0).values
    residual_pred = model.predict(X_val)
    
    # Apply correction with different strategies
    results = []
    
    for alpha in [0.0, 0.02, 0.05, 0.10]:
        for clip in [0, 10, 20, 30]:
            
            if alpha == 0.0:
                final_pred = df_val['da_price'].values
            else:
                if clip > 0:
                    correction = alpha * np.clip(residual_pred, -clip, clip)
                else:
                    correction = alpha * residual_pred
                final_pred = df_val['da_price'].values + correction
            
            smape = compute_day_level_smape(df_val['rt_price'], final_pred, df_val['times'])
            da_smape = compute_day_level_smape(df_val['rt_price'], df_val['da_price'], df_val['times'])
            
            results.append({
                'approach': 'bucket_as_feature',
                'alpha': alpha,
                'clip': clip,
                'model_smape': smape,
                'da_smape': da_smape,
                'improvement': da_smape - smape
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / 'approach1_bucket_as_feature.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n  Approach 1 results (top 5):")
    print(results_df.sort_values('improvement', ascending=False).head(5)[['alpha', 'clip', 'improvement']])
    
    # Approach 2: Bucket-specific alpha/clip
    print(f"\nApproach 2: Bucket-specific alpha/clip...")
    
    # For each bucket, find best alpha/clip
    bucket_results = []
    
    for bucket in [0, 1, 2, 3]:
        df_bucket = df_val[df_val['bucket'] == bucket].copy()
        
        if len(df_bucket) < 10:
            continue
        
        X_bucket = df_bucket[feature_cols].fillna(0).values
        residual_pred_bucket = model.predict(X_bucket)
        
        best_improvement = -999
        best_alpha = 0.0
        best_clip = 0.0
        
        for alpha in [0.0, 0.02, 0.05, 0.10]:
            for clip in [0, 10, 20, 30]:
                
                if alpha == 0.0:
                    final_pred = df_bucket['da_price'].values
                else:
                    if clip > 0:
                        correction = alpha * np.clip(residual_pred_bucket, -clip, clip)
                    else:
                        correction = alpha * residual_pred_bucket
                    final_pred = df_bucket['da_price'].values + correction
                
                smape = compute_day_level_smape(df_bucket['rt_price'], final_pred, df_bucket['times'])
                da_smape = compute_day_level_smape(df_bucket['rt_price'], df_bucket['da_price'], df_bucket['times'])
                
                improvement = da_smape - smape
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_alpha = alpha
                    best_clip = clip
        
        bucket_results.append({
            'bucket': bucket,
            'bucket_name': bucket_names[bucket],
            'best_alpha': best_alpha,
            'best_clip': best_clip,
            'best_improvement': best_improvement,
            'n_rows': len(df_bucket)
        })
        
        print(f"  Bucket {bucket} ({bucket_names[bucket]}): best improvement = {best_improvement:.4f}, alpha = {best_alpha}, clip = {best_clip}")
    
    bucket_results_df = pd.DataFrame(bucket_results)
    bucket_results_df.to_csv(out_dir / 'approach2_bucket_specific.csv', index=False, encoding='utf-8-sig')
    
    # Evaluate Approach 2 on full validation set
    print(f"\n  Evaluating Approach 2 on full validation set...")
    
    # Apply bucket-specific correction
    final_pred_full = df_val['da_price'].values.copy()
    
    for _, row in bucket_results_df.iterrows():
        bucket = row['bucket']
        alpha = row['best_alpha']
        clip = row['best_clip']
        
        if alpha == 0.0:
            continue  # No correction for this bucket
        
        mask = df_val['bucket'] == bucket
        if mask.sum() == 0:
            continue
        
        X_bucket = df_val.loc[mask, feature_cols].fillna(0).values
        residual_pred_bucket = model.predict(X_bucket)
        correction = alpha * np.clip(residual_pred_bucket, -clip, clip)
        
        final_pred_full[mask] = df_val.loc[mask, 'da_price'].values + correction
    
    smape_full = compute_day_level_smape(df_val['rt_price'], final_pred_full, df_val['times'])
    da_smape_full = compute_day_level_smape(df_val['rt_price'], df_val['da_price'], df_val['times'])
    
    print(f"\n  Approach 2 (bucket-specific) on full validation:")
    print(f"    Model sMAPE = {smape_full:.2f}")
    print(f"    DA sMAPE = {da_smape_full:.2f}")
    print(f"    Improvement = {da_smape_full - smape_full:.2f}")
    
    # === Summary ===
    print(f"\n=== Summary ===")
    
    approach1_best = results_df.sort_values('improvement', ascending=False).iloc[0]
    approach2_improvement = da_smape_full - smape_full
    
    print(f"\nApproach 1 (bucket as feature) best improvement: {approach1_best['improvement']:.4f}")
    print(f"Approach 2 (bucket-specific) improvement: {approach2_improvement:.4f}")
    
    if approach2_improvement > approach1_best['improvement'] and approach2_improvement > 0.1:
        verdict = "✅ KEEP bucket-specific handling"
        print(f"\n{verdict}")
    elif approach1_best['improvement'] > 0.1:
        verdict = "✅ KEEP bucket as feature"
        print(f"\n{verdict}")
    else:
        verdict = "❌ KILL bucket-specific handling (no significant improvement)"
        print(f"\n{verdict}")
    
    # Generate report
    report_path = out_dir / 'phase2_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 2: Data Classification + Bucket-specific Handling\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Bucket distribution:\n")
        for bucket in [0, 1, 2, 3]:
            count = (df['bucket'] == bucket).sum()
            f.write(f"- Bucket {bucket} ({bucket_names[bucket]}): {count} rows ({count/len(df)*100:.1f}%)\n")
        f.write(f"\n")
        
        f.write("## Approach 1: Bucket as Feature\n\n")
        f.write(f"Best improvement: {approach1_best['improvement']:.4f}pp\n\n")
        
        f.write("## Approach 2: Bucket-specific Alpha/Clip\n\n")
        f.write(f"Improvement: {approach2_improvement:.4f}pp\n\n")
        
        f.write("## Bucket-specific Results\n\n")
        f.write(bucket_results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        
        if 'KEEP' in verdict:
            f.write("Bucket-specific handling shows promise. Continue to Phase 3.\n")
        else:
            f.write("No significant improvement from bucket-specific handling. Skip to Phase 3 or try different approach.\n")
    
    print(f"\nReport saved to {report_path}")
    
    return results_df, bucket_results_df, verdict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    run_phase2_experiment(args.data_path, args.out_dir)

if __name__ == '__main__':
    main()
