"""
Compute baseline metrics for DeepRT-SOTA v2.

Baselines:
1. DA anchor (day-ahead price)
2. Naive previous day same hour
3. Naive previous 7-day same hour mean
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.metrics import smape_floor50


def compute_baseline_metrics(df: pd.DataFrame):
    """Compute baseline metrics.
    
    Args:
        df: DataFrame with columns ['business_day', 'hour_business', 'rt_actual', 'da_anchor'].
        
    Returns:
        Dictionary of baseline metrics.
    """
    results = {}
    
    # Filter valid rows
    valid_mask = df['rt_actual'].notna() & df['da_anchor'].notna()
    df_valid = df[valid_mask].copy()
    
    print(f"Valid rows for baseline: {len(df_valid)}")
    
    # 1. DA anchor baseline
    da_smape = smape_floor50(df_valid['rt_actual'].values, df_valid['da_anchor'].values)
    da_mae = np.mean(np.abs(df_valid['rt_actual'].values - df_valid['da_anchor'].values))
    
    results['da_anchor'] = {
        'smape_floor50': da_smape,
        'mae': da_mae,
        'description': 'Day-ahead price as prediction',
    }
    
    print(f"\nDA Anchor Baseline:")
    print(f"  sMAPE_floor50: {da_smape:.4f}")
    print(f"  MAE: {da_mae:.4f}")
    
    # 2. Naive previous day same hour
    df_valid = df_valid.sort_values(['business_day', 'hour_business'])
    df_valid['rt_lag_24h'] = df_valid.groupby('hour_business')['rt_actual'].shift(24)
    
    naive_mask = df_valid['rt_lag_24h'].notna()
    naive_valid = df_valid[naive_mask]
    
    if len(naive_valid) > 0:
        naive_smape = smape_floor50(naive_valid['rt_actual'].values, naive_valid['rt_lag_24h'].values)
        naive_mae = np.mean(np.abs(naive_valid['rt_actual'].values - naive_valid['rt_lag_24h'].values))
        
        results['naive_previous_day'] = {
            'smape_floor50': naive_smape,
            'mae': naive_mae,
            'description': 'Previous day same hour',
            'n_samples': len(naive_valid),
        }
        
        print(f"\nNaive Previous Day Baseline:")
        print(f"  sMAPE_floor50: {naive_smape:.4f}")
        print(f"  MAE: {naive_mae:.4f}")
        print(f"  N samples: {len(naive_valid)}")
    
    # 3. Naive previous 7-day same hour mean
    df_valid['rt_lag_168h'] = df_valid.groupby('hour_business')['rt_actual'].shift(168)
    
    # For 7-day mean, we need to compute mean of past 7 days
    # Simplified: use 7-day lag (not mean)
    naive7_mask = df_valid['rt_lag_168h'].notna()
    naive7_valid = df_valid[naive7_mask]
    
    if len(naive7_valid) > 0:
        naive7_smape = smape_floor50(naive7_valid['rt_actual'].values, naive7_valid['rt_lag_168h'].values)
        naive7_mae = np.mean(np.abs(naive7_valid['rt_actual'].values - naive7_valid['rt_lag_168h'].values))
        
        results['naive_previous_7day'] = {
            'smape_floor50': naive7_smape,
            'mae': naive7_mae,
            'description': 'Previous 7-day same hour',
            'n_samples': len(naive7_valid),
        }
        
        print(f"\nNaive Previous 7-Day Baseline:")
        print(f"  sMAPE_floor50: {naive7_smape:.4f}")
        print(f"  MAE: {naive7_mae:.4f}")
        print(f"  N samples: {len(naive7_valid)}")
    
    return results


def main():
    """Main function."""
    # Load data
    data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding='gbk')
    print(f"Loaded {len(df)} rows")
    
    # Rename columns
    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })
    
    # Parse timestamp
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Add business time columns
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    df = add_business_time_columns(df, timestamp_col='ds')
    
    # Compute baselines on test period (2026-02)
    test_mask = (df['ds'] >= '2026-02-01') & (df['ds'] < '2026-03-01')
    test_df = df[test_mask].copy()
    
    print(f"\nTest period (2026-02): {len(test_df)} rows")
    
    # Compute metrics
    results = compute_baseline_metrics(test_df)
    
    # Save results
    import json
    
    output_path = Path('artifacts/deep_rt_sota/baseline_comparison.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable_results = {}
    for key, val in results.items():
        serializable_results[key] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in val.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY (2026-02)")
    print("="*80)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  sMAPE_floor50: {metrics['smape_floor50']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    print("\n" + "="*80)
    print("DeepRT-SOTA v2 TCN (current): sMAPE_floor50 = 42.76")
    print("="*80)


if __name__ == '__main__':
    main()
