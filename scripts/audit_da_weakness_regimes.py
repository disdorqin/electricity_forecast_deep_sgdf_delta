"""
Phase A: DA Weakness / Regime Audit

目标：
找出DA anchor在哪些条件下误差最大，以及这些条件是否可预测。

输出：
- regime_inventory.csv
- da_error_by_regime.csv
- regime_predictability.csv
- da_weakness_report.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def compute_smape_floor50(y_true, y_pred, floor=50.0):
    """Compute sMAPE with floor=50 for denominator."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clean data: remove NaN and infinite values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, floor)
    return 100 * np.mean(np.abs(y_true - y_pred) / denom)

def compute_day_level_smape(y_true, y_pred, timestamps):
    """Compute day-level sMAPE."""
    # Clean data
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN and infinite values
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(timestamps)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    timestamps = timestamps[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'y_true': y_true,
        'y_pred': y_pred
    })
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    daily_true = df.groupby('date')['y_true'].mean()
    daily_pred = df.groupby('date')['y_pred'].mean()
    
    return compute_smape_floor50(daily_true.values, daily_pred.values)

def create_regime_features(df):
    """Create regime feature columns."""
    
    # 1. period (1-8, 9-16, 17-24)
    df['period'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], 
                          labels=['1_8', '9_16', '17_24'], 
                          include_lowest=True, right=False)
    
    # 2. hour bucket (each hour 1-24)
    df['hour_bucket'] = df['hour'].apply(lambda x: f"h{x:02d}")
    
    # 3. DA price level
    da_quantile = df['da_price'].quantile([0.25, 0.5, 0.75])
    df['da_quantile_level'] = pd.cut(df['da_price'], 
                                      bins=[-np.inf, da_quantile[0.25], da_quantile[0.5], da_quantile[0.75], np.inf],
                                      labels=['da_quantile_low', 'da_quantile_mid', 'da_quantile_high', 'da_quantile_very_high'],
                                      include_lowest=True)
    df['da_negative'] = (df['da_price'] < 0).astype(str)
    df['da_high_above_500'] = (df['da_price'] > 500).astype(str)
    
    # 4. calendar
    df['weekday'] = df['hour'].apply(lambda x: 'weekday' if x < 5 else 'weekend')
    df['month'] = df['times'].dt.month
    
    # 5. target buckets (ONLY for evaluation, NOT for online trigger)
    df['target_negative_rt'] = (df['rt_price'] < 0).astype(str)
    df['target_spike_rt'] = (df['rt_price'] > 1000).astype(str)  # Spike defined as >1000
    df['target_large_abs_delta'] = (np.abs(df['residual']) > 200).astype(str)
    df['target_normal'] = (~(df['target_negative_rt'] == 'True') & 
                          ~(df['target_spike_rt'] == 'True') & 
                          ~(df['target_large_abs_delta'] == 'True')).astype(str)
    
    # 6. online-available proxy buckets
    df['online_high_da'] = (df['da_price'] > df['da_price'].quantile(0.75)).astype(str)
    df['online_low_da'] = (df['da_price'] < df['da_price'].quantile(0.25)).astype(str)
    
    # DA volatility (past 7 days)
    df['da_volatility_prev7d'] = df['da_price'].rolling(window=168, min_periods=1).std().shift(1)
    df['online_da_vol_high'] = (df['da_volatility_prev7d'] > df['da_volatility_prev7d'].quantile(0.75)).astype(str)
    
    # Prior day residual
    df['prior_day_residual_abs'] = np.abs(df['residual']).shift(24)
    df['online_prior_day_residual_abs_high'] = (df['prior_day_residual_abs'] > df['prior_day_residual_abs'].quantile(0.75)).astype(str)
    
    # Prior 7d residual volatility
    df['prior_7d_residual_vol'] = np.abs(df['residual']).rolling(window=168, min_periods=1).std().shift(1)
    df['online_prior_7d_residual_vol_high'] = (df['prior_7d_residual_vol'] > df['prior_7d_residual_vol'].quantile(0.75)).astype(str)
    
    # Period + DA level combo
    df['online_period_da_combo'] = df['period'].astype(str) + '_' + df['da_quantile_level'].astype(str)
    
    return df

def audit_da_weakness(df, regimes, out_dir):
    """
    Audit DA weakness by regime.
    
    Args:
        df: DataFrame with features and targets
        regimes: List of regime column names
        out_dir: Output directory
    """
    results = []
    
    # Compute global DA sMAPE
    global_da_smape = compute_day_level_smape(df['rt_price'].values, 
                                              df['da_price'].values, 
                                              df['times'])
    print(f"\nGlobal DA sMAPE (day-level): {global_da_smape:.2f}")
    
    for regime in regimes:
        if regime not in df.columns:
            print(f"Warning: {regime} not in columns, skipping...")
            continue
        
        # Group by regime
        grouped = df.groupby(regime)
        
        for regime_value, group in grouped:
            if len(group) < 10:  # Skip small regimes
                continue
            
            # Compute DA error metrics
            da_smape = compute_day_level_smape(group['rt_price'].values, 
                                                group['da_price'].values, 
                                                group['times'])
            
            da_mae = np.mean(np.abs(group['rt_price'].values - group['da_price'].values))
            mean_abs_residual = np.mean(np.abs(group['residual'].values))
            residual_std = np.std(group['residual'].values)
            
            # Check if online-available
            online_available = not regime.startswith('target_')
            
            # Compute priority
            n_rows = len(group)
            smape_excess = da_smape - global_da_smape
            
            if online_available and n_rows >= 50 and smape_excess >= 5:
                priority = 'HIGH_PRIORITY'
            elif n_rows >= 30 and smape_excess >= 3:
                priority = 'MEDIUM_PRIORITY'
            else:
                priority = 'KILL'
            
            results.append({
                'regime': regime,
                'regime_value': str(regime_value),
                'n_rows': n_rows,
                'coverage': n_rows / len(df),
                'da_smape_floor50': da_smape,
                'da_mae': da_mae,
                'mean_abs_residual': mean_abs_residual,
                'residual_std': residual_std,
                'whether_online_available': online_available,
                'candidate_priority': priority,
                'smape_excess_vs_global': smape_excess
            })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Sort by priority and sMAPE excess
    priority_order = {'HIGH_PRIORITY': 0, 'MEDIUM_PRIORITY': 1, 'KILL': 2}
    results_df['priority_order'] = results_df['candidate_priority'].map(priority_order)
    results_df = results_df.sort_values(['priority_order', 'smape_excess_vs_global'], 
                                       ascending=[True, False]).drop('priority_order', axis=1)
    
    # Save
    results_df.to_csv(out_dir / 'da_error_by_regime.csv', index=False, encoding='utf-8-sig')
    
    # Also save regime inventory
    regime_inventory = results_df[['regime', 'regime_value', 'n_rows', 'coverage', 
                                   'da_smape_floor50', 'whether_online_available', 
                                   'candidate_priority']].copy()
    regime_inventory.to_csv(out_dir / 'regime_inventory.csv', index=False, encoding='utf-8-sig')
    
    # Save predictability (dummy for now, will be updated in Phase B)
    predictability = results_df[['regime', 'regime_value', 'whether_online_available', 
                                'candidate_priority']].copy()
    predictability['predictability_score'] = np.nan  # To be filled in Phase B
    predictability.to_csv(out_dir / 'regime_predictability.csv', index=False, encoding='utf-8-sig')
    
    return results_df

def generate_report(results_df, global_da_smape, out_dir):
    """Generate audit report."""
    report_path = out_dir / 'da_weakness_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DA Weakness Regime Audit Report\n\n")
        f.write(f"## Global DA sMAPE (day-level): {global_da_smape:.2f}\n\n")
        
        f.write("## Top 10 High-Priority Regimes\n\n")
        high_priority = results_df[results_df['candidate_priority'] == 'HIGH_PRIORITY'].head(10)
        if len(high_priority) > 0:
            # Write as table
            f.write("| Regime | Regime Value | N Rows | Coverage | DA sMAPE | Excess vs Global | Priority |\n")
            f.write("|---------|---------------|---------|----------|----------|-------------------|----------|\n")
            for _, row in high_priority.iterrows():
                f.write(f"| {row['regime']} | {row['regime_value']} | {row['n_rows']} | {row['coverage']:.2%} | {row['da_smape_floor50']:.2f} | {row['smape_excess_vs_global']:.2f} | {row['candidate_priority']} |\n")
            f.write("\n")
        else:
            f.write("No high-priority regimes found.\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total regimes analyzed: {len(results_df)}\n")
        f.write(f"- High-priority regimes: {len(results_df[results_df['candidate_priority'] == 'HIGH_PRIORITY'])}\n")
        f.write(f"- Medium-priority regimes: {len(results_df[results_df['candidate_priority'] == 'MEDIUM_PRIORITY'])}\n")
        f.write(f"- KILL regimes: {len(results_df[results_df['candidate_priority'] == 'KILL'])}\n")
        f.write(f"- Online-available regimes: {len(results_df[results_df['whether_online_available'] == True])}\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("Proceed to Phase B: Train trigger model to predict DA errors.\n")
    
    print(f"\nReport saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Phase A: DA Weakness / Regime Audit')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to shandong_pmos_hourly.csv')
    parser.add_argument('--target-months', type=str, 
                       default='2026-01,2026-02,2026-03,2026-04,2026-05',
                       help='Target months for evaluation (comma-separated)')
    parser.add_argument('--out-dir', type=str, required=True,
                       help='Output directory')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, parse_dates=['时刻'], encoding='gbk')
    df = df.sort_values('时刻').reset_index(drop=True)
    
    # Rename columns
    col_map = {
        '时刻': 'times',
        '日前电价': 'da_price',
        '实时电价': 'rt_price',
    }
    df = df.rename(columns=col_map)
    
    # Compute residual
    df['residual'] = df['rt_price'] - df['da_price']
    
    # Create time features
    df['hour'] = df['times'].dt.hour
    df['dayofweek'] = df['times'].dt.dayofweek
    
    # Create regime features
    print("Creating regime features...")
    df = create_regime_features(df)
    
    # Define regimes to audit
    regimes = [
        'period',
        'hour_bucket',
        'da_quantile_level',
        'da_negative',
        'da_high_above_500',
        'weekday',
        'month',
        'target_negative_rt',
        'target_spike_rt',
        'target_large_abs_delta',
        'target_normal',
        'online_high_da',
        'online_low_da',
        'online_da_vol_high',
        'online_prior_day_residual_abs_high',
        'online_prior_7d_residual_vol_high',
        'online_period_da_combo'
    ]
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run audit
    print("\n=== Running DA Weakness Audit ===")
    results_df = audit_da_weakness(df, regimes, out_dir)
    
    # Compute global DA sMAPE
    global_da_smape = compute_day_level_smape(df['rt_price'].values, 
                                              df['da_price'].values, 
                                              df['times'])
    
    # Generate report
    generate_report(results_df, global_da_smape, out_dir)
    
    print(f"\n=== Audit Complete ===")
    print(f"Results saved to {out_dir}")
    print(f"\nTop 3 high-priority regimes:")
    high_priority = results_df[results_df['candidate_priority'] == 'HIGH_PRIORITY'].head(3)
    if len(high_priority) > 0:
        print(high_priority[['regime', 'regime_value', 'n_rows', 'da_smape_floor50', 'smape_excess_vs_global']])

if __name__ == '__main__':
    main()
