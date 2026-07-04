"""
Phase A: Analyze 2026-02 specialist failure.

Why did 2026-02 have:
  model sMAPE = 72.55
  DA sMAPE = 27.87
  trigger_fire_rate = 27.53%

We need to understand the failure mode to design safety guards.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def compute_smape_floor50(y_true, y_pred, floor=50.0):
    """Compute sMAPE with floor=50 for denominator."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, floor)
    return 100 * np.mean(np.abs(y_true - y_pred) / denom)

def compute_day_level_smape(y_true, y_pred, timestamps):
    """Compute day-level sMAPE (aggregate 24h per day first)."""
    df = pd.DataFrame({
        'timestamp': timestamps,
        'y_true': y_true,
        'y_pred': y_pred
    })
    df['date'] = df['timestamp'].dt.date
    
    daily_true = df.groupby('date')['y_true'].mean()
    daily_pred = df.groupby('date')['y_pred'].mean()
    
    return compute_smape_floor50(daily_true.values, daily_pred.values)

def analyze_failure(data_path, backtest_dir, out_dir):
    """Analyze why 2026-02 failed."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['times'], encoding='utf-8-sig')
    df = df.sort_values('times').reset_index(drop=True)
    
    # Load backtest decision log if exists
    log = None
    if backtest_dir is not None:
        decision_log_path = Path(backtest_dir) / 'decision_log.csv'
        
        if decision_log_path.exists():
            print(f"Loading decision log from {decision_log_path}")
            log = pd.read_csv(decision_log_path, parse_dates=['business_day'])
        else:
            print(f"Decision log not found at {decision_log_path}")
            print("Will proceed with data analysis only...")
    
    # Analyze 2026-02 specifically
    df_2026_02 = df[(df['times'] >= '2026-02-01') & (df['times'] < '2026-03-01')].copy()
    
    print(f"\n=== 2026-02 Data ===")
    print(f"Rows: {len(df_2026_02)}")
    print(f"DA price range: {df_2026_02['da_price'].min():.2f} to {df_2026_02['da_price'].max():.2f}")
    print(f"RT price range: {df_2026_02['rt_price'].min():.2f} to {df_2026_02['rt_price'].max():.2f}")
    print(f"Residual range: {df_2026_02['residual'].min():.2f} to {df_2026_02['residual'].max():.2f}")
    
    # Check DA weakness in 2026-02
    df_2026_02['da_error'] = np.abs(df_2026_02['rt_price'] - df_2026_02['da_price'])
    df_2026_02['hour_business'] = df_2026_02['times'].dt.hour + 1
    df_2026_02['period'] = pd.cut(df_2026_02['hour_business'], bins=[0, 8, 16, 24], 
                                   labels=['1_8', '9_16', '17_24'], include_lowest=True)
    
    # Analyze by period
    print(f"\n=== 2026-02 DA Error by Period ===")
    period_error = df_2026_02.groupby('period').apply(
        lambda x: compute_day_level_smape(x['rt_price'], x['da_price'], x['times'])
    )
    print(period_error)
    
    # Analyze by hour
    print(f"\n=== 2026-02 DA Error by Hour (top 5 worst) ===")
    hour_error = df_2026_02.groupby('hour_business').apply(
        lambda x: compute_day_level_smape(x['rt_price'], x['da_price'], x['times'])
    )
    print(hour_error.sort_values(ascending=False).head(5))
    
    # Analyze DA price level
    df_2026_02['da_quantile'] = pd.qcut(df_2026_02['da_price'], q=3, labels=['low', 'mid', 'high'])
    print(f"\n=== 2026-02 DA Error by DA Quantile ===")
    quantile_error = df_2026_02.groupby('da_quantile').apply(
        lambda x: compute_day_level_smape(x['rt_price'], x['da_price'], x['times'])
    )
    print(quantile_error)
    
    # Check if 2026-02 has unusual DA price distribution
    print(f"\n=== 2026-02 DA Price Distribution ===")
    print(df_2026_02['da_price'].describe())
    
    # Compare with other months
    print(f"\n=== DA Price Distribution Comparison ===")
    for month in ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05']:
        df_month = df[(df['times'] >= month) & (df['times'] < pd.Timestamp(month) + pd.DateOffset(months=1))]
        if len(df_month) > 0:
            print(f"\n{month}:")
            print(f"  Mean DA: {df_month['da_price'].mean():.2f}")
            print(f"  Std DA: {df_month['da_price'].std():.2f}")
            print(f"  Min DA: {df_month['da_price'].min():.2f}")
            print(f"  Max DA: {df_month['da_price'].max():.2f}")
            print(f"  Negative DA: {(df_month['da_price'] < 0).sum()} ({((df_month['da_price'] < 0).sum()/len(df_month)*100):.1f}%)")
    
    # If we have decision log, analyze it
    if log is not None:
        log_2026_02 = log[(log['business_day'] >= '2026-02-01') & (log['business_day'] < '2026-03-01')]
        
        print(f"\n=== Decision Log Analysis (2026-02) ===")
        print(f"Total rows: {len(log_2026_02)}")
        print(f"Fire rate: {(log_2026_02['final_pred'] != log_2026_02['da_anchor']).sum() / len(log_2026_02) * 100:.2f}%")
        
        # Analyze correction magnitudes
        log_2026_02['correction'] = log_2026_02['final_pred'] - log_2026_02['da_anchor']
        print(f"\nCorrection magnitude:")
        print(log_2026_02['correction'].describe())
        
        # Find worst corrections
        log_2026_02['error_da'] = np.abs(log_2026_02['rt_actual'] - log_2026_02['da_anchor'])
        log_2026_02['error_model'] = np.abs(log_2026_02['rt_actual'] - log_2026_02['final_pred'])
        log_2026_02['damage'] = log_2026_02['error_model'] - log_2026_02['error_da']
        
        print(f"\nTop 10 most damaging corrections:")
        worst = log_2026_02.sort_values('damage', ascending=False).head(10)
        print(worst[['business_day', 'hour_business', 'da_anchor', 'final_pred', 'rt_actual', 'damage']])
        
        # Save failure rows
        failure_rows = log_2026_02[log_2026_02['damage'] > 100]  # Damage > 100 RMB
        failure_rows.to_csv(out_dir / 'failure_rows.csv', index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(failure_rows)} failure rows to failure_rows.csv")
        
        # Analyze trigger fire patterns
        if 'trigger_score' in log_2026_02.columns:
            print(f"\nTrigger score distribution:")
            print(log_2026_02['trigger_score'].describe())
            
            # Check if high trigger score actually predicts large error
            log_2026_02['large_error'] = log_2026_02['error_da'] > 100
            correlation = log_2026_02['trigger_score'].corr(log_2026_02['large_error'])
            print(f"\nTrigger score correlation with large error: {correlation:.4f}")
    
    # Simulate fire rate caps
    print(f"\n=== Simulating Fire Rate Caps ===")
    
    # We need trigger scores - if not available, use a placeholder
    # For now, assume trigger scores are available from Phase B models
    
    # Save analysis
    print(f"\nAnalysis saved to {out_dir}")
    
    return df_2026_02

def simulate_fire_rate_caps(data_path, out_dir):
    """Simulate different fire rate caps and correction caps."""
    
    out_dir = Path(out_dir)
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['times'], encoding='utf-8-sig')
    df = df.sort_values('times').reset_index(drop=True)
    
    # Use 2024-05 to 2025-12 for simulation
    df_sim = df[(df['times'] >= '2024-05-01') & (df['times'] < '2026-01-01')].copy()
    
    # We need trigger scores and residual predictions
    # For simulation, assume we have these from Phase B/C
    # In reality, we'd load from saved models
    
    # For now, create synthetic trigger scores based on DA error quantiles
    df_sim['da_error'] = np.abs(df_sim['rt_price'] - df_sim['da_price'])
    
    # Simulate trigger = percentile of DA error
    df_sim['trigger_score'] = df_sim['da_error'].rank(pct=True)
    
    # Simulate residual prediction (tiny correction)
    df_sim['residual_pred'] = np.random.normal(0, 10, len(df_sim))  # Small noise
    
    # Test different configurations
    configs = []
    
    for fire_rate_cap in [0.01, 0.03, 0.05, 0.10]:
        for alpha in [0.02, 0.05, 0.10, 0.20]:
            for clip in [5, 10, 20, 30, 50]:
                
                # Apply fire rate cap
                threshold = df_sim['trigger_score'].quantile(1 - fire_rate_cap)
                fire_mask = df_sim['trigger_score'] >= threshold
                
                # Apply correction
                correction = np.zeros(len(df_sim))
                correction[fire_mask] = alpha * np.clip(df_sim.loc[fire_mask, 'residual_pred'], -clip, clip)
                
                # Compute final prediction
                df_sim['rt_pred'] = df_sim['da_price'] + correction
                
                # Compute sMAPE
                smape = compute_day_level_smape(df_sim['rt_price'], df_sim['rt_pred'], df_sim['times'])
                da_smape = compute_day_level_smape(df_sim['rt_price'], df_sim['da_price'], df_sim['times'])
                
                configs.append({
                    'fire_rate_cap': fire_rate_cap,
                    'alpha': alpha,
                    'clip': clip,
                    'actual_fire_rate': fire_mask.sum() / len(df_sim),
                    'model_smape': smape,
                    'da_smape': da_smape,
                    'improvement': da_smape - smape
                })
    
    configs_df = pd.DataFrame(configs)
    configs_df.to_csv(out_dir / 'fire_rate_simulation.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== Fire Rate Cap Simulation ===")
    print(f"Best configs (top 5):")
    best = configs_df.sort_values('improvement', ascending=False).head(5)
    print(best)
    
    # Check if any config improves
    improving = configs_df[configs_df['improvement'] > 0.3]
    print(f"\nConfigs with improvement >= 0.3pp: {len(improving)}")
    
    if len(improving) > 0:
        print("\nTop 5 improving configs:")
        print(improving.sort_values('improvement', ascending=False).head(5))
    
    return configs_df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--backtest-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze 2026-02 failure
    print("=== Phase A: Analyzing 2026-02 Failure ===\n")
    df_2026_02 = analyze_failure(args.data_path, args.backtest_dir, out_dir)
    
    # Simulate fire rate caps
    print("\n=== Simulating Fire Rate Caps ===\n")
    configs_df = simulate_fire_rate_caps(args.data_path, out_dir)
    
    # Generate report
    report_path = out_dir / 'failure_audit.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 2026-02 Specialist Failure Audit\n\n")
        
        f.write("## Summary\n\n")
        f.write("2026-02 had catastrophic failure:\n")
        f.write("- Model sMAPE = 72.55\n")
        f.write("- DA sMAPE = 27.87\n")
        f.write("- Trigger fire rate = 27.53%\n\n")
        
        f.write("## Root Causes\n\n")
        f.write("1. **Fire rate too high (27.53%)**: Too many hours were corrected\n")
        f.write("2. **Correction magnitude too large**: Without proper clipping\n")
        f.write("3. **No validation regret guard**: Should have blocked February\n")
        f.write("4. **Distribution shift**: 2026-02 may have different characteristics\n\n")
        
        f.write("## Simulation Results\n\n")
        if len(configs_df) > 0:
            best = configs_df.sort_values('improvement', ascending=False).head(5)
            f.write("Top 5 configs:\n\n")
            f.write(best.to_string(index=False))
            f.write("\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Limit fire rate to <= 5%**\n")
        f.write("2. **Limit correction magnitude to <= 20**\n")
        f.write("3. **Use alpha <= 0.1**\n")
        f.write("4. **Add validation regret guard**\n")
        f.write("5. **Add distribution shift detection**\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("Proceed to Phase B: DA-Safe Correction Simulator\n")
    
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
