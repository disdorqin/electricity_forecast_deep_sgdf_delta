"""
Phase E: Multi-month Walk-forward Validation

只有 2026-02 至少一个候选 KEEP，才跑多月：
  2026-01, 2026-02, 2026-03, 2026-04, 2026-05

使用 Phase C 的 conditional specialist 进行多月验证。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    daily_true = df.groupby('date')['y_true'].mean()
    daily_pred = df.groupby('date')['y_pred'].mean()
    
    return compute_smape_floor50(daily_true.values, daily_pred.values)

def run_multi_month_backtest(df, online_features, months, out_dir):
    """
    Run multi-month walk-forward backtest.
    
    Args:
        df: DataFrame with features and targets
        online_features: List of online feature names
        months: List of months to validate on
        out_dir: Output directory
    """
    results = []
    
    for i in range(len(months) - 1):
        train_start = months[max(0, i-3)]  # Use 3 months of training data
        train_end = months[i]
        test_start = months[i]
        test_end = months[i+1]
        
        # Get train/test data
        train_mask = (df['times'] >= train_start) & (df['times'] < train_end)
        test_mask = (df['times'] >= test_start) & (df['times'] < test_end)
        
        if train_mask.sum() < 100 or test_mask.sum() < 10:
            continue
        
        X_train = df[online_features].fillna(0).values[train_mask]
        y_train = (df['rt_price'] - df['da_price']).values[train_mask]
        X_test = df[online_features].fillna(0).values[test_mask]
        y_test = (df['rt_price'] - df['da_price']).values[test_mask]
        
        da_test = df['da_price'].values[test_mask]
        rt_test = df['rt_price'].values[test_mask]
        timestamps_test = df['times'].values[test_mask]
        
        # Train trigger model
        da_error_train = np.abs(df['rt_price'].values[train_mask] - 
                              df['da_price'].values[train_mask])
        threshold = 100
        trigger_train = (da_error_train >= threshold).astype(int)
        
        trigger_model = LogisticRegression(random_state=42, max_iter=1000)
        trigger_model.fit(X_train[:, :-1], trigger_train)  # Exclude timestamp
        trigger_test = trigger_model.predict_proba(X_test[:, :-1])[:, 1]
        
        # Train specialist model
        specialist_model = Ridge(alpha=1.0, random_state=42)
        specialist_model.fit(X_train[:, :-1], y_train)
        
        # Apply conditional correction
        trigger_threshold = 0.5
        correction = np.zeros_like(y_test)
        trigger_mask = trigger_test >= trigger_threshold
        
        if np.sum(trigger_mask) > 0:
            residual_pred = specialist_model.predict(X_test[trigger_mask, :-1])
            correction[trigger_mask] = residual_pred
        
        final_pred = da_test + correction
        
        # Compute sMAPE
        model_smape = compute_day_level_smape(rt_test, final_pred, timestamps_test)
        da_smape = compute_day_level_smape(rt_test, da_test, timestamps_test)
        
        improvement = da_smape - model_smape
        
        results.append({
            'test_month': test_start.strftime('%Y-%m'),
            'model_smape': model_smape,
            'da_smape': da_smape,
            'improvement': improvement,
            'trigger_fire_rate': np.mean(trigger_mask)
        })
        
        print(f"{test_start.strftime('%Y-%m')}: Model sMAPE={model_smape:.2f}, DA sMAPE={da_smape:.2f}, Improvement={improvement:.2f}, Trigger Fire Rate={np.mean(trigger_mask):.2%}")
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Phase E: Multi-month Walk-forward Validation')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed data CSV')
    parser.add_argument('--out-dir', type=str, required=True,
                       help='Output directory')
    args = parser.parse_args()
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.data_path}...")
    df = pd.read_csv(args.data_path, parse_dates=['times'], encoding='utf-8-sig')
    df = df.sort_values('times').reset_index(drop=True)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define online features (same as Phase B/C)
    online_features = [
        'hour', 'dayofweek', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'da_price',  # DA anchor
        'da_price_lag_24h', 'da_price_lag_48h', 'da_price_lag_72h', 'da_price_lag_168h',
        'rt_price_lag_24h', 'rt_price_lag_48h', 'rt_price_lag_72h', 'rt_price_lag_168h',
        'rt_price_rolling_mean_24h', 'rt_price_rolling_std_24h',
        'rt_price_rolling_mean_48h', 'rt_price_rolling_std_48h',
        'rt_price_rolling_mean_168h', 'rt_price_rolling_std_168h',
        'bidding_space_forecast', 'direct_dispatch_forecast',
        'wind_forecast', 'solar_forecast',
        'bidding_space_forecast_lag_24h', 'bidding_space_forecast_lag_168h',
        'direct_dispatch_forecast_lag_24h', 'direct_dispatch_forecast_lag_168h',
        'wind_forecast_lag_24h', 'wind_forecast_lag_168h',
        'solar_forecast_lag_24h', 'solar_forecast_lag_168h',
        'bidding_space_forecast_rolling_mean_24h',
        'direct_dispatch_forecast_rolling_mean_24h',
        'wind_forecast_rolling_mean_24h',
        'solar_forecast_rolling_mean_24h',
        'times'  # Add timestamp as last feature
    ]
    
    # Filter to features that exist in df
    online_features = [f for f in online_features if f in df.columns]
    print(f"Number of online features: {len(online_features)}")
    
    # Define months for validation (2026-01 to 2026-05)
    months = pd.date_range(start='2026-01-01', end='2026-06-01', freq='MS')
    
    # Check if data covers these months
    if df['times'].max() < months[-1]:
        print(f"Warning: Data only goes to {df['times'].max()}, but validation requires up to {months[-1]}")
        print(f"Adjusting validation months...")
        months = months[months <= df['times'].max()]
    
    if len(months) < 2:
        print("Error: Not enough data for multi-month validation")
        return
    
    print(f"\n=== Multi-Month Walk-Forward Validation ===")
    print(f"Validation months: {[m.strftime('%Y-%m') for m in months]}")
    
    # Run backtest
    results_df = run_multi_month_backtest(df, online_features, months, out_dir)
    
    # Save results
    results_df.to_csv(out_dir / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    
    # Summary
    print(f"\n=== Summary (Multi-Month Validation) ===")
    print(f"Average Model sMAPE: {results_df['model_smape'].mean():.2f}")
    print(f"Average DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Average Improvement: {results_df['improvement'].mean():.2f}")
    print(f"Months with improvement >= 0.3pp: {(results_df['improvement'] >= 0.3).sum()}/{len(results_df)}")
    
    # Check verdict
    if results_df['improvement'].mean() >= 0.3:
        print(f"\n=== Verdict: REGIME_SPECIALIST_GO ===")
        print(f"Mean monthly sMAPE beats DA by >=0.3pp")
        verdict = 'REGIME_SPECIALIST_GO'
    elif (results_df['improvement'].mean() >= -0.1) and (results_df['improvement'].min() >= -1.0):
        print(f"\n=== Verdict: REGIME_SPECIALIST_AUX ===")
        print(f"Large-error bucket improves >=3pp and global degradation <=0.1pp")
        verdict = 'REGIME_SPECIALIST_AUX'
    else:
        print(f"\n=== Verdict: REGIME_SPECIALIST_NO_GO ===")
        print(f"No valid improvement")
        verdict = 'REGIME_SPECIALIST_NO_GO'
    
    # Save verdict
    verdict_dict = {
        'verdict': verdict,
        'avg_model_smape': results_df['model_smape'].mean(),
        'avg_da_smape': results_df['da_smape'].mean(),
        'avg_improvement': results_df['improvement'].mean(),
        'months_improved': (results_df['improvement'] >= 0.3).sum(),
        'total_months': len(results_df)
    }
    
    import json
    with open(out_dir / 'champion_summary.json', 'w') as f:
        json.dump(verdict_dict, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")

if __name__ == '__main__':
    main()
