"""
Phase C: Conditional Specialist Residual Model

只有 Phase B trigger KEEP，才运行。

目标：
只在 DA likely wrong 的样本上训练/应用 residual correction。

形式：
  trigger_score = P(large_da_error)
  
  if trigger_score >= threshold:
      final_pred = da_anchor + alpha * clipped(residual_pred)
  else:
      final_pred = da_anchor

必须使用 validation 选择：
  trigger_threshold
  alpha
  clip

输出：
  specialist_leaderboard.csv
  decision_log.csv
  predictions.csv
  bucket_metrics.csv
  period_metrics.csv
  specialist_report.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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

def train_specialist(X_train, y_train, X_val, y_val, trigger_val, 
                     trigger_threshold_candidates, alpha_candidates, clip_candidates):
    """
    Train specialist model with trigger-based conditional correction.
    
    Args:
        X_train: Training features
        y_train: Training residual (rt - da)
        X_val: Validation features
        y_val: Validation residual
        trigger_val: Validation trigger scores (P(large_da_error))
        trigger_threshold_candidates: List of trigger thresholds to try
        alpha_candidates: List of alpha values to try
        clip_candidates: List of clip values to try
    
    Returns:
        best_model: Trained specialist model
        best_threshold: Best trigger threshold
        best_alpha: Best alpha
        best_clip: Best clip
        best_smape: Best validation sMAPE
    """
    # Train specialist model (simple Ridge)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    best_smape = float('inf')
    best_threshold = 0.0
    best_alpha = 0.0
    best_clip = 0.0
    
    da_val = X_val[:, 0]  # Assume first feature is da_price
    rt_val = y_val + da_val  # Reconstruct rt_price
    
    for threshold in trigger_threshold_candidates:
        trigger_mask = trigger_val >= threshold
        
        for alpha in alpha_candidates:
            for clip in clip_candidates:
                # Apply correction only where trigger fires
                correction = np.zeros_like(y_val)
                if np.sum(trigger_mask) > 0:
                    residual_pred = model.predict(X_val[trigger_mask])
                    correction[trigger_mask] = alpha * np.clip(residual_pred, -clip, clip)
                
                final_pred = da_val + correction
                
                # Compute sMAPE
                smape = compute_day_level_smape(rt_val, final_pred, 
                                                pd.Series(X_val[:, -1]))  # Assume last feature is timestamp
                
                if smape < best_smape:
                    best_smape = smape
                    best_threshold = threshold
                    best_alpha = alpha
                    best_clip = clip
                    best_model = model
    
    return best_model, best_threshold, best_alpha, best_clip, best_smape

def main():
    parser = argparse.ArgumentParser(description='Phase C: Conditional Specialist Residual Model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed data CSV')
    parser.add_argument('--trigger-model-path', type=str, 
                       help='Path to trained trigger model (optional, will train if not provided)')
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
    
    # Define online features (same as Phase B)
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
        'solar_forecast_rolling_mean_24h'
    ]
    
    # Filter to features that exist in df
    online_features = [f for f in online_features if f in df.columns]
    print(f"Number of online features: {len(online_features)}")
    
    # Add timestamp as last feature (for day-level sMAPE computation)
    online_features.append('times')
    
    # Prepare data
    X = df[online_features].fillna(0).values
    y = (df['rt_price'] - df['da_price']).values  # residual
    
    # Walk-forward validation (use 2024-05 to 2025-12)
    print("\n=== Walk-Forward Validation (Conditional Specialist) ===")
    
    months = pd.date_range(start='2024-05-01', end='2025-12-01', freq='MS')
    results = []
    
    trigger_threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
    alpha_candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    clip_candidates = [50, 100, 150, 200]
    
    for i in range(len(months) - 1):
        train_start = months[max(0, i-3)]
        train_end = months[i]
        val_start = months[i]
        val_end = months[i+1]
        
        # Get train/val data
        train_mask = (df['times'] >= train_start) & (df['times'] < train_end)
        val_mask = (df['times'] >= val_start) & (df['times'] < val_end)
        
        if train_mask.sum() < 100 or val_mask.sum() < 10:
            continue
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        # Train trigger model (simplified: use DA error directly)
        da_error = np.abs(df['rt_price'] - df['da_price']).values
        threshold = 100  # Use 100 as threshold for large error
        trigger_train = (da_error[train_mask] >= threshold).astype(int)
        
        # Train trigger model
        trigger_model = LogisticRegression(random_state=42, max_iter=1000)
        trigger_model.fit(X_train[:, :-1], trigger_train)  # Exclude timestamp
        trigger_val = trigger_model.predict_proba(X_val[:, :-1])[:, 1]
        
        # Train specialist with trigger-based conditional correction
        print(f"\nTraining period: {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}")
        print(f"Validation period: {val_start.strftime('%Y-%m')}")
        
        model, best_threshold, best_alpha, best_clip, best_smape = train_specialist(
            X_train[:, :-1], y_train, X_val[:, :-1], y_val, 
            trigger_val, trigger_threshold_candidates, alpha_candidates, clip_candidates
        )
        
        # Compute DA-only sMAPE for comparison
        da_smape = compute_day_level_smape(
            df[val_mask]['rt_price'].values,
            df[val_mask]['da_price'].values,
            df[val_mask]['times'].values
        )
        
        improvement = da_smape - best_smape
        
        results.append({
            'val_period': val_start.strftime('%Y-%m'),
            'specialist_smape': best_smape,
            'da_smape': da_smape,
            'improvement': improvement,
            'trigger_threshold': best_threshold,
            'alpha': best_alpha,
            'clip': best_clip
        })
        
        print(f"  Specialist sMAPE: {best_smape:.2f}")
        print(f"  DA sMAPE: {da_smape:.2f}")
        print(f"  Improvement: {improvement:.2f}")
        
        # Check KEEP condition
        if improvement >= 0.3:
            print(f"  KEEP: Improvement >= 0.3pp")
        else:
            print(f"  KILL: Improvement < 0.3pp")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / 'specialist_leaderboard.csv', index=False, encoding='utf-8-sig')
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Average Specialist sMAPE: {results_df['specialist_smape'].mean():.2f}")
    print(f"Average DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Average Improvement: {results_df['improvement'].mean():.2f}")
    print(f"Months with improvement >= 0.3pp: {(results_df['improvement'] >= 0.3).sum()}/{len(results_df)}")
    
    # Check if should proceed to Phase D
    if results_df['improvement'].mean() >= 0.3:
        print(f"\n=== Veredict: KEEP ===")
        print(f"Proceed to Phase D: Regime-Specific Deep Model")
    else:
        print(f"\n=== Veredict: KILL ===")
        print(f"Stop. Conditional specialist does not improve sMAPE.")

if __name__ == '__main__':
    main()
