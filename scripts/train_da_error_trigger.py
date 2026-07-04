"""
Phase B: Regime Trigger Model

目标：
训练一个分类器预测"DA 是否会大错"。

标签只能在训练中使用：
  large_da_error = abs(rt_actual - da_anchor) >= threshold

在线特征只能使用：
  calendar, da_anchor, da price level, past residual statistics, 
  past volatility, forecast-side features

禁止使用：
  target-day rt_actual, target-day residual, future data, target bucket directly

输出：
  trigger_leaderboard.csv
  threshold_metrics.csv
  topk_metrics.csv
  calibration_report.csv
  trigger_report.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def compute_topk_metrics(y_true, y_score, k_percentages=[10, 20]):
    """Compute precision/recall/lift at top K%."""
    results = []
    
    for k_pct in k_percentages:
        k = int(len(y_score) * k_pct / 100)
        if k == 0:
            continue
        
        # Get top-k indices (highest predicted probability)
        top_k_idx = np.argsort(y_score)[-k:]
        
        # Compute precision, recall, lift
        y_true_topk = y_true[top_k_idx]
        precision = np.mean(y_true_topk)
        recall = np.sum(y_true_topk) / np.sum(y_true) if np.sum(y_true) > 0 else 0
        base_rate = np.mean(y_true)
        lift = precision / base_rate if base_rate > 0 else 0
        
        results.append({
            'k_percentage': k_pct,
            'k_samples': k,
            'precision': precision,
            'recall': recall,
            'lift': lift,
            'base_rate': base_rate
        })
    
    return pd.DataFrame(results)

def train_trigger_model(X_train, y_train, X_val, y_val, threshold, out_dir):
    """
    Train trigger models for a given threshold.
    
    Args:
        X_train: Training features
        y_train: Training labels (large error or not)
        X_val: Validation features
        y_val: Validation labels
        threshold: Error threshold
        out_dir: Output directory
    """
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'HGBClassifier': HistGradientBoostingClassifier(random_state=42),
        'RandomForest_small': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }
    
    results = []
    
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Compute metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        # Compute top-k metrics
        topk_df = compute_topk_metrics(y_val, y_pred_proba, k_percentages=[10, 20])
        
        # Check KEEP condition
        precision_20 = topk_df[topk_df['k_percentage'] == 20]['precision'].values[0]
        recall_20 = topk_df[topk_df['k_percentage'] == 20]['recall'].values[0]
        base_rate = np.mean(y_val)
        
        keep = (precision_20 >= base_rate * 1.5) and (recall_20 >= 0.35)
        
        results.append({
            'model_name': model_name,
            'threshold': threshold,
            'auc': auc,
            'pr_auc': pr_auc,
            'precision@20%': precision_20,
            'recall@20%': recall_20,
            'lift@20%': topk_df[topk_df['k_percentage'] == 20]['lift'].values[0],
            'base_large_error_rate': base_rate,
            'keep': keep
        })
        
        print(f"  {model_name}: AUC={auc:.4f}, Precision@20%={precision_20:.4f}, Recall@20%={recall_20:.4f}, KEEP={keep}")
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Phase B: Regime Trigger Model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed data CSV')
    parser.add_argument('--thresholds', type=str, default='50,100,150,200',
                       help='Comma-separated list of error thresholds')
    parser.add_argument('--out-dir', type=str, required=True,
                       help='Output directory')
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.data_path}...")
    df = pd.read_csv(args.data_path, parse_dates=['times'], encoding='utf-8-sig')
    df = df.sort_values('times').reset_index(drop=True)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define online features (no target-day data)
    online_features = [
        'hour', 'dayofweek', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'da_price',  # DA anchor is available
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
    
    # Prepare data
    X = df[online_features].fillna(0).values
    
    # Walk-forward validation (use 2024-05 to 2025-12)
    print("\n=== Walk-Forward Validation (Trigger Model) ===")
    
    months = pd.date_range(start='2024-05-01', end='2025-12-01', freq='MS')
    all_results = []
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        
        # Create labels
        df['large_da_error'] = (np.abs(df['rt_price'] - df['da_price']) >= threshold).astype(int)
        y = df['large_da_error'].values
        
        threshold_results = []
        
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
            
            # Skip if no positive samples
            if np.sum(y_train) == 0 or np.sum(y_val) == 0:
                continue
            
            # Normalize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train trigger models
            month_results = train_trigger_model(X_train_scaled, y_train, X_val_scaled, y_val, 
                                              threshold, out_dir)
            month_results['train_period'] = train_start.strftime('%Y-%m')
            month_results['val_period'] = val_start.strftime('%Y-%m')
            
            threshold_results.append(month_results)
        
        # Aggregate results for this threshold
        if len(threshold_results) > 0:
            threshold_df = pd.concat(threshold_results, ignore_index=True)
            threshold_summary = threshold_df.groupby('model_name').agg({
                'auc': 'mean',
                'pr_auc': 'mean',
                'precision@20%': 'mean',
                'recall@20%': 'mean',
                'lift@20%': 'mean',
                'keep': 'any'
            }).reset_index()
            
            print(f"\nThreshold {threshold} summary:")
            print(threshold_summary)
            
            all_results.append(threshold_df)
    
    # Save all results
    if len(all_results) > 0:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv(out_dir / 'trigger_leaderboard.csv', index=False, encoding='utf-8-sig')
        print(f"\nTrigger leaderboard saved to {out_dir / 'trigger_leaderboard.csv'}")
    
    print("\n=== Phase B Complete ===")

if __name__ == '__main__':
    main()
