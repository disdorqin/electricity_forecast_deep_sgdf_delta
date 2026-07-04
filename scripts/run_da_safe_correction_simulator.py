"""
Phase B: DA-Safe Correction Simulator.

Test conservative correction strategies without training new models.
Use validation to select parameters, then evaluate on test set.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
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

def prepare_features(df, feature_cols):
    """Prepare features for trigger/specialist models."""
    X = df[feature_cols].fillna(0).values
    return X

def train_trigger_model(df_train, feature_cols, threshold):
    """Train trigger model to predict large DA error."""
    
    # Create label
    df_train['large_error'] = (np.abs(df_train['rt_price'] - df_train['da_price']) >= threshold).astype(int)
    
    X = prepare_features(df_train, feature_cols)
    y = df_train['large_error'].values
    
    # Check if we have both classes
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  Warning: Only one class for threshold {threshold}")
        return None, 0.0
    
    # Train HGB classifier
    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model.fit(X, y)
    
    # Compute AUC on training data (for reference)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    
    return model, auc

def train_residual_model(df_train, feature_cols):
    """Train residual prediction model."""
    
    X = prepare_features(df_train, feature_cols)
    y = (df_train['rt_price'] - df_train['da_price']).values
    
    # Train HGB regressor
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X, y)
    
    return model

def simulate_strategy(df_test, trigger_model, residual_model, feature_cols, strategy_config):
    """Simulate a correction strategy and return predictions."""
    
    X_test = prepare_features(df_test, feature_cols)
    
    # Get trigger scores
    if trigger_model is not None:
        trigger_scores = trigger_model.predict_proba(X_test)[:, 1]
    else:
        # No trigger - use uniform scores
        trigger_scores = np.ones(len(df_test)) * 0.5
    
    # Get residual predictions
    if residual_model is not None:
        residual_pred = residual_model.predict(X_test)
    else:
        residual_pred = np.zeros(len(df_test))
    
    # Apply strategy
    strategy = strategy_config['strategy']
    alpha = strategy_config.get('alpha', 0.1)
    clip = strategy_config.get('clip', 20)
    top_k = strategy_config.get('top_k', 0.05)
    
    # Initialize correction
    correction = np.zeros(len(df_test))
    
    if strategy == 'top_k':
        # Only correct top-k% highest trigger scores
        threshold = np.percentile(trigger_scores, 100 * (1 - top_k))
        fire_mask = trigger_scores >= threshold
        correction[fire_mask] = alpha * np.clip(residual_pred[fire_mask], -clip, clip)
    
    elif strategy == 'small_alpha':
        # Apply correction with small alpha
        fire_mask = np.ones(len(df_test), dtype=bool)  # Correct all
        correction[fire_mask] = alpha * np.clip(residual_pred[fire_mask], -clip, clip)
    
    elif strategy == 'direction_only':
        # Only predict direction, use fixed step
        fire_mask = trigger_scores >= 0.5
        direction = np.sign(residual_pred[fire_mask])
        fixed_step = strategy_config.get('fixed_step', 10)
        correction[fire_mask] = direction * fixed_step
    
    elif strategy == 'bucket_only':
        # Only correct specific buckets
        bucket = strategy_config.get('bucket', 'low_da')
        if bucket == 'low_da':
            fire_mask = df_test['da_price'] < df_test['da_price'].quantile(0.33)
        elif bucket == 'negative_da':
            fire_mask = df_test['da_price'] < 0
        else:
            fire_mask = np.ones(len(df_test), dtype=bool)
        
        correction[fire_mask] = alpha * np.clip(residual_pred[fire_mask], -clip, clip)
    
    # Compute final prediction
    final_pred = df_test['da_price'].values + correction
    
    return final_pred, correction, trigger_scores

def run_phase_b_simulation(data_path, out_dir):
    """Run Phase B simulation."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['times'], encoding='utf-8-sig')
    df = df.sort_values('times').reset_index(drop=True)
    
    # Define feature columns
    exclude_cols = ['times', 'rt_price', 'da_price', 'residual',
                   'local_plant_actual', 'tie_line_load_actual', 'wind_actual',
                   'solar_actual', 'nuclear_actual', 'self_supply_actual',
                   'test_unit_actual', 'direct_dispatch_actual',
                   'bidding_space_actual', 'renewable_actual']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Feature columns: {len(feature_cols)}")
    
    # Define validation and test periods
    val_start = '2025-09-01'
    val_end = '2025-12-01'
    test_start = '2026-02-01'
    test_end = '2026-06-01'
    
    df_val = df[(df['times'] >= val_start) & (df['times'] < val_end)].copy()
    df_test = df[(df['times'] >= test_start) & (df['times'] < test_end)].copy()
    
    print(f"Validation: {len(df_val)} rows")
    print(f"Test: {len(df_test)} rows")
    
    # Train trigger and residual models on pre-validation data
    df_train = df[df['times'] < val_start].copy()
    print(f"Training: {len(df_train)} rows")
    
    # Train trigger models for different thresholds
    thresholds = [50, 100, 150, 200]
    trigger_models = {}
    
    for threshold in thresholds:
        print(f"\nTraining trigger model (threshold={threshold})...")
        model, auc = train_trigger_model(df_train, feature_cols, threshold)
        trigger_models[threshold] = model
        print(f"  AUC: {auc:.4f}")
    
    # Train residual model
    print(f"\nTraining residual model...")
    residual_model = train_residual_model(df_train, feature_cols)
    
    # Define strategies to test
    strategies = []
    
    # Strategy 1: top-k
    for top_k in [0.01, 0.03, 0.05, 0.10]:
        for alpha in [0.02, 0.05, 0.10]:
            for clip in [5, 10, 20, 30]:
                strategies.append({
                    'strategy': 'top_k',
                    'top_k': top_k,
                    'alpha': alpha,
                    'clip': clip,
                    'threshold': 100  # Use threshold=100 trigger
                })
    
    # Strategy 2: bucket-only
    for bucket in ['low_da', 'negative_da']:
        for alpha in [0.02, 0.05, 0.10]:
            for clip in [5, 10, 20]:
                strategies.append({
                    'strategy': 'bucket_only',
                    'bucket': bucket,
                    'alpha': alpha,
                    'clip': clip,
                    'threshold': 100
                })
    
    print(f"\nTotal strategies to test: {len(strategies)}")
    
    # Evaluate on validation set first
    print(f"\n=== Validation Evaluation ===")
    
    val_results = []
    
    for i, config in enumerate(strategies):
        threshold = config['threshold']
        trigger_model = trigger_models.get(threshold)
        
        if trigger_model is None:
            continue
        
        # Simulate on validation set
        val_pred, correction, trigger_scores = simulate_strategy(
            df_val, trigger_model, residual_model, feature_cols, config
        )
        
        # Compute sMAPE
        val_smape = compute_day_level_smape(df_val['rt_price'], val_pred, df_val['times'])
        da_smape = compute_day_level_smape(df_val['rt_price'], df_val['da_price'], df_val['times'])
        
        # Compute fire rate
        fire_rate = (np.abs(correction) > 0).sum() / len(correction)
        
        val_results.append({
            'strategy': config['strategy'],
            'top_k': config.get('top_k', None),
            'bucket': config.get('bucket', None),
            'alpha': config['alpha'],
            'clip': config['clip'],
            'threshold': threshold,
            'val_smape': val_smape,
            'da_smape_val': da_smape,
            'val_improvement': da_smape - val_smape,
            'fire_rate': fire_rate
        })
        
        if i % 20 == 0:
            print(f"  Strategy {i}/{len(strategies)}: improvement={da_smape - val_smape:.4f}")
    
    val_results_df = pd.DataFrame(val_results)
    val_results_df.to_csv(out_dir / 'validation_results.csv', index=False, encoding='utf-8-sig')
    
    # Select best strategies (top 10 by validation improvement)
    best_val = val_results_df.sort_values('val_improvement', ascending=False).head(10)
    
    print(f"\n=== Top 10 Strategies (Validation) ===")
    print(best_val[['strategy', 'alpha', 'clip', 'val_improvement', 'fire_rate']])
    
    # Evaluate best strategies on test set
    print(f"\n=== Test Evaluation ===")
    
    test_results = []
    
    for _, config_row in best_val.iterrows():
        config = config_row.to_dict()
        threshold = int(config['threshold'])
        trigger_model = trigger_models.get(threshold)
        
        if trigger_model is None:
            continue
        
        # Convert row to strategy config
        strategy_config = {
            'strategy': config['strategy'],
            'alpha': config['alpha'],
            'clip': config['clip'],
            'threshold': threshold
        }
        
        if config['strategy'] == 'top_k':
            strategy_config['top_k'] = config['top_k']
        elif config['strategy'] == 'bucket_only':
            strategy_config['bucket'] = config['bucket']
        
        # Simulate on test set
        test_pred, correction, trigger_scores = simulate_strategy(
            df_test, trigger_model, residual_model, feature_cols, strategy_config
        )
        
        # Compute sMAPE
        test_smape = compute_day_level_smape(df_test['rt_price'], test_pred, df_test['times'])
        da_smape_test = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
        
        # Compute fire rate
        fire_rate = (np.abs(correction) > 0).sum() / len(correction)
        
        test_results.append({
            'strategy': config['strategy'],
            'top_k': config.get('top_k', None),
            'bucket': config.get('bucket', None),
            'alpha': config['alpha'],
            'clip': config['clip'],
            'threshold': threshold,
            'test_smape': test_smape,
            'da_smape_test': da_smape_test,
            'test_improvement': da_smape_test - test_smape,
            'fire_rate': fire_rate,
            'val_improvement': config['val_improvement']
        })
    
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(out_dir / 'test_results.csv', index=False, encoding='utf-8-sig')
    
    # Print results
    print(f"\n=== Test Results ===")
    print(test_results_df[['strategy', 'alpha', 'clip', 'test_improvement', 'fire_rate']])
    
    # Apply KEEP/KILL criteria
    mean_improvement = test_results_df['test_improvement'].mean()
    max_damage = test_results_df['test_improvement'].min()  # Most negative
    
    print(f"\n=== KEEP/KILL Decision ===")
    print(f"Mean improvement: {mean_improvement:.4f}")
    print(f"Max damage: {max_damage:.4f}")
    
    if mean_improvement >= 0.3 and max_damage > -0.5:
        verdict = "KEEP"
    elif mean_improvement >= 0 and max_damage > -0.2:
        verdict = "AUX_KEEP"
    else:
        verdict = "KILL"
    
    print(f"Verdict: {verdict}")
    
    # Generate report
    report_path = out_dir / 'simulator_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DA-Safe Correction Simulator Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Total strategies tested: {len(strategies)}\n\n")
        
        f.write("## Top 10 Strategies (Validation)\n\n")
        f.write(best_val[['strategy', 'alpha', 'clip', 'val_improvement', 'fire_rate']].to_string())
        f.write("\n\n")
        
        f.write("## Test Results\n\n")
        f.write(test_results_df[['strategy', 'alpha', 'clip', 'test_improvement', 'fire_rate']].to_string())
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        f.write(f"- Mean improvement: {mean_improvement:.4f}\n")
        f.write(f"- Max damage: {max_damage:.4f}\n")
        f.write(f"- **Verdict: {verdict}**\n\n")
        
        if verdict == "KILL":
            f.write("## Conclusion\n\n")
            f.write("All strategies failed to improve DA-only significantly.\n")
            f.write("Recommendation: Do not proceed to model training.\n")
    
    print(f"\nReport saved to {report_path}")
    
    return test_results_df, verdict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    run_phase_b_simulation(args.data_path, args.out_dir)

if __name__ == '__main__':
    main()
