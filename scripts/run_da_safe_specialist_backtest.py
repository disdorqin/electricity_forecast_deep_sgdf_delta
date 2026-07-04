"""
Phase D: Safe Specialist Re-run.

Re-run specialist with DA-safe guards to prevent catastrophic failures.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """Prepare features."""
    X = df[feature_cols].fillna(0).values
    return X

def walk_forward_backtest_with_guards(data_path, out_dir, guard_config=None):
    """
    Run walk-forward backtest with safety guards.
    
    Args:
        data_path: Path to preprocessed data
        out_dir: Output directory
        guard_config: Dict with guard configuration
    """
    
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
    
    # Define test months
    test_months = ['2026-02', '2026-03', '2026-04', '2026-05']
    
    # Results
    all_results = []
    all_decisions = []
    
    for test_month in test_months:
        print(f"\n=== Testing {test_month} ===")
        
        # Define train/val/test periods
        test_start = test_month
        test_end = (pd.Timestamp(test_month) + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        val_start = (pd.Timestamp(test_month) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        val_end = test_start
        train_end = val_start
        
        df_train = df[df['times'] < train_end].copy()
        df_val = df[(df['times'] >= val_start) & (df['times'] < val_end)].copy()
        df_test = df[(df['times'] >= test_start) & (df['times'] < test_end)].copy()
        
        print(f"  Train: {len(df_train)} rows")
        print(f"  Val: {len(df_val)} rows")
        print(f"  Test: {len(df_test)} rows")
        
        if len(df_test) == 0:
            print(f"  Skipping {test_month} (no test data)")
            continue
        
        # Train trigger model (threshold=100)
        print(f"  Training trigger model...")
        df_train['large_error'] = (np.abs(df_train['rt_price'] - df_train['da_price']) >= 100).astype(int)
        
        X_train = prepare_features(df_train, feature_cols)
        y_trigger = df_train['large_error'].values
        
        if y_trigger.sum() > 0 and y_trigger.sum() < len(y_trigger):
            trigger_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
            trigger_model.fit(X_train, y_trigger)
        else:
            print(f"  Warning: Only one class in training data")
            trigger_model = None
        
        # Train residual model
        print(f"  Training residual model...")
        residual_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        y_residual = (df_train['rt_price'] - df_train['da_price']).values
        residual_model.fit(X_train, y_residual)
        
        # Select best alpha/clip on validation set
        print(f"  Selecting parameters on validation...")
        
        X_val = prepare_features(df_val, feature_cols)
        
        if trigger_model is not None:
            val_trigger_scores = trigger_model.predict_proba(X_val)[:, 1]
        else:
            val_trigger_scores = np.ones(len(df_val)) * 0.5
        
        val_residual_pred = residual_model.predict(X_val)
        
        best_improvement = -999
        best_alpha = 0.0
        best_clip = 0.0
        best_top_k = 0.05
        
        for alpha in [0.02, 0.05, 0.10, 0.20]:
            for clip in [5, 10, 20, 30, 50]:
                for top_k in [0.03, 0.05, 0.10]:
                    
                    # Apply correction only to top-k
                    threshold = np.percentile(val_trigger_scores, 100 * (1 - top_k))
                    fire_mask = val_trigger_scores >= threshold
                    
                    correction = np.zeros(len(df_val))
                    correction[fire_mask] = alpha * np.clip(val_residual_pred[fire_mask], -clip, clip)
                    
                    val_pred = df_val['da_price'].values + correction
                    val_smape = compute_day_level_smape(df_val['rt_price'], val_pred, df_val['times'])
                    da_smape = compute_day_level_smape(df_val['rt_price'], df_val['da_price'], df_val['times'])
                    
                    improvement = da_smape - val_smape
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_alpha = alpha
                        best_clip = clip
                        best_top_k = top_k
        
        print(f"  Best validation improvement: {best_improvement:.4f}")
        print(f"  Best params: alpha={best_alpha}, clip={best_clip}, top_k={best_top_k}")
        
        # Initialize guard
        from models.deep_sgdf_delta.da_safe_guard import DASafeGuard
        guard = DASafeGuard(config=guard_config or {'max_fire_rate_per_month': 0.05, 'max_abs_correction': 20})
        
        # Apply to test set
        print(f"  Applying to test set...")
        
        X_test = prepare_features(df_test, feature_cols)
        
        if trigger_model is not None:
            test_trigger_scores = trigger_model.predict_proba(X_test)[:, 1]
        else:
            test_trigger_scores = np.ones(len(df_test)) * 0.5
        
        test_residual_pred = residual_model.predict(X_test)
        
        # Apply top-k correction
        threshold = np.percentile(test_trigger_scores, 100 * (1 - best_top_k))
        fire_mask = test_trigger_scores >= threshold
        
        raw_corrections = np.zeros(len(df_test))
        raw_corrections[fire_mask] = best_alpha * np.clip(test_residual_pred[fire_mask], -best_clip, best_clip)
        
        # Apply guard
        test_df = df_test.copy()
        test_df['da_anchor'] = test_df['da_price']
        test_df['business_day'] = test_df['times']
        test_df['hour_business'] = test_df['times'].dt.hour + 1
        
        final_predictions = []
        decisions = []
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            correction = raw_corrections[i]
            trigger_score = test_trigger_scores[i]
            
            # Reset month if needed
            month = pd.Timestamp(row['times']).to_period('M')
            guard.reset_month(month)
            
            # Check guards
            decision = guard.check_guards(
                row.to_dict(),
                correction,
                trigger_score,
                val_improvement=best_improvement
            )
            
            final_pred = row['da_price'] + decision['safe_correction']
            final_predictions.append(final_pred)
            
            decisions.append({
                'business_day': row['times'],
                'hour_business': row['hour_business'],
                'da_anchor': row['da_price'],
                'raw_correction': decision['raw_correction'],
                'safe_correction': decision['safe_correction'],
                'final_pred': final_pred,
                'trigger_score': trigger_score,
                'fire': decision['fire'],
                'blocked_by_guard': decision['blocked_by_guard'],
                'block_reason': decision['block_reason']
            })
        
        # Compute metrics
        test_smape = compute_day_level_smape(df_test['rt_price'], np.array(final_predictions), df_test['times'])
        da_smape = compute_day_level_smape(df_test['rt_price'], df_test['da_price'], df_test['times'])
        
        fire_rate = len([d for d in decisions if d['fire']]) / len(decisions)
        
        print(f"  Test sMAPE: {test_smape:.2f}")
        print(f"  DA sMAPE: {da_smape:.2f}")
        print(f"  Improvement: {da_smape - test_smape:.2f}")
        print(f"  Fire rate: {fire_rate*100:.2f}%")
        
        all_results.append({
            'test_month': test_month,
            'model_smape': test_smape,
            'da_smape': da_smape,
            'improvement': da_smape - test_smape,
            'fire_rate': fire_rate,
            'best_alpha': best_alpha,
            'best_clip': best_clip,
            'best_top_k': best_top_k
        })
        
        all_decisions.extend(decisions)
    
    # Summary
    results_df = pd.DataFrame(all_results)
    decisions_df = pd.DataFrame(all_decisions)
    
    print(f"\n=== Summary ===")
    print(f"Mean model sMAPE: {results_df['model_smape'].mean():.2f}")
    print(f"Mean DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Mean improvement: {results_df['improvement'].mean():.2f}")
    print(f"Months with improvement: {(results_df['improvement'] > 0).sum()}/{len(results_df)}")
    
    # Save results
    results_df.to_csv(out_dir / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    decisions_df.to_csv(out_dir / 'decision_log.csv', index=False, encoding='utf-8-sig')
    
    # Generate report
    report_path = out_dir / 'da_safe_specialist_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DA-Safe Specialist Backtest Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Mean model sMAPE: {results_df['model_smape'].mean():.2f}\n")
        f.write(f"Mean DA sMAPE: {results_df['da_smape'].mean():.2f}\n")
        f.write(f"Mean improvement: {results_df['improvement'].mean():.2f}\n\n")
        
        f.write("## Monthly Metrics\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Verdict\n\n")
        
        mean_imp = results_df['improvement'].mean()
        max_damage = results_df['improvement'].min()
        
        if mean_imp >= 0.3 and max_damage > -0.5:
            verdict = "DA_SAFE_GO"
        elif mean_imp > 0 and max_damage > -0.2:
            verdict = "DA_SAFE_STABLE"
        elif max_damage > -0.5:
            verdict = "DA_SAFE_AUX"
        else:
            verdict = "DA_SAFE_NO_GO"
        
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"- Mean improvement: {mean_imp:.2f}\n")
        f.write(f"- Max damage: {max_damage:.2f}\n")
    
    print(f"\nReport saved to {report_path}")
    
    return results_df, decisions_df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--max-fire-rate', type=float, default=0.05)
    parser.add_argument('--max-correction', type=float, default=20)
    args = parser.parse_args()
    
    guard_config = {
        'max_fire_rate_per_month': args.max_fire_rate,
        'max_abs_correction': args.max_correction
    }
    
    walk_forward_backtest_with_guards(args.data_path, args.out_dir, guard_config)

if __name__ == '__main__':
    main()
