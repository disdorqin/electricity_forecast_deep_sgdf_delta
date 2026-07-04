"""
Phase B: Tabular Residual Probe
================================

Use strong tabular models (Ridge, HGB, RF) to predict residual.
If tabular cannot beat DA, deep model won't either.

Kill-switch:
- If ALL tabular models fail to beat DA -> NO_SIGNAL, stop.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import csv
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
import warnings
warnings.filterwarnings('ignore')

# ── Feature columns (forecast only, no leakage) ──────────────────────────

FORECAST_FEATURES = [
    '地方电厂总加预测值',
    '联络线受电负荷预测值',
    '风电总加预测值',
    '光伏总加预测值',
    '核电总加预测值',
    '自备机组总加预测值',
    '试验机组总加预测值',
    '直调负荷预测值',
    '竞价空间预测值',
    '新能源总加预测值',
]

CALENDAR_FEATURES = [
    'hour',
    'dow',
    'month',
    'is_weekend',
    'hour_sin',
    'hour_cos',
]


def load_data(data_path: str) -> pd.DataFrame:
    """Load data with proper encoding."""
    for enc in ['gbk', 'gb18030', 'utf-8']:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            return df
        except:
            continue
    raise ValueError("Cannot load data with any encoding")


def prepare_data(df: pd.DataFrame, target_month: str):
    """
    Prepare train/val/test splits for a target month.
    
    Train: all data before target_month
    Val: last 30 days before target_month
    Test: target_month
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['时刻'])
    
    # Add calendar features
    df['hour'] = df['ds'].dt.hour
    df['dow'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Compute residual
    df['residual'] = df['实时电价'] - df['日前电价']
    
    # Parse target month
    target_start = pd.to_datetime(target_month + '-01')
    if target_month == '2026-01':
        train_end = pd.to_datetime('2025-12-01')
    else:
        train_end = target_start - pd.Timedelta(days=30)
    val_start = train_end
    train_start = df['ds'].min()
    
    # Split
    train_df = df[(df['ds'] >= train_start) & (df['ds'] < train_end)].copy()
    val_df = df[(df['ds'] >= val_start) & (df['ds'] < target_start)].copy()
    test_df = df[df['ds'].dt.to_period('M').astype(str) == target_month].copy()
    
    return train_df, val_df, test_df


def smape_floor50(y_true, y_pred, floor=50):
    """Canonical sMAPE with floor=50."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), floor) + np.maximum(np.abs(y_pred), floor)
    return 2.0 * np.mean(np.abs(y_pred - y_true) / denom) * 100


def select_shrink_gate(residual_pred, da_anchor, rt_actual, alpha_candidates, clip_candidates):
    """Select best alpha/clip on validation set."""
    best_smape = float('inf')
    best_alpha = 0.0
    best_clip = 0.0
    
    for alpha in alpha_candidates:
        for clip in clip_candidates:
            residual_clipped = np.clip(residual_pred, -clip, clip)
            final_pred = da_anchor + alpha * residual_clipped
            smape = smape_floor50(rt_actual, final_pred)
            if smape < best_smape:
                best_smape = smape
                best_alpha = alpha
                best_clip = clip
    
    return best_alpha, best_clip, best_smape


def run_tabular_probe(data_path: str, target_month: str, out_dir: Path):
    """Run tabular probe for a target month."""
    print(f"\n{'=' * 80}")
    print(f"Tabular Residual Probe: {target_month}")
    print(f"{'=' * 80}")
    
    # Load data
    df = load_data(data_path)
    train_df, val_df, test_df = prepare_data(df, target_month)
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Prepare features
    all_features = FORECAST_FEATURES + CALENDAR_FEATURES
    available_features = [f for f in all_features if f in train_df.columns]
    
    # Drop NaN
    train_valid = train_df.dropna(subset=available_features + ['residual']).copy()
    val_valid = val_df.dropna(subset=available_features + ['residual']).copy()
    test_valid = test_df.dropna(subset=available_features + ['residual', '日前电价', '实时电价']).copy()
    
    if len(train_valid) < 100 or len(test_valid) < 10:
        print(f"\n  ERROR: Not enough valid samples")
        return None
    
    # Prepare arrays
    X_train = train_valid[available_features].fillna(0).values
    y_train = train_valid['residual'].values
    
    X_val = val_valid[available_features].fillna(0).values
    y_val = val_valid['residual'].values
    da_val = val_valid['日前电价'].values
    rt_val = val_valid['实时电价'].values
    
    X_test = test_valid[available_features].fillna(0).values
    da_test = test_valid['日前电价'].values
    rt_test = test_valid['实时电价'].values
    
    # DA anchor baselines
    da_smape_test = smape_floor50(rt_test, da_test)
    da_smape_val = smape_floor50(rt_val, da_val)
    
    print(f"\nDA anchor baselines:")
    print(f"  Val sMAPE:   {da_smape_val:.2f}")
    print(f"  Test sMAPE:  {da_smape_test:.2f}")
    
    # Alpha/clip candidates (select on val)
    alpha_candidates = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    clip_candidates = [50, 100, 150, 200, 300]
    
    # ── Models ─────────────────────────────────────────────────────────────
    candidates = []
    
    # Helper function
    def evaluate_model(name, model):
        try:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            
            # Select alpha/clip on val
            best_alpha, best_clip, best_val_smape = select_shrink_gate(
                y_val_pred, da_val, rt_val, alpha_candidates, clip_candidates
            )
            
            # Apply to test
            y_test_pred = model.predict(X_test)
            residual_clipped = np.clip(y_test_pred, -best_clip, best_clip)
            final_test_pred = da_test + best_alpha * residual_clipped
            test_smape = smape_floor50(rt_test, final_test_pred)
            
            improvement = da_smape_test - test_smape
            status = 'KEEP' if improvement >= 0.3 else ('WEAK_KEEP' if improvement > 0 else 'KILL')
            
            print(f"\n  {name}:")
            print(f"    Val sMAPE (shrunk):  {best_val_smape:.2f}")
            print(f"    Test sMAPE:            {test_smape:.2f}")
            print(f"    DA sMAPE:             {da_smape_test:.2f}")
            print(f"    Improvement:           {improvement:+.2f} pp")
            print(f"    Status:                {status}")
            print(f"    Alpha:                 {best_alpha}")
            print(f"    Clip:                  {best_clip}")
            
            candidates.append({
                'model': name,
                'val_smape': best_val_smape,
                'test_smape': test_smape,
                'da_smape': da_smape_test,
                'improvement': improvement,
                'status': status,
                'alpha': best_alpha,
                'clip': best_clip,
            })
        except Exception as e:
            print(f"\n  {name} ERROR: {e}")
    
    # 1. Ridge
    evaluate_model('Ridge', Ridge(alpha=1.0, random_state=42))
    
    # 2. ElasticNet
    evaluate_model('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000))
    
    # 3. HGB
    evaluate_model('HGB', HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.1, max_depth=5, random_state=42
    ))
    
    # 4. Random Forest (small)
    evaluate_model('RF', RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    ))
    
    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    
    if len(candidates) > 0:
        best = max(candidates, key=lambda x: x['improvement'])
        print(f"\nBest model: {best['model']}")
        print(f"  Test sMAPE:  {best['test_smape']:.2f}")
        print(f"  DA sMAPE:     {best['da_smape']:.2f}")
        print(f"  Improvement:  {best['improvement']:+.2f} pp")
        print(f"  Status:        {best['status']}")
        
        # Kill-switch check
        all_kill = all(c['status'] == 'KILL' for c in candidates)
        if all_kill:
            print(f"\n  VERDICT: ALL KILL -> NO_SIGNAL")
            print(f"  Tabular models cannot beat DA anchor.")
            print(f"  DO NOT proceed to deep model.")
        else:
            print(f"\n  VERDICT: Some KEEP/WEAK_KEEP")
            print(f"  Proceed to deep model with caution.")
    else:
        print("\n  VERDICT: NO models ran successfully.")
    
    # Save results
    if len(candidates) > 0:
        with open(out_dir / 'leaderboard.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'model', 'val_smape', 'test_smape', 'da_smape', 'improvement', 'status', 'alpha', 'clip'
            ])
            writer.writeheader()
            for c in candidates:
                writer.writerow(c)
        
        with open(out_dir / 'candidate_decisions.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'status', 'decision'])
            writer.writeheader()
            for c in candidates:
                decision = 'PROCEED' if c['status'] in ['KEEP', 'WEAK_KEEP'] else 'KILL'
                writer.writerow({'model': c['model'], 'status': c['status'], 'decision': decision})
    
    return candidates


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--target-month', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    candidates = run_tabular_probe(args.data_path, args.target_month, out_dir)
    
    print("\n" + "=" * 80)
    print("Phase B Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
