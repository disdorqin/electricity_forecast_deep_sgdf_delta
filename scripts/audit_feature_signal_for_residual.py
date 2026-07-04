"""
Phase A: Feature Signal Audit for Residual Prediction
====================================================

Audit all available non-leakage features in the dataset to see
if ANY feature can explain residual = rt_actual - da_anchor.

If no feature has signal, deep model won't help.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ── Column Mapping (Chinese -> English) ────────────────────────────────────────

# These are the columns in the dataset
ALL_COLUMNS = [
    '时刻',  # 0: timestamp
    '日前电价',  # 1: da_anchor
    '实时电价',  # 2: rt_actual (TARGET - must exclude)
    '地方电厂总加预测值',  # 3: local plant forecast
    '联络线受电负荷预测值',  # 4: tie-line load forecast
    '风电总加预测值',  # 5: wind forecast
    '光伏总加预测值',  # 6: solar forecast
    '核电总加预测值',  # 7: nuclear forecast
    '自备机组总加预测值',  # 8: self-owned unit forecast
    '试验机组总加预测值',  # 9: test unit forecast
    '直调负荷预测值',  # 10: direct dispatch load forecast
    '竞价空间预测值',  # 11: bidding space forecast
    '新能源总加预测值',  # 12: renewable total forecast
    '地方电厂总加实际值',  # 13: local plant actual (LEAKAGE if target month)
    '联络线受电负荷实际值',  # 14: tie-line load actual (LEAKAGE)
    '风电总加实际值',  # 15: wind actual (LEAKAGE)
    '光伏总加实际值',  # 16: solar actual (LEAKAGE)
    '核电总加实际值',  # 17: nuclear actual (LEAKAGE)
    '自备机组总加实际值',  # 18: self-owned actual (LEAKAGE)
    '试验机组总加实际值',  # 19: test unit actual (LEAKAGE)
    '直调负荷实际值',  # 20: direct dispatch actual (LEAKAGE)
    '竞价空间实际值',  # 21: bidding space actual (LEAKAGE)
    '新能源总加实际值',  # 22: renewable actual (LEAKAGE)
]

# Features that are LEGITIMATELY available before RT price is known
# These are FORECAST values (predicted before the day)
EXOGENOUS_FORECAST_FEATURES = [
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

# Features that are ACTUAL values (only available AFTER the fact)
# These are LEAKAGE if used to predict RT price
LEAKAGE_COLUMNS = [
    '实时电价',  # rt_actual (the target itself)
    '地方电厂总加实际值',
    '联络线受电负荷实际值',
    '风电总加实际值',
    '光伏总加实际值',
    '核电总加实际值',
    '自备机组总加实际值',
    '试验机组总加实际值',
    '直调负荷实际值',
    '竞价空间实际值',
    '新能源总加实际值',
]

# Feature groups for grouped analysis
FEATURE_GROUPS = {
    'load_forecast': [
        '直调负荷预测值',
        '地方电厂总加预测值',
        '自备机组总加预测值',
    ],
    'renewable_forecast': [
        '风电总加预测值',
        '光伏总加预测值',
        '新能源总加预测值',
    ],
    'supply_demand': [
        '竞价空间预测值',
        '联络线受电负荷预测值',
    ],
    'generation_mix': [
        '核电总加预测值',
        '试验机组总加预测值',
    ],
}


def load_data(data_path: str) -> pd.DataFrame:
    """Load data with proper encoding."""
    for enc in ['gbk', 'gb18030', 'utf-8']:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            print(f"  Loaded with encoding: {enc}")
            return df
        except:
            continue
    raise ValueError("Cannot load data with any encoding")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features."""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['时刻'])
    df['hour'] = df['ds'].dt.hour
    df['dow'] = df['ds'].dt.dayofweek  # 0=Monday
    df['month'] = df['ds'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def compute_residual(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residual = rt_actual - da_anchor."""
    df = df.copy()
    df['residual'] = df['实时电价'] - df['日前电价']
    return df


def audit_feature_signal(df: pd.DataFrame, target_months: list) -> dict:
    """
    Audit all features for signal with residual.
    
    Args:
        df: Full dataframe
        target_months: List of target months to audit
    
    Returns:
        dict with audit results
    """
    results = {
        'feature_inventory': [],
        'feature_signal_scores': [],
        'feature_group_scores': [],
        'leakage_excluded': LEAKAGE_COLUMNS,
    }
    
    # Add time features
    df = add_time_features(df)
    df = compute_residual(df)
    
    # Filter to months before target for training
    target_months_dt = pd.to_datetime(target_months)
    max_target_month = max(target_months_dt)
    
    # Use data up to max_target_month for signal analysis
    # (exclude target month itself to avoid overfitting assessment)
    train_df = df[df['ds'] < max_target_month].copy()
    
    print(f"\n  Train period: {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"  Target months: {target_months}")
    
    # ── 1. Feature Inventory ─────────────────────────────────────────────────
    print("\n  Building feature inventory...")
    for col in df.columns:
        if col in ['时刻', 'ds', 'hour_sin', 'hour_cos']:
            continue
        
        info = {
            'column': col,
            'is_leakage': col in LEAKAGE_COLUMNS,
            'is_forecast_feature': col in EXOGENOUS_FORECAST_FEATURES,
            'is_calendar': col in ['hour', 'dow', 'month', 'is_weekend'],
            'nan_rate': float(df[col].isna().mean()),
            'coverage': float(df[col].notna().mean()),
        }
        
        # Check if column has any variation
        if df[col].notna().sum() > 10:
            info['std'] = float(df[col].std())
            info['mean'] = float(df[col].mean())
        
        results['feature_inventory'].append(info)
    
    # ── 2. Individual Feature Signal ────────────────────────────────────────
    print("\n  Computing individual feature signal...")
    
    # Prepare train data (exclude rows with NaN residual)
    valid_train = train_df.dropna(subset=['residual', '日前电价'])
    
    for col in EXOGENOUS_FORECAST_FEATURES + ['hour', 'dow', 'month']:
        if col not in valid_train.columns:
            continue
        
        # Drop NaN in this feature
        valid = valid_train.dropna(subset=[col, 'residual'])
        if len(valid) < 100:
            continue
        
        # Pearson correlation
        pearson_corr = float(valid[col].corr(valid['residual']))
        
        # Spearman correlation
        spearman_corr, _ = spearmanr(valid[col], valid['residual'])
        spearman_corr = float(spearman_corr)
        
        # Mutual information
        try:
            mi = mutual_info_regression(
                valid[[col]].fillna(0), 
                valid['residual'], 
                random_state=42
            )[0]
            mi = float(mi)
        except:
            mi = 0.0
        
        results['feature_signal_scores'].append({
            'feature': col,
            'pearson_corr_with_residual': pearson_corr,
            'spearman_corr_with_residual': spearman_corr,
            'mutual_info_with_residual': mi,
            'abs_pearson': abs(pearson_corr),
            'nan_rate': float(valid[col].isna().mean()),
            'n_samples': len(valid),
        })
    
    # Sort by absolute pearson correlation
    results['feature_signal_scores'].sort(key=lambda x: x['abs_pearson'], reverse=True)
    
    # ── 3. Feature Group Signal (HGB probe) ───────────────────────────────
    print("\n  Running feature group probe (HGB regression)...")
    
    for group_name, group_features in FEATURE_GROUPS.items():
        # Check all features in group are available
        available_features = [f for f in group_features if f in valid_train.columns]
        if len(available_features) < 1:
            continue
        
        # Prepare data
        valid = valid_train.dropna(subset=available_features + ['residual'])
        if len(valid) < 200:
            continue
        
        X = valid[available_features].fillna(0).values
        y = valid['residual'].values
        
        # Train HGB
        try:
            model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            )
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Compute sMAPE of (DA + residual_pred) vs RT
            da = valid['日前电价'].values
            rt = valid['实时电价'].values
            final_pred = da + y_pred
            
            from models.deep_sgdf_delta.metrics import smape_floor50
            smape = smape_floor50(rt, final_pred)
            da_smape = smape_floor50(rt, da)
            
            results['feature_group_scores'].append({
                'feature_group': group_name,
                'features': available_features,
                'hgb_residual_train_smape': float(smape),
                'da_anchor_train_smape': float(da_smape),
                'improvement_vs_da': float(da_smape - smape),
            })
        except Exception as e:
            print(f"    Error in group {group_name}: {e}")
            continue
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--target-months', type=str, required=True,
                        help='Comma-separated list of target months (e.g., 2026-02,2026-03)')
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()
    
    target_months = [m.strip() for m in args.target_months.split(',')]
    
    print("=" * 80)
    print("Phase A: Feature Signal Audit for Residual Prediction")
    print("=" * 80)
    print(f"\nData path: {args.data_path}")
    print(f"Target months: {target_months}")
    print(f"Out dir: {args.out_dir}")
    
    # Load data
    print("\nLoading data...")
    df = load_data(args.data_path)
    print(f"  Shape: {df.shape}")
    
    # Run audit
    print("\nRunning feature signal audit...")
    results = audit_feature_signal(df, target_months)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Save feature_inventory.csv ──────────────────────────────────────────
    inventory_df = pd.DataFrame(results['feature_inventory'])
    inventory_df.to_csv(out_dir / 'feature_inventory.csv', index=False, encoding='utf-8-sig')
    print(f"\n  Saved feature_inventory.csv ({len(inventory_df)} features)")
    
    # ── Save feature_signal_score.csv ───────────────────────────────────────
    signal_df = pd.DataFrame(results['feature_signal_scores'])
    if len(signal_df) > 0:
        signal_df.to_csv(out_dir / 'feature_signal_score.csv', index=False, encoding='utf-8-sig')
        print(f"  Saved feature_signal_score.csv ({len(signal_df)} features)")
    
    # ── Save feature_group_score.csv ────────────────────────────────────────
    group_df = pd.DataFrame(results['feature_group_scores'])
    if len(group_df) > 0:
        group_df.to_csv(out_dir / 'feature_group_score.csv', index=False, encoding='utf-8-sig')
        print(f"  Saved feature_group_score.csv ({len(group_df)} groups)")
    
    # ── Write report ────────────────────────────────────────────────────────
    report_path = out_dir / 'feature_signal_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Feature Signal Audit Report\n\n")
        f.write(f"Target months: {', '.join(target_months)}\n\n")
        
        f.write("## Leakage Excluded Columns\n\n")
        for col in results['leakage_excluded']:
            f.write(f"- {col}\n")
        f.write("\n")
        
        f.write("## Top Features by |Pearson Correlation with Residual|\n\n")
        if len(signal_df) > 0:
            f.write("| Feature | Pearson Corr | Spearman Corr | Mutual Info | NaN Rate |\n")
            f.write("|---------|---------------|----------------|--------------|----------|\n")
            for row in results['feature_signal_scores'][:15]:
                f.write(f"| {row['feature']} | {row['pearson_corr_with_residual']:.4f} | "
                        f"{row['spearman_corr_with_residual']:.4f} | "
                        f"{row['mutual_info_with_residual']:.4f} | {row['nan_rate']:.2%} |\n")
        f.write("\n")
        
        f.write("## Feature Group HGB Probe (Train sMAPE)\n\n")
        if len(group_df) > 0:
            f.write("| Group | HGB sMAPE | DA sMAPE | Improvement |\n")
            f.write("|-------|-----------|----------|-------------|\n")
            for row in results['feature_group_scores']:
                f.write(f"| {row['feature_group']} | {row['hgb_residual_train_smape']:.2f} | "
                        f"{row['da_anchor_train_smape']:.2f} | "
                        f"{row['improvement_vs_da']:+.2f} |\n")
        f.write("\n")
        
        # Verdict
        f.write("## Verdict\n\n")
        if len(group_df) > 0:
            best_improvement = max([r['improvement_vs_da'] for r in results['feature_group_scores']])
            if best_improvement > 0.3:
                f.write(f"**SIGNAL_FOUND**: Best improvement = {best_improvement:.2f} pp\n\n")
                f.write("Proceed to Phase B (Tabular Probe).\n")
            elif best_improvement > 0:
                f.write(f"**WEAK_SIGNAL**: Best improvement = {best_improvement:.2f} pp\n\n")
                f.write("Proceed with caution. May not generalize.\n")
            else:
                f.write(f"**NO_SIGNAL**: Best improvement = {best_improvement:.2f} pp\n\n")
                f.write("Feature groups cannot beat DA anchor. Consider adding external features.\n")
        else:
            f.write("**NO_SIGNAL**: No feature groups evaluated successfully.\n")
    
    print(f"\n  Saved feature_signal_report.md")
    print("\n" + "=" * 80)
    print("Phase A Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
