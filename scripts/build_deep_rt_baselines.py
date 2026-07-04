"""
DeepRT-SOTA v2 - Baseline Leaderboard.

Build baseline predictions for comparison:
1. naive_prev_day_same_hour
2. naive_prev_7d_same_hour_mean
3. naive_prev_14d_same_hour_mean
4. calendar_hour_mean_train
5. da_anchor (if valid non-oracle)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.metrics import smape_floor50

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def build_baselines(df: pd.DataFrame, target_month: str):
    """Build baseline predictions.
    
    Args:
        df: Preprocessed DataFrame with columns ['business_day', 'hour_business', 'rt_actual', 'da_anchor'].
        target_month: Target month (e.g., '2026-02').
        
    Returns:
        Dictionary with baseline predictions and metrics.
    """
    results = {}
    
    # Split train/test
    train_mask = df['business_day'] < target_month
    test_mask = (df['business_day'] >= target_month) & (df['business_day'] < target_month[:4] + '-' + str(int(target_month[5:7]) + 1).zfill(2) if int(target_month[5:7]) < 12 else target_month[:4] + '-01')
    
    # Simpler: use 2026-03-01 as end
    if target_month == '2026-02':
        test_end = '2026-03-01'
    else:
        # Generic
        year, month = map(int, target_month.split('-'))
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1
        test_end = f"{next_year}-{next_month:02d}-01"
    
    test_mask = (df['business_day'] >= target_month) & (df['business_day'] < test_end)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    logger.info(f"Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")
    
    # Sort by time
    train_df = train_df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    test_df = test_df.sort_values(['business_day', 'hour_business']).reset_index(drop=True)
    
    # 1. Naive previous day same hour
    logger.info("\nBuilding baseline: naive_prev_day_same_hour...")
    train_df['rt_lag_24h'] = train_df.groupby('hour_business')['rt_actual'].shift(24)
    
    # For test set, use previous day same hour from train or test itself
    test_predictions = []
    test_actual = []
    test_dates = []
    
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        business_day = row['business_day']
        hour = row['hour_business']
        
        # Look for previous day same hour
        prev_day = business_day - pd.Timedelta(days=1)
        prev_row = train_df[(train_df['business_day'] == prev_day) & (train_df['hour_business'] == hour)]
        
        if len(prev_row) > 0:
            pred = prev_row['rt_actual'].values[0]
        else:
            # Use train mean for this hour
            train_hour_mean = train_df[train_df['hour_business'] == hour]['rt_actual'].mean()
            pred = train_hour_mean
        
        test_predictions.append(pred)
        test_actual.append(row['rt_actual'])
        test_dates.append(business_day)
    
    test_predictions = np.array(test_predictions)
    test_actual = np.array(test_actual)
    
    # Compute metrics
    mae = mean_absolute_error(test_actual, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
    smape = smape_floor50(test_actual, test_predictions)
    
    results['naive_prev_day_same_hour'] = {
        'smape_floor50': float(smape),
        'mae': float(mae),
        'rmse': float(rmse),
        'description': 'Previous day same hour',
        'n_test': len(test_actual),
    }
    
    logger.info(f"  sMAPE: {smape:.4f}, MAE: {mae:.4f}")
    
    # 2. Naive previous 7-day same hour mean
    logger.info("\nBuilding baseline: naive_prev_7d_same_hour_mean...")
    
    test_predictions_7d = []
    
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        business_day = row['business_day']
        hour = row['hour_business']
        
        # Look for previous 7 days same hour
        hist_vals = []
        for offset in range(1, 8):
            prev_day = business_day - pd.Timedelta(days=offset)
            prev_row = train_df[(train_df['business_day'] == prev_day) & (train_df['hour_business'] == hour)]
            
            if len(prev_row) > 0:
                hist_vals.append(prev_row['rt_actual'].values[0])
        
        if len(hist_vals) > 0:
            pred = np.mean(hist_vals)
        else:
            # Use train mean for this hour
            train_hour_mean = train_df[train_df['hour_business'] == hour]['rt_actual'].mean()
            pred = train_hour_mean
        
        test_predictions_7d.append(pred)
    
    test_predictions_7d = np.array(test_predictions_7d)
    
    mae_7d = mean_absolute_error(test_actual, test_predictions_7d)
    rmse_7d = np.sqrt(mean_squared_error(test_actual, test_predictions_7d))
    smape_7d = smape_floor50(test_actual, test_predictions_7d)
    
    results['naive_prev_7d_same_hour_mean'] = {
        'smape_floor50': float(smape_7d),
        'mae': float(mae_7d),
        'rmse': float(rmse_7d),
        'description': 'Previous 7-day same hour mean',
        'n_test': len(test_actual),
    }
    
    logger.info(f"  sMAPE: {smape_7d:.4f}, MAE: {mae_7d:.4f}")
    
    # 3. DA anchor (if valid)
    logger.info("\nChecking DA anchor baseline...")
    
    if 'da_anchor' in test_df.columns:
        da_coverage = test_df['da_anchor'].notna().mean()
        
        if da_coverage > 0.95:
            # Check oracle
            valid_mask = test_df['rt_actual'].notna() & test_df['da_anchor'].notna()
            is_oracle = False
            
            if valid_mask.sum() > 0:
                is_oracle = np.allclose(
                    test_df.loc[valid_mask, 'da_anchor'].values,
                    test_df.loc[valid_mask, 'rt_actual'].values
                )
            
            if not is_oracle:
                da_pred = test_df['da_anchor'].values[:len(test_actual)]
                da_mae = mean_absolute_error(test_actual, da_pred)
                da_rmse = np.sqrt(mean_squared_error(test_actual, da_pred))
                da_smape = smape_floor50(test_actual, da_pred)
                
                results['da_anchor'] = {
                    'smape_floor50': float(da_smape),
                    'mae': float(da_mae),
                    'rmse': float(da_rmse),
                    'description': 'Day-ahead price (non-oracle)',
                    'n_test': len(test_actual),
                    'coverage': float(da_coverage),
                    'is_oracle': is_oracle,
                }
                
                logger.info(f"  sMAPE: {da_smape:.4f}, MAE: {da_mae:.4f}")
                logger.info(f"  Coverage: {da_coverage*100:.1f}%, Oracle: {is_oracle}")
            else:
                logger.warning("  DA anchor is ORACLE, skipping...")
        else:
            logger.warning(f"  DA anchor coverage too low: {da_coverage*100:.1f}%")
    
    return results, test_df, test_actual


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DeepRT-SOTA v2 Baseline Leaderboard')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--target-month', type=str, required=True, help='Target month (e.g., 2026-02)')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DeepRT-SOTA v2 - Baseline Leaderboard")
    logger.info("="*80)
    
    # Load data
    data_path = args.data_path
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, encoding='gbk')
    logger.info(f"Loaded {len(df)} rows")
    
    # Rename and preprocess
    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })
    df['ds'] = pd.to_datetime(df['ds'])
    
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    df = add_business_time_columns(df, timestamp_col='ds')
    
    # Build baselines
    results, test_df, test_actual = build_baselines(df, args.target_month)
    
    # Save results
    if args.out_dir is None:
        args.out_dir = f"reports/local/deep_rt_sota/baselines_{args.target_month.replace('-', '_')}"
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    with open(output_dir / 'baseline_report.md', 'w') as f:
        f.write(f"# DeepRT-SOTA v2 Baseline Leaderboard\n\n")
        f.write(f"**Target Month**: {args.target_month}\n\n")
        f.write(f"**Test Samples**: {len(test_actual)}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Baselines\n\n")
        f.write(f"| Baseline | sMAPE_floor50 | MAE | RMSE | N Test |\n")
        f.write(f"|----------|----------------|-----|------|--------|\n")
        
        for name, metrics in results.items():
            f.write(f"| {name} | {metrics['smape_floor50']:.4f} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['n_test']} |\n")
        
        f.write(f"\n---\n\n")
        f.write(f"## Details\n\n")
        
        for name, metrics in results.items():
            f.write(f"### {name}\n\n")
            f.write(f"- **Description**: {metrics['description']}\n")
            f.write(f"- **sMAPE_floor50**: {metrics['smape_floor50']:.4f}\n")
            f.write(f"- **MAE**: {metrics['mae']:.4f}\n")
            f.write(f"- **RMSE**: {metrics['rmse']:.4f}\n")
            f.write(f"- **N Test**: {metrics['n_test']}\n\n")
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("="*80)
    logger.info("Baseline leaderboard complete.")
    logger.info("="*80)
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE LEADERBOARD SUMMARY")
    print("="*80)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  sMAPE_floor50: {metrics['smape_floor50']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
