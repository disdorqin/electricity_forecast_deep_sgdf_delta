"""
DeepRT-SOTA v2 - Data Audit Script.

Audit data pipeline before training:
- Check data coverage
- Check train/val/test split
- Check sequence sample count
- Check feature NaN rate
- Check oracle baseline
- Check synthetic risk features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.business_time import add_business_time_columns

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def audit_data(data_path: str, target_month: str, seq_len_days: int = 7, risk_features: str = 'off'):
    """Audit data pipeline.
    
    Args:
        data_path: Path to CSV data file.
        target_month: Target month (e.g., '2026-02').
        seq_len_days: Sequence length in days.
        risk_features: 'off' | 'real' | 'synthetic'.
        
    Returns:
        Audit result dictionary.
    """
    result = {
        'data_path': data_path,
        'target_month': target_month,
        'seq_len_days': seq_len_days,
        'risk_features': risk_features,
        'audit_time': datetime.now().isoformat(),
        'checks': {},
        'errors': [],
        'warnings': [],
        'passed': False,
    }
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='utf-8')
    
    logger.info(f"Loaded {len(df)} rows")
    
    # Rename columns
    df = df.rename(columns={
        '时刻': 'ds',
        '日前电价': 'da_anchor',
        '实时电价': 'rt_actual',
    })
    
    # Parse timestamp
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Add business time columns
    df = add_business_time_columns(df, timestamp_col='ds')
    
    # Check 1: Raw row count
    result['checks']['raw_row_count'] = len(df)
    logger.info(f"Raw rows: {len(df)}")
    
    # Check 2: Date range
    date_min = df['ds'].min()
    date_max = df['ds'].max()
    result['checks']['date_range'] = {
        'min': date_min.isoformat(),
        'max': date_max.isoformat(),
    }
    logger.info(f"Date range: {date_min} to {date_max}")
    
    # Check 3: Target month rows
    target_start = pd.to_datetime(f"{target_month}-01")
    if target_month == '2026-02':
        target_end = pd.to_datetime('2026-03-01')
    else:
        # Generic: next month
        year, month = map(int, target_month.split('-'))
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1
        target_end = pd.to_datetime(f"{next_year}-{next_month:02d}-01")
    
    target_mask = (df['ds'] >= target_start) & (df['ds'] < target_end)
    target_df = df[target_mask].copy()
    
    result['checks']['target_month'] = {
        'name': target_month,
        'rows': len(target_df),
        'business_days': df.loc[target_mask, 'business_day'].nunique(),
    }
    
    logger.info(f"Target month {target_month}: {len(target_df)} rows, {result['checks']['target_month']['business_days']} business_days")
    
    # Check 4: Each business_day has 24 hours
    target_business_days = target_df['business_day'].unique()
    hours_per_day = target_df.groupby('business_day')['hour_business'].nunique()
    
    incomplete_days = hours_per_day[hours_per_day < 24]
    result['checks']['target_month_24h_check'] = {
        'total_business_days': len(hours_per_day),
        'incomplete_days': len(incomplete_days),
        'incomplete_day_list': incomplete_days.index.tolist()[:10],  # First 10
    }
    
    if len(incomplete_days) > 0:
        result['warnings'].append(f"Target month has {len(incomplete_days)} incomplete days (< 24 hours)")
        logger.warning(f"Incomplete days in target month: {len(incomplete_days)}")
    
    # Check 5: Duplicates
    dup_mask = target_df.duplicated(subset=['business_day', 'hour_business'])
    dup_count = dup_mask.sum()
    result['checks']['duplicates'] = {
        'count': int(dup_count),
        'rows': target_df[dup_mask].index.tolist()[:10],
    }
    
    if dup_count > 0:
        result['errors'].append(f"Found {dup_count} duplicate rows in target month")
        logger.error(f"Duplicate rows found: {dup_count}")
    
    # Check 6: Train/val/test split
    # Ensure business_day is datetime type
    df['business_day'] = pd.to_datetime(df['business_day'])
    
    target_start_date = pd.to_datetime(target_start)
    val_start = target_start_date - pd.Timedelta(days=30)
    val_end = target_start_date
    
    train_mask = df['business_day'] < val_start
    val_mask = (df['business_day'] >= val_start) & (df['business_day'] < val_end)
    test_mask = (df['business_day'] >= target_start_date) & (df['business_day'] < pd.to_datetime(target_end))
    
    train_days = df.loc[train_mask, 'business_day'].nunique()
    val_days = df.loc[val_mask, 'business_day'].nunique()
    test_days = df.loc[test_mask, 'business_day'].nunique()
    
    result['checks']['split'] = {
        'train_days': int(train_days),
        'val_days': int(val_days),
        'test_days': int(test_days),
        'train_rows': int(train_mask.sum()),
        'val_rows': int(val_mask.sum()),
        'test_rows': int(test_mask.sum()),
    }
    
    logger.info(f"Split: train={train_days}d/{train_mask.sum()}r, val={val_days}d/{val_mask.sum()}r, test={test_days}d/{test_mask.sum()}r")
    
    # Check 7: Target NaN rate
    target_nan_rate = df['rt_actual'].isna().mean()
    result['checks']['target_nan_rate'] = float(target_nan_rate)
    
    if target_nan_rate > 0:
        result['warnings'].append(f"Target (rt_actual) has {target_nan_rate*100:.1f}% NaN")
        logger.warning(f"Target NaN rate: {target_nan_rate*100:.1f}%")
    
    # Check 8: DA anchor availability and oracle check
    if 'da_anchor' in df.columns:
        da_coverage = df['da_anchor'].notna().mean()
        result['checks']['da_anchor'] = {
            'coverage': float(da_coverage),
            'mean': float(df['da_anchor'].mean()) if da_coverage > 0 else None,
        }
        
        # Oracle check: da_anchor == rt_actual?
        valid_mask = df['rt_actual'].notna() & df['da_anchor'].notna()
        if valid_mask.sum() > 0:
            is_oracle = np.allclose(df.loc[valid_mask, 'da_anchor'].values, df.loc[valid_mask, 'rt_actual'].values)
            result['checks']['da_anchor']['is_oracle'] = is_oracle
            
            if is_oracle:
                result['errors'].append("DA anchor is ORACLE (da_anchor == rt_actual)")
                logger.error("DA anchor is ORACLE baseline!")
        
        logger.info(f"DA anchor coverage: {da_coverage*100:.1f}%, oracle: {result['checks']['da_anchor'].get('is_oracle', 'unknown')}")
    
    # Check 9: Sequence samples estimation
    # For day-level: need seq_len_days of valid history before each test day
    test_business_days = sorted(df.loc[test_mask, 'business_day'].unique())
    
    day_level_samples = 0
    for day in test_business_days:
        # Check if we have seq_len_days of history
        history_start = day - pd.Timedelta(days=seq_len_days)
        history_mask = (df['business_day'] >= history_start) & (df['business_day'] < day)
        history_days = df.loc[history_mask, 'business_day'].nunique()
        
        if history_days >= seq_len_days:
            # Check if this day has 24 hours
            day_hours = df[(df['business_day'] == day)]['hour_business'].nunique()
            if day_hours == 24:
                day_level_samples += 1
    
    result['checks']['day_level_samples'] = {
        'test_business_days': len(test_business_days),
        'valid_samples': day_level_samples,
        'missing_samples': len(test_business_days) - day_level_samples,
    }
    
    logger.info(f"Day-level samples: {day_level_samples}/{len(test_business_days)} (seq_len={seq_len_days}d)")
    
    # Check 10: Risk features source
    result['checks']['risk_features_source'] = risk_features
    
    if risk_features == 'synthetic':
        result['warnings'].append("Risk features are SYNTHETIC - not for formal metrics")
        logger.warning("Risk features: SYNTHETIC (debug only)")
    elif risk_features == 'real':
        # TODO: Check if real risk features are provided
        result['warnings'].append("Risk features set to 'real' but need to verify path")
        logger.warning("Risk features: REAL (need to verify)")
    else:
        logger.info("Risk features: OFF (formal default)")
    
    # Final verdict
    errors = result['errors']
    warnings = result['warnings']
    
    if len(errors) == 0:
        result['passed'] = True
        result['verdict'] = 'PASS'
    else:
        result['passed'] = False
        result['verdict'] = 'FAIL'
    
    # Save audit result
    output_dir = Path('reports/local/deep_rt_sota/audit_' + target_month.replace('-', '_'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'data_audit.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Generate markdown report
    with open(output_dir / 'data_audit.md', 'w') as f:
        f.write(f"# DeepRT-SOTA v2 Data Audit\n\n")
        f.write(f"**Target Month**: {target_month}\n\n")
        f.write(f"**Timestamp**: {result['audit_time']}\n\n")
        f.write(f"**Verdict**: {'✅ PASS' if result['passed'] else '❌ FAIL'}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Checks\n\n")
        for key, val in result['checks'].items():
            f.write(f"### {key}\n\n")
            f.write(f"```json\n{json.dumps(val, indent=2, default=str)}\n```\n\n")
        
        if len(errors) > 0:
            f.write(f"## Errors\n\n")
            for err in errors:
                f.write(f"- ❌ {err}\n")
            f.write(f"\n")
        
        if len(warnings) > 0:
            f.write(f"## Warnings\n\n")
            for warn in warnings:
                f.write(f"- ⚠️ {warn}\n")
            f.write(f"\n")
    
    logger.info(f"Audit result saved to {output_dir}")
    logger.info(f"Verdict: {result['verdict']}")
    
    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DeepRT-SOTA v2 Data Audit')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--target-month', type=str, required=True, help='Target month (e.g., 2026-02)')
    parser.add_argument('--seq-len-days', type=int, default=7, help='Sequence length in days')
    parser.add_argument('--mode', type=str, default='FULL_DAY', help='Mode (FULL_DAY | INTRADAY)')
    parser.add_argument('--risk-features', type=str, default='off', choices=['off', 'real', 'synthetic'], help='Risk features source')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DeepRT-SOTA v2 - Data Audit")
    logger.info("="*80)
    
    result = audit_data(
        data_path=args.data_path,
        target_month=args.target_month,
        seq_len_days=args.seq_len_days,
        risk_features=args.risk_features,
    )
    
    logger.info("="*80)
    logger.info("Audit complete.")
    logger.info("="*80)


if __name__ == '__main__':
    main()
