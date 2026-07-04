"""
Debug: Check why test set has only 6 samples.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features

# Load data
data_path = 'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv'
df = pd.read_csv(data_path, encoding='gbk')
df = df.rename(columns={'时刻': 'ds', '日前电价': 'da_anchor', '实时电价': 'rt_actual'})
df['ds'] = pd.to_datetime(df['ds'])
df = add_business_time_columns(df, timestamp_col='ds')

# Split
target_month = '2026-02'
train_mask = df['business_day'] < target_month
test_mask = (df['business_day'] >= target_month) & (df['business_day'] < '2026-03')

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

print(f"Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
print(f"Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

# Build features for test set
print("\nBuilding features for test set...")
test_df, manifest = build_deep_rt_sota_features(test_df, risk_features=False, forecast_features=False)

print(f"\nTest feature manifest: {manifest}")

# Check NaN
print(f"\nTest set after feature building: {len(test_df)} rows")
print(f"NaN check per feature:")

feature_cols = [
    'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
    'previous_day_rt_mean', 'previous_day_rt_std',
    'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
    'da_anchor', 'forecast_price', 'anchor_spread',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'is_weekend', 'period_id',
]

for col in feature_cols:
    if col in test_df.columns:
        nan_rate = test_df[col].isna().mean()
        print(f"  {col}: {nan_rate*100:.1f}% NaN ({test_df[col].isna().sum()} rows)")
    else:
        print(f"  {col}: MISSING")

# Check how many rows have all features non-NaN
available_features = [col for col in feature_cols if col in test_df.columns]
valid_rows = test_df[available_features].notna().all(axis=1).sum()
print(f"\nRows with all features non-NaN: {valid_rows}/{len(test_df)}")

# Check day-level
print(f"\nDay-level check:")
test_df_sorted = test_df.sort_values(['business_day', 'hour_business'])
unique_days = sorted(test_df_sorted['business_day'].unique())

valid_day_samples = 0
for day in unique_days:
    day_data = test_df_sorted[test_df_sorted['business_day'] == day]
    
    if len(day_data) != 24:
        print(f"  {day}: ❌ Only {len(day_data)} hours")
        continue
    
    # Check NaN in features
    day_nan = day_data[available_features].isna().any().any()
    
    if day_nan:
        print(f"  {day}: ❌ Has NaN in features")
    else:
        valid_day_samples += 1
        print(f"  {day}: ✅ Valid")

print(f"\nValid day-level samples: {valid_day_samples}/{len(unique_days)}")
