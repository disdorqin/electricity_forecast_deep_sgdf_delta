"""
Debug residual dataset empty bug.
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

print(f"Train: {len(train_df)} rows")
print(f"Test: {len(test_df)} rows")

# Build features (without risk features)
print("\nBuilding features (without risk features)...")
train_df, manifest = build_deep_rt_sota_features(train_df, risk_features=False, forecast_features=False)
test_df, _ = build_deep_rt_sota_features(test_df, risk_features=False, forecast_features=False)

print(f"After building features:")
print(f"  Train: {len(train_df)} rows")
print(f"  Test: {len(test_df)} rows")

# Check residual target
print("\nChecking residual target (rt_actual - da_anchor)...")
print(f"  Train da_anchor coverage: {train_df['da_anchor'].notna().mean()*100:.1f}%")
print(f"  Test da_anchor coverage: {test_df['da_anchor'].notna().mean()*100:.1f}%")

# Check NaN in features
feature_cols = [
    'rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
    'previous_day_rt_mean', 'previous_day_rt_std',
    'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
    'da_anchor', 'forecast_price', 'anchor_spread',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'is_weekend', 'period_id',
]

available_features = [col for col in feature_cols if col in train_df.columns]
print(f"\nAvailable features: {len(available_features)}")

# Create hourly samples (residual target)
print("\nCreating hourly samples (residual target)...")
X_train = []
y_train_residual = []
y_train_actual = []

for idx in range(len(train_df)):
    row = train_df.iloc[idx]
    
    # Skip if any feature is NaN
    if row[available_features].isna().any():
        continue
    
    # Skip if rt_actual or da_anchor is NaN
    if pd.isna(row['rt_actual']) or pd.isna(row['da_anchor']):
        continue
    
    X_train.append(row[available_features].values)
    y_train_residual.append(row['rt_actual'] - row['da_anchor'])
    y_train_actual.append(row['rt_actual'])

print(f"  Train: X={len(X_train)}, y_residual={len(y_train_residual)}")

# Test set
X_test = []
y_test_residual = []
y_test_actual = []

for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    
    # Skip if any feature is NaN
    if row[available_features].isna().any():
        continue
    
    # Skip if rt_actual or da_anchor is NaN
    if pd.isna(row['rt_actual']) or pd.isna(row['da_anchor']):
        continue
    
    X_test.append(row[available_features].values)
    y_test_residual.append(row['rt_actual'] - row['da_anchor'])
    y_test_actual.append(row['rt_actual'])

print(f"  Test: X={len(X_test)}, y_residual={len(y_test_residual)}")

# Check why empty
print("\nChecking why empty...")
print(f"  Train rows after feature building: {len(train_df)}")
print(f"  Train rows with all features non-NaN: {train_df[available_features].notna().all(axis=1).sum()}")

# Check first few rows
print("\nFirst 5 rows (train):")
for i in range(min(5, len(train_df))):
    row = train_df.iloc[i]
    has_nan = row[available_features].isna().any()
    print(f"  {i}: nan={has_nan}, rt_actual={row['rt_actual']}, da_anchor={row['da_anchor']}")
