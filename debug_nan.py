"""Debug script to check for NaN values in features."""
import pandas as pd
import numpy as np
from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_features import build_deep_rt_sota_features

# Load data
df = pd.read_csv('D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv', encoding='gbk')
df = df.rename(columns={'时刻': 'ds', '日前电价': 'da_anchor', '实时电价': 'rt_actual'})
df['ds'] = pd.to_datetime(df['ds'])
df = add_business_time_columns(df, timestamp_col='ds')

# Build features
df, manifest = build_deep_rt_sota_features(df, risk_features=False, forecast_features=False)

# Check for NaN values
feature_cols = ['rt_lag_24h', 'rt_lag_48h', 'rt_lag_72h', 'rt_lag_168h',
               'previous_day_rt_mean', 'previous_day_rt_std',
               'previous_7d_same_hour_mean', 'previous_7d_same_hour_std',
               'da_anchor', 'forecast_price', 'anchor_spread',
               'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
               'month_sin', 'month_cos', 'is_weekend', 'period_id']

available = [col for col in feature_cols if col in df.columns]
print(f"Available features: {len(available)}")

# Check NaN counts
nan_counts = df[available].isna().sum()
print("\nNaN counts per feature:")
print(nan_counts)

# Check total NaN rows
nan_rows = df[available].isna().any(axis=1).sum()
print(f"\nTotal rows with NaN: {nan_rows} / {len(df)}")

# Check first few rows
print("\nFirst 5 rows of features:")
print(df[available].head(10))
