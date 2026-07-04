"""Diagnostic: Compare DA sMAPE between implementations"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
for enc in ['gbk', 'gb18030', 'utf-8']:
    try:
        df = pd.read_csv('../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv', encoding=enc)
        print(f'Encoding: {enc}')
        break
    except:
        continue

# Rename columns
df = df.rename(columns={'时刻': 'ds', '日前电价': 'da_anchor', '实时电价': 'rt_actual'})
df['ds'] = pd.to_datetime(df['ds'])

# Filter to 2026-02
test_df = df[(df['ds'] >= '2026-02-01') & (df['ds'] < '2026-03-01')].copy()

rt = test_df['rt_actual'].dropna().values
da = test_df['da_anchor'].dropna().values

print(f'\n2026-02 test set:')
print(f'  Samples: {len(rt)}')
print(f'  RT mean: {rt.mean():.2f}, std: {rt.std():.2f}')
print(f'  DA mean: {da.mean():.2f}, std: {da.std():.2f}')

# sMAPE implementations
def smape_floor50_v1(y_true, y_pred, floor=50):
    """Implementation from metrics.py"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), floor) + np.maximum(np.abs(y_pred), floor)
    return 2.0 * np.mean(np.abs(y_pred - y_true) / denom) * 100

def smape_floor50_v2(y_true, y_pred, floor=50):
    """Alternative implementation"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

# Test both
print(f'\nDA sMAPE (predict RT using DA):')
print(f'  v1 (floor=50): {smape_floor50_v1(rt, da):.4f}')
print(f'  v2 (no floor): {smape_floor50_v2(rt, da):.4f}')

# Check if floor=50 is the issue
for floor in [0, 10, 50, 100]:
    smape = smape_floor50_v1(rt, da, floor=floor)
    print(f'  v1 (floor={floor}): {smape:.4f}')

# Also check: are we using day-level or hourly?
print(f'\nDay-level (24h avg):')
rt_day = test_df.groupby(test_df['ds'].dt.date)['rt_actual'].mean().dropna().values
da_day = test_df.groupby(test_df['ds'].dt.date)['da_anchor'].mean().dropna().values
print(f'  Day samples: {len(rt_day)}')
print(f'  DA sMAPE (day-level): {smape_floor50_v1(rt_day, da_day):.4f}')
