"""Diagnose why model can't beat DA anchor."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.metrics import smape_floor50

# Load data
for enc in ['gbk', 'gb18030', 'utf-8']:
    try:
        df = pd.read_csv('../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv', encoding=enc)
        break
    except:
        continue

if '时刻' in df.columns:
    df = df.rename(columns={'时刻':'ds', '日前电价':'da_anchor', '实时电价':'rt_actual'})
df['ds'] = pd.to_datetime(df['ds'])
df = add_business_time_columns(df, timestamp_col='ds')

# Diagnostics for 2026-02
test_df = df[(df['business_day'] >= '2026-02-01') & (df['business_day'] < '2026-03-01')].copy()

rt = test_df['rt_actual'].values
da = test_df['da_anchor'].values
residual = rt - da

print('=== 2026-02 Diagnostics ===')
print(f'Samples: {len(rt)}')
print(f'RT actual mean: {rt.mean():.2f}, std: {rt.std():.2f}')
print(f'DA anchor mean: {da.mean():.2f}, std: {da.std():.2f}')
print(f'Residual mean: {residual.mean():.2f}, std: {residual.std():.2f}')
print(f'|Residual| mean: {np.abs(residual).mean():.2f}')
print(f'Correlation(RT, DA): {np.corrcoef(rt, da)[0,1]:.4f}')
print(f'Correlation(Residual, DA): {np.corrcoef(residual, da)[0,1]:.4f}')
print()
print('DA anchor sMAPE:')
smape_da = smape_floor50(rt, da)
print(f'  sMAPE(rt, da) = {smape_da:.4f}')
print()
print('If model predicts residual perfectly:')
perfect_pred = rt
smape_perfect = smape_floor50(rt, perfect_pred)
print(f'  sMAPE(rt, rt) = {smape_perfect:.4f}')
print()
print('If model predicts residual = 0 (use DA directly):')
smape_da_2 = smape_floor50(rt, da)
print(f'  sMAPE(rt, da) = {smape_da_2:.4f}')
print()
print('Residual lag-24h correlation:')
# Compute residual lag-24h correlation
df_sorted = df.sort_values('ds').reset_index(drop=True)
df_sorted['residual'] = df_sorted['rt_actual'] - df_sorted['da_anchor']
df_sorted['residual_lag_24h'] = df_sorted['residual'].shift(24)
mask_2026_02 = (df_sorted['business_day'] >= '2026-02-01') & (df_sorted['business_day'] < '2026-03-01')
df_test = df_sorted[mask_2026_02].dropna()
if len(df_test) > 0:
    corr = np.corrcoef(df_test['residual'], df_test['residual_lag_24h'])[0,1]
    print(f'  Corr(residual, residual_lag_24h): {corr:.4f}')
else:
    print('  No data after dropna')
