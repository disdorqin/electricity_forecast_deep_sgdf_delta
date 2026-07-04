import pandas as pd
import numpy as np

# Try reading the data
for enc in ['gbk', 'gb18030', 'utf-8']:
    try:
        df = pd.read_csv('../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv', encoding=enc)
        print(f'Encoding: {enc}')
        print(f'Shape: {df.shape}')
        print(f'\nColumns ({len(df.columns)}):')
        for i, col in enumerate(df.columns):
            print(f'  {i}: {col}')
        break
    except Exception as e:
        print(f'{enc}: {e}')
        continue

# Check data types and sample values
print(f'\n\ndtypes:')
print(df.dtypes.head(20))

print(f'\nSample values (first row):')
for col in df.columns[:15]:
    print(f'  {col}: {df[col].iloc[0]}')

print(f'\nSample values (last row):')
for col in df.columns[:15]:
    print(f'  {col}: {df[col].iloc[-1]}')
