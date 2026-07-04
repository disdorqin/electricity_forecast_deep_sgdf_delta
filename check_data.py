"""Check real data structure and date range."""
import pandas as pd

df = pd.read_csv(
    'D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv',
    encoding='gbk'
)

df['ds'] = pd.to_datetime(df['时刻'])

print(f"Shape: {df.shape}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
print(f"Unique days: {df['ds'].dt.date.nunique()}")
print(f"\nColumns (Chinese -> English):")
print(f"  时刻 (timestamp) -> ds")
print(f"  日前电价 (day-ahead price) -> da_anchor")
print(f"  实时电价 (realtime price) -> rt_actual")
print(f"\nSample rt_actual values:")
print(df['实时电价'].head(10).values)
print(f"\nSample da_anchor values:")
print(df['日前电价'].head(10).values)
