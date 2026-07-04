"""Verify da_anchor is not oracle baseline."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

df = pd.read_csv("../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv", encoding="gbk")
df.columns = ["ds" if c == "时刻" else c for c in df.columns]
df.columns = ["da_anchor" if c == "日前电价" else c for c in df.columns]
df.columns = ["rt_actual" if c == "实时电价" else c for c in df.columns]

# Check if da_anchor == rt_actual (oracle)
is_oracle = np.allclose(df["da_anchor"].values, df["rt_actual"].values, equal_nan=True)
print(f"Is da_anchor == rt_actual (oracle)? {is_oracle}")

# Check correlation
valid_mask = df["da_anchor"].notna() & df["rt_actual"].notna()
corr = np.corrcoef(df.loc[valid_mask, "da_anchor"], df.loc[valid_mask, "rt_actual"])[0, 1]
print(f"Correlation between da_anchor and rt_actual: {corr:.4f}")

# Check 2026-02
df["ds"] = pd.to_datetime(df["ds"])
feb_mask = (df["ds"] >= "2026-02-01") & (df["ds"] < "2026-03-01")
feb_df = df[feb_mask]
feb_corr = np.corrcoef(feb_df["da_anchor"].dropna(), feb_df["rt_actual"].dropna())[0, 1]
print(f"2026-02 correlation: {feb_corr:.4f}")

# Check sMAPE of da_anchor as prediction
from models.deep_sgdf_delta.metrics import smape_floor50
feb_smape = smape_floor50(feb_df["rt_actual"].values, feb_df["da_anchor"].values)
print(f"2026-02 da_anchor sMAPE_floor50: {feb_smape:.4f}")

# Show some examples
print("\nFirst 10 rows of 2026-02:")
print(feb_df[["ds", "da_anchor", "rt_actual"]].head(10))
