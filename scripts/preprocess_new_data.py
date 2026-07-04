"""
Preprocess new data from shandong_pmos_hourly.csv
- Merge all available features
- Create time features
- Create lag features
- Output preprocessed data for model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def preprocess_data(input_path: str, output_path: str, encoding='gbk'):
    """
    Preprocess shandong_pmos_hourly.csv data.
    
    Args:
        input_path: Path to shandong_pmos_hourly.csv
        output_path: Path to output preprocessed CSV
        encoding: File encoding (default: gbk)
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['时刻'], encoding=encoding)
    df = df.sort_values('时刻').reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} rows, time range: {df['时刻'].min()} to {df['时刻'].max()}")
    
    # Rename columns to English for consistency
    col_map = {
        '时刻': 'times',
        '日前电价': 'da_price',
        '实时电价': 'rt_price',
        '地方电厂总加预测值': 'local_plant_forecast',
        '联络线受电负荷预测值': 'tie_line_load_forecast',
        '风电总加预测值': 'wind_forecast',
        '光伏总加预测值': 'solar_forecast',
        '核电总加预测值': 'nuclear_forecast',
        '自备机组总加预测值': 'self_supply_forecast',
        '试验机组总加预测值': 'test_unit_forecast',
        '直调负荷预测值': 'direct_dispatch_forecast',
        '竞价空间预测值': 'bidding_space_forecast',
        '新能源总加预测值': 'renewable_forecast',
        '地方电厂总加实际值': 'local_plant_actual',
        '联络线受电负荷实际值': 'tie_line_load_actual',
        '风电总加实际值': 'wind_actual',
        '光伏总加实际值': 'solar_actual',
        '核电总加实际值': 'nuclear_actual',
        '自备机组总加实际值': 'self_supply_actual',
        '试验机组总加实际值': 'test_unit_actual',
        '直调负荷实际值': 'direct_dispatch_actual',
        '竞价空间实际值': 'bidding_space_actual',
        '新能源总加实际值': 'renewable_actual',
    }
    df = df.rename(columns=col_map)
    
    # Create time features
    print("Creating time features...")
    df['hour'] = df['times'].dt.hour
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Create lag features for target (rt_price)
    print("Creating lag features...")
    for lag in [24, 48, 72, 168]:  # 1 day, 2 days, 3 days, 1 week
        df[f'rt_price_lag_{lag}h'] = df['rt_price'].shift(lag)
        df[f'da_price_lag_{lag}h'] = df['da_price'].shift(lag)
    
    # Create lag features for important exogenous features
    important_features = ['bidding_space_forecast', 'direct_dispatch_forecast', 
                          'wind_forecast', 'solar_forecast']
    for feat in important_features:
        if feat in df.columns:
            df[f'{feat}_lag_24h'] = df[feat].shift(24)
            df[f'{feat}_lag_168h'] = df[feat].shift(168)
    
    # Create rolling statistics for rt_price
    print("Creating rolling statistics...")
    for window in [24, 48, 168]:  # 1 day, 2 days, 1 week
        df[f'rt_price_rolling_mean_{window}h'] = df['rt_price'].rolling(window=window, min_periods=1).mean().shift(1)
        df[f'rt_price_rolling_std_{window}h'] = df['rt_price'].rolling(window=window, min_periods=1).std().shift(1)
    
    # Create rolling statistics for important exogenous features
    for feat in important_features:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_24h'] = df[feat].rolling(window=24, min_periods=1).mean().shift(1)
    
    # Compute residual (RT - DA)
    df['residual'] = df['rt_price'] - df['da_price']
    
    # Drop rows with NaN in key columns
    key_cols = ['rt_price', 'da_price'] + important_features
    df_clean = df.dropna(subset=key_cols).reset_index(drop=True)
    
    print(f"After preprocessing: {len(df_clean)} rows (dropped {len(df) - len(df_clean)} rows with NaN)")
    
    # Save preprocessed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nPreprocessed data saved to {output_path}")
    print(f"Columns: {list(df_clean.columns)}")
    print(f"Time range: {df_clean['times'].min()} to {df_clean['times'].max()}")
    
    return df_clean

if __name__ == '__main__':
    input_path = r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\data\shandong_pmos_hourly.csv"
    output_path = r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\deep_model_for_electricity\data\preprocessed_data.csv"
    
    df = preprocess_data(input_path, output_path)
    
    # Quick EDA
    print("\n=== Quick EDA ===")
    print("RT price stats:")
    print(df['rt_price'].describe())
    print("\nDA price stats:")
    print(df['da_price'].describe())
    print("\nResidual stats:")
    print(df['residual'].describe())
    
    # Check correlation between features and residual
    print("\n=== Correlation with residual (top 10) ===")
    feature_cols = [col for col in df.columns if col not in ['times', 'rt_price', 'da_price', 'residual', 'hour', 'dayofweek', 'month']]
    correlations = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.int64]:
            valid = df[[col, 'residual']].dropna()
            if len(valid) > 100:
                corr = valid[col].corr(valid['residual'])
                correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for col, corr in sorted_corrs[:10]:
        print(f"  {col}: {corr:.4f}")
