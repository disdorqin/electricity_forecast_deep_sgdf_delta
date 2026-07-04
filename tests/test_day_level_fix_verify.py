"""Quick test: Verify day-level test days fix.

This script tests whether the fixed dataset can properly
build sequences for test days using history from before the test period.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.deep_rt_sota_dataset import (
    DeepRTSOTADatasetConfig,
    build_deep_rt_sota_dataset,
)
from models.deep_sgdf_delta.deep_rt_sota_features import (
    build_deep_rt_sota_features,
    get_feature_columns,
)


def test_day_level_test_days():
    """Test that day-level test has >= 27 days for 2026-02."""

    print("=" * 80)
    print("Test: Day-level test days for 2026-02")
    print("=" * 80)

    # Create dummy data: 2025-12-01 to 2026-02-28
    dates = pd.date_range("2025-12-01", "2026-02-28", freq="h")
    n = len(dates)

    df = pd.DataFrame({
        "ds": dates,
        "rt_actual": np.random.rand(n) * 100 + 300,  # 300-400 range
        "da_anchor": np.random.rand(n) * 100 + 300,
    })

    # Add business time columns
    df = add_business_time_columns(df, timestamp_col="ds")

    print(f"\nData: {len(df)} rows, {df['business_day'].nunique()} days")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Split train/test (target month = 2026-02)
    target_start = pd.Timestamp("2026-02-01")
    target_end = pd.Timestamp("2026-03-01")

    train_mask = df["business_day"] < target_start
    test_mask = (df["business_day"] >= target_start) & (df["business_day"] < target_end)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"\nTrain: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    print(f"Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

    # Merge and build features (THIS IS THE KEY FIX)
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df = merged_df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    print("\nBuilding features on merged data...")
    merged_df, feature_manifest = build_deep_rt_sota_features(
        merged_df,
        risk_features=False,
        forecast_features=False,
    )

    # Split back
    train_df = merged_df[merged_df["business_day"] < target_start].copy()
    test_df = merged_df[
        (merged_df["business_day"] >= target_start) & (merged_df["business_day"] < target_end)
    ].copy()

    print(f"\nAfter feature building:")
    print(f"  Train: {len(train_df)} rows, {train_df['business_day'].nunique()} days")
    print(f"  Test: {len(test_df)} rows, {test_df['business_day'].nunique()} days")

    # Check feature NaN
    feature_cols = get_feature_columns(risk_features=False, forecast_features=False)
    feature_cols = [c for c in feature_cols if c in merged_df.columns]

    train_nan = train_df[feature_cols].isna().sum().sum()
    test_nan = test_df[feature_cols].isna().sum().sum()

    print(f"\nFeature NaN:")
    print(f"  Train: {train_nan}")
    print(f"  Test: {test_nan}")

    # Create dataset for test (using merged_df as full_df)
    test_days = sorted(test_df["business_day"].unique())
    print(f"\nTest days: {len(test_days)}")

    config = DeepRTSOTADatasetConfig(
        seq_len_days=7,
        target_mode="direct",
        risk_features=False,
        forecast_features=False,
        target_granularity="day",
    )

    # KEY: Pass merged_df as full_df so test days can access train history
    dataset = build_deep_rt_sota_dataset(
        config=config,
        full_df=merged_df,
        target_days=test_days,
        feature_columns=feature_cols,
    )

    print(f"\nDataset created:")
    print(f"  Target days: {len(test_days)}")
    print(f"  Samples: {len(dataset)}")

    if len(dataset) >= 27:
        print(f"  ✅ PASS: test samples ({len(dataset)}) >= 27")
    else:
        print(f"  ❌ FAIL: test samples ({len(dataset)}) < 27")

    # Check first few samples
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  business_day: {sample['business_day']}")
        print(f"  X_seq shape: {sample['X_seq'].shape}")
        print(f"  y shape: {sample['y'].shape}")

    print("\n" + "=" * 80)
    print("Test complete.")
    print("=" * 80)


if __name__ == "__main__":
    test_day_level_test_days()
