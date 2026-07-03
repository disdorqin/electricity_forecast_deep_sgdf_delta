"""Smoke tests for realtime training pipeline.

Covers:
  - RealtimeDayDataset builds from synthetic DataFrame
  - Dataset __getitem__ returns correct keys and shapes
  - build_training_datasets_final produces train/val/test splits
  - collate_fn_final stacks batch items correctly
  - Predict-mode dataset has no targets
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import torch

from models.deep_sgdf_delta.realtime_dataset_final import (
    RealtimeDayDataset,
    collate_fn_final,
    build_training_datasets_final,
)
from models.deep_sgdf_delta.realtime_feature_contract import REQUIRED_FEATURES


# ── Helpers ────────────────────────────────────────────────────────────

def _make_synthetic_df(
    n_days: int = 120,
    start_date: str = "2024-01-01",
    with_rt_actual: bool = True,
) -> pd.DataFrame:
    """Create a synthetic hourly DataFrame for testing.

    Generates *n_days* of business data (24 hours each) starting from
    *start_date*.  Timestamps follow the Shandong business-time convention:
      - hour 01:00 -> business_day = D, hour_business = 1
      - hour 00:00 -> business_day = D-1, hour_business = 24
    """
    rows = []
    base = pd.Timestamp(start_date)

    for day_offset in range(n_days):
        bd = base + pd.Timedelta(days=day_offset)
        for hour in range(1, 25):
            # Map business hour to wall-clock timestamp
            if hour == 24:
                # hour_business 24 -> midnight of next day
                ds = bd + pd.Timedelta(days=1)
            else:
                ds = bd + pd.Timedelta(hours=hour)

            row = {
                "ds": ds,
                "forecast_price": 300.0 + np.random.randn() * 20,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "dow_sin": np.sin(2 * np.pi * bd.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * bd.dayofweek / 7),
                "month_sin": np.sin(2 * np.pi * bd.month / 12),
                "month_cos": np.cos(2 * np.pi * bd.month / 12),
                "is_weekend": float(bd.dayofweek >= 5),
                "is_holiday": 0.0,
                "rt_lag_1h": 295.0 + np.random.randn() * 10,
                "rt_lag_2h": 293.0 + np.random.randn() * 10,
                "rt_lag_3h": 291.0 + np.random.randn() * 10,
                "rt_lag_24h": 290.0 + np.random.randn() * 15,
                "rt_lag_48h": 288.0 + np.random.randn() * 15,
                "rt_mean_6h": 294.0 + np.random.randn() * 8,
                "rt_std_6h": abs(np.random.randn() * 5),
                "rt_mean_24h": 292.0 + np.random.randn() * 12,
                "rt_std_24h": abs(np.random.randn() * 8),
                "delta_lag_24h": np.random.randn() * 10,
                "delta_lag_48h": np.random.randn() * 10,
            }
            if with_rt_actual:
                row["rt_actual"] = row["forecast_price"] + np.random.randn() * 15
            rows.append(row)

    return pd.DataFrame(rows)


# ── Dataset builds ────────────────────────────────────────────────────

class TestDatasetBuilds:
    def test_dataset_builds(self):
        """RealtimeDayDataset builds from synthetic DataFrame."""
        np.random.seed(42)
        df = _make_synthetic_df(n_days=10)
        ds = RealtimeDayDataset(df, mode="train")
        assert ds.n_days > 0, "Dataset should have at least one day"
        assert len(ds) == ds.n_days

    def test_dataset_builds_predict_mode(self):
        """RealtimeDayDataset builds in predict mode without rt_actual."""
        np.random.seed(42)
        df = _make_synthetic_df(n_days=5, with_rt_actual=False)
        ds = RealtimeDayDataset(df, mode="predict")
        assert ds.n_days > 0


# ── Dataset __getitem__ ───────────────────────────────────────────────

class TestDatasetGetitem:
    def test_dataset_getitem(self):
        """Returns correct keys and shapes."""
        np.random.seed(42)
        df = _make_synthetic_df(n_days=10)
        ds = RealtimeDayDataset(df, mode="train")

        sample = ds[0]

        # Check all expected keys
        expected_keys = {
            "features_24h", "delta_target_24", "residual_target_24",
            "mask_24", "period_24", "segment_id", "da_anchor_24",
            "sgdfnet_pred_24", "hour_ids", "business_day",
        }
        assert set(sample.keys()) == expected_keys

        # Check shapes
        assert sample["features_24h"].shape == (24, ds.input_dim)
        assert sample["delta_target_24"].shape == (24,)
        assert sample["residual_target_24"].shape == (24,)
        assert sample["mask_24"].shape == (24,)
        assert sample["period_24"].shape == (24,)
        assert sample["da_anchor_24"].shape == (24,)
        assert sample["sgdfnet_pred_24"].shape == (24,)
        assert sample["hour_ids"].shape == (24,)

        # Check types
        assert sample["features_24h"].dtype == torch.float32
        assert sample["segment_id"].dtype == torch.long
        assert sample["hour_ids"].dtype == torch.int64

        # Hour IDs should be 1-24
        assert sample["hour_ids"].min() == 1
        assert sample["hour_ids"].max() == 24


# ── Walk-forward split ────────────────────────────────────────────────

class TestWalkForwardSplit:
    def test_walk_forward_split(self):
        """build_training_datasets_final returns train/val/test."""
        np.random.seed(42)
        # Create enough data: 200 days covering Jan-Jul 2024
        df = _make_synthetic_df(n_days=200, start_date="2024-01-01")

        train_ds, val_ds, test_ds, manifest = build_training_datasets_final(
            df,
            target_month="2024-06",
            val_days=30,
            train_min_days=30,
        )

        # All three datasets should be non-empty
        assert train_ds.n_days > 0, "Train set should have days"
        assert val_ds.n_days > 0, "Val set should have days"
        assert test_ds.n_days > 0, "Test set should have days"

        # Manifest should have expected keys
        assert "n_train_days" in manifest
        assert "n_val_days" in manifest
        assert "n_test_days" in manifest
        assert manifest["n_train_days"] == train_ds.n_days
        assert manifest["n_val_days"] == val_ds.n_days
        assert manifest["n_test_days"] == test_ds.n_days

    def test_walk_forward_split_insufficient_data(self):
        """Raises ValueError when training data is insufficient."""
        np.random.seed(42)
        # Only 10 days — not enough for train_min_days=90
        df = _make_synthetic_df(n_days=10, start_date="2024-06-01")

        with pytest.raises(ValueError, match="Insufficient"):
            build_training_datasets_final(
                df,
                target_month="2024-06",
                val_days=5,
                train_min_days=90,
            )


# ── Collate function ──────────────────────────────────────────────────

class TestCollateFn:
    def test_collate_fn(self):
        """collate_fn_final stacks correctly."""
        np.random.seed(42)
        df = _make_synthetic_df(n_days=10)
        ds = RealtimeDayDataset(df, mode="train")

        # Grab a batch of 3 samples
        batch = [ds[i] for i in range(min(3, len(ds)))]
        collated = collate_fn_final(batch)

        B = len(batch)
        assert collated["features_24h"].shape == (B, 24, ds.input_dim)
        assert collated["delta_target_24"].shape == (B, 24)
        assert collated["segment_id"].shape == (B,)
        assert collated["da_anchor_24"].shape == (B, 24)
        assert collated["hour_ids"].shape == (B, 24)
        assert collated["business_day"].shape == (B,)


# ── Predict mode ──────────────────────────────────────────────────────

class TestPredictMode:
    def test_predict_mode(self):
        """Predict dataset has no targets (delta/residual are zeros)."""
        np.random.seed(42)
        df = _make_synthetic_df(n_days=5, with_rt_actual=False)
        ds = RealtimeDayDataset(df, mode="predict")

        assert ds.n_days > 0
        sample = ds[0]

        # In predict mode, targets should be all zeros
        assert (sample["delta_target_24"] == 0).all()
        assert (sample["residual_target_24"] == 0).all()

        # Features and anchors should still be populated
        assert sample["da_anchor_24"].abs().sum() > 0
