"""Tests for residual_baselines.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.residual_baselines import (
    baseline_sgdfnet_only, baseline_da_anchor_only, baseline_mean_bias,
    HGBResidualModel, RidgeResidualModel,
)


class TestBaselines:
    """Simple baseline tests."""

    def test_sgdfnet_only(self):
        df = pd.DataFrame({"sgdfnet_pred": [100, 200]})
        result = baseline_sgdfnet_only(df)
        assert np.allclose(result, [100, 200])

    def test_da_anchor_only(self):
        df = pd.DataFrame({"da_anchor": [100, 200]})
        result = baseline_da_anchor_only(df)
        assert np.allclose(result, [100, 200])

    def test_mean_bias(self):
        df = pd.DataFrame({"sgdfnet_pred": [100, 200], "rt_actual": [110, 210]})
        result = baseline_mean_bias(df, residual_mean=10.0)
        assert np.allclose(result, [110, 210])


class TestHGB:
    """HGB model tests."""

    def test_fit_predict(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "sgdfnet_pred": np.random.uniform(200, 400, n),
            "rt_actual": np.random.uniform(200, 400, n),
            "hour_sin": np.sin(2 * np.pi * np.arange(n) / 24),
            "hour_cos": np.cos(2 * np.pi * np.arange(n) / 24),
            "dow_sin": np.zeros(n),
            "dow_cos": np.zeros(n),
            "month_sin": np.zeros(n),
            "month_cos": np.zeros(n),
            "is_weekend": np.zeros(n),
            "is_holiday": np.zeros(n),
            "rt_lag_24h": np.random.uniform(200, 400, n),
            "rt_lag_48h": np.random.uniform(200, 400, n),
            "rt_mean_24h": np.random.uniform(200, 400, n),
            "rt_std_24h": np.abs(np.random.randn(n) * 20),
            "delta_lag_24h": np.random.randn(n) * 10,
            "delta_lag_48h": np.random.randn(n) * 10,
            "previous_day_delta_mean_24h": np.random.randn(n) * 10,
            "previous_day_delta_std_24h": np.abs(np.random.randn(n) * 5),
            "sgdfnet_residual_lag_24h": np.random.randn(n) * 10,
            "sgdfnet_residual_mean_7d": np.random.randn(n) * 10,
        })
        model = HGBResidualModel(max_iter=50)
        model.fit(df, target_col="residual_target" if "residual_target" in df.columns else "rt_actual")
        preds = model.predict(df)
        assert len(preds) == n
        assert not np.isnan(preds).any()


class TestRidge:
    """Ridge model tests."""

    def test_fit_predict(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "sgdfnet_pred": np.random.uniform(200, 400, n),
            "rt_actual": np.random.uniform(200, 400, n),
            "hour_sin": np.sin(2 * np.pi * np.arange(n) / 24),
            "hour_cos": np.cos(2 * np.pi * np.arange(n) / 24),
            "dow_sin": np.zeros(n),
            "dow_cos": np.zeros(n),
            "month_sin": np.zeros(n),
            "month_cos": np.zeros(n),
            "is_weekend": np.zeros(n),
            "rt_lag_24h": np.random.uniform(200, 400, n),
            "rt_lag_48h": np.random.uniform(200, 400, n),
            "rt_mean_24h": np.random.uniform(200, 400, n),
            "rt_std_24h": np.abs(np.random.randn(n) * 20),
            "delta_lag_24h": np.random.randn(n) * 10,
            "delta_lag_48h": np.random.randn(n) * 10,
            "previous_day_delta_mean_24h": np.random.randn(n) * 10,
            "previous_day_delta_std_24h": np.abs(np.random.randn(n) * 5),
        })
        model = RidgeResidualModel(alpha=1.0)
        model.fit(df)
        preds = model.predict(df)
        assert len(preds) == n
