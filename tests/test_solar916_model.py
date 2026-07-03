"""Tests for Solar916 model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.solar916_model import (
    Solar916Config,
    build_model,
    get_available_features,
    prepare_training_data,
    smape_floor50,
    train_walk_forward,
)


def _make_dataset(n_days: int = 60) -> pd.DataFrame:
    """Create a mock Solar916 dataset."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for h in range(9, 17):
            bd = d
            rows.append({
                "business_day": bd,
                "hour_business": h,
                "ds": d + pd.Timedelta(hours=h),
                "period": "9_16",
                "rt_actual": 100 + np.random.randn() * 30,
                "da_anchor": 90 + np.random.randn() * 10,
                "sgdfnet_pred": 95 + np.random.randn() * 25,
                "sgdfnet_residual": 5 + np.random.randn() * 20,
                "hour_business": h,
                "weekday": d.weekday(),
                "month": d.month,
                "forecast_solar": max(0, 50 * np.sin(np.pi * (h - 6) / 12)),
                "forecast_wind": 30 + np.random.randn() * 10,
                "forecast_new_energy": 60 + np.random.randn() * 15,
                "bidding_space": 200 + np.random.randn() * 50,
                "forecast_load": 500 + np.random.randn() * 50,
                "net_load": 440 + np.random.randn() * 40,
                "renewable_share": 0.2 + np.random.randn() * 0.05,
                "delta_lag_24": np.random.randn() * 20,
                "delta_lag_168": np.random.randn() * 20,
                "residual_lag_24": np.random.randn() * 15,
                "residual_lag_168": np.random.randn() * 15,
                "rolling_residual_mean_7d": np.random.randn() * 10,
                "rolling_residual_std_7d": abs(np.random.randn() * 5),
                "same_hour_residual_mean_7d": np.random.randn() * 10,
                "same_hour_residual_std_7d": abs(np.random.randn() * 5),
            })
    df = pd.DataFrame(rows)
    df["sgdfnet_residual"] = df["rt_actual"] - df["sgdfnet_pred"]
    return df


class TestSolar916Config:
    def test_default_config(self):
        cfg = Solar916Config()
        assert cfg.model_type == "hist_gradient_boosting"
        assert cfg.max_iter == 500

    def test_custom_config(self):
        cfg = Solar916Config(model_type="catboost", max_iter=100)
        assert cfg.model_type == "catboost"
        assert cfg.max_iter == 100


class TestBuildModel:
    def test_hist_gbr(self):
        cfg = Solar916Config(model_type="hist_gradient_boosting", max_iter=10)
        model = build_model(cfg)
        assert model is not None

    def test_catboost(self):
        cfg = Solar916Config(model_type="catboost", max_iter=10)
        model = build_model(cfg)
        assert model is not None

    def test_lightgbm(self):
        cfg = Solar916Config(model_type="lightgbm", max_iter=10)
        model = build_model(cfg)
        assert model is not None

    def test_unknown_falls_back(self):
        cfg = Solar916Config(model_type="unknown_model")
        with pytest.raises(ValueError):
            build_model(cfg)


class TestGetAvailableFeatures:
    def test_filters_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
        available = get_available_features(df, ["a", "b"])
        assert "a" in available
        assert "b" not in available

    def test_all_available(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        available = get_available_features(df, ["a", "b"])
        assert len(available) == 2


class TestPrepareTrainingData:
    def test_basic(self):
        df = _make_dataset(n_days=30)
        X, y, features = prepare_training_data(df, ["hour_business", "weekday"])
        assert X.shape[0] > 0
        assert X.shape[1] == 2
        assert len(y) == X.shape[0]


class TestTrainWalkForward:
    def test_basic_training(self):
        df = _make_dataset(n_days=120)
        cfg = Solar916Config(max_iter=10)
        result = train_walk_forward(df, "2026-03", cfg)

        assert "model" in result
        assert "test_pred" in result
        assert "test_actual" in result
        assert result["test_rows"] > 0
        assert len(result["test_pred"]) == result["test_rows"]

    def test_feature_importance(self):
        df = _make_dataset(n_days=120)
        cfg = Solar916Config(max_iter=10)
        result = train_walk_forward(df, "2026-03", cfg)

        assert "feature_importance" in result
        assert isinstance(result["feature_importance"], dict)


class TestSMAPEFloor50:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert smape_floor50(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_with_floor(self):
        y_true = np.array([10.0, 20.0])  # Below floor → clipped to 50
        y_pred = np.array([200.0, 300.0])  # Above floor
        result = smape_floor50(y_true, y_pred)
        assert result > 0
