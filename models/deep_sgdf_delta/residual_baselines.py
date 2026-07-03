"""Residual baseline models for TrendKnightRT comparison.

Provides simple residual prediction baselines that predict:
    residual = rt_actual - sgdfnet_pred

Baselines:
    1. SGDFNet only (no residual correction)
    2. DA anchor only (no SGDFNet)
    3. SGDFNet + mean residual bias
    4. SGDFNet + hour-wise residual bias
    5. SGDFNet + period-wise residual bias
    6. SGDFNet + HGB (HistGradientBoosting) residual
    7. SGDFNet + Ridge residual
    8. SGDFNet + MLP residual
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def baseline_sgdfnet_only(df: pd.DataFrame) -> np.ndarray:
    """Baseline 1: rt_pred = sgdfnet_pred"""
    return df["sgdfnet_pred"].values


def baseline_da_anchor_only(df: pd.DataFrame) -> np.ndarray:
    """Baseline 2: rt_pred = da_anchor"""
    return df["da_anchor"].values


def baseline_mean_bias(df: pd.DataFrame, residual_mean: float | None = None) -> np.ndarray:
    """Baseline 3: rt_pred = sgdfnet_pred + global mean residual"""
    if residual_mean is None:
        residual_mean = (df["rt_actual"] - df["sgdfnet_pred"]).mean()
    return df["sgdfnet_pred"].values + residual_mean


def baseline_hour_bias(df: pd.DataFrame,
                       hour_bias_map: dict[int, float] | None = None) -> np.ndarray:
    """Baseline 4: rt_pred = sgdfnet_pred + hour-wise residual bias

    Usage (train → test)::

        # Compute bias on train set
        train_resid = train["rt_actual"] - train["sgdfnet_pred"]
        bias_map = train.groupby("hour_business")[train_resid].mean().to_dict()

        # Apply to test set
        preds = baseline_hour_bias(test_df, hour_bias_map=bias_map)
    """
    if hour_bias_map is not None:
        bias = df["hour_business"].map(hour_bias_map).fillna(0).values
    else:
        # Compute bias from this DataFrame (only use when this IS the train set)
        tmp = df.copy()
        tmp["_resid"] = tmp["rt_actual"] - tmp["sgdfnet_pred"]
        bias_map = tmp.groupby("hour_business")["_resid"].mean().to_dict()
        bias = tmp["hour_business"].map(bias_map).fillna(0).values
    return df["sgdfnet_pred"].values + bias


def baseline_period_bias(df: pd.DataFrame,
                         period_bias_map: dict[str, float] | None = None) -> np.ndarray:
    """Baseline 5: rt_pred = sgdfnet_pred + period-wise residual bias

    Usage (train → test)::

        train_resid = train["rt_actual"] - train["sgdfnet_pred"]
        bias_map = train.groupby(train["hour_business"].apply(get_period))[train_resid].mean().to_dict()
        preds = baseline_period_bias(test_df, period_bias_map=bias_map)
    """
    from models.deep_sgdf_delta.realtime_feature_contract import get_period
    if period_bias_map is not None:
        periods = df["hour_business"].apply(get_period)
        bias = periods.map(period_bias_map).fillna(0).values
    else:
        # Compute bias from this DataFrame (only use when this IS the train set)
        tmp = df.copy()
        tmp["_resid"] = tmp["rt_actual"] - tmp["sgdfnet_pred"]
        tmp["_period"] = tmp["hour_business"].apply(get_period)
        bias_map = tmp.groupby("_period")["_resid"].mean().to_dict()
        periods = tmp["_period"]
        bias = periods.map(bias_map).fillna(0).values
    return df["sgdfnet_pred"].values + bias


class HGBResidualModel:
    """Baseline 6: HistGradientBoosting residual model."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingRegressor
        params = dict(
            loss="absolute_error", max_depth=4, learning_rate=0.1,
            max_iter=300, random_state=42,
        )
        params.update(kwargs)
        self.model = HistGradientBoostingRegressor(**params)
        self._trained = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame."""
        feature_cols = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "is_weekend", "is_holiday",
            "rt_lag_24h", "rt_lag_48h", "rt_mean_24h", "rt_std_24h",
            "delta_lag_24h", "delta_lag_48h",
            "previous_day_delta_mean_24h", "previous_day_delta_std_24h",
            "sgdfnet_residual_lag_24h", "sgdfnet_residual_mean_7d",
        ]
        present = [c for c in feature_cols if c in df.columns]
        if not present:
            return np.zeros((len(df), 1))
        return df[present].fillna(0).values

    def fit(self, df: pd.DataFrame, target_col: str = "residual_target") -> None:
        """Fit the model."""
        X = self._extract_features(df)
        y = df[target_col].values if target_col in df.columns else (df["rt_actual"] - df["sgdfnet_pred"]).values
        self.model.fit(X, y)
        self._trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict residuals, return rt_pred = sgdfnet_pred + residual."""
        if not self._trained:
            raise RuntimeError("Model not trained")
        X = self._extract_features(df)
        residual = self.model.predict(X)
        return df["sgdfnet_pred"].values + residual


class RidgeResidualModel:
    """Baseline 7: Ridge regression residual model."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha, random_state=42)
        self._trained = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "is_weekend",
            "rt_lag_24h", "rt_lag_48h", "rt_mean_24h", "rt_std_24h",
            "delta_lag_24h", "delta_lag_48h",
            "previous_day_delta_mean_24h", "previous_day_delta_std_24h",
        ]
        present = [c for c in feature_cols if c in df.columns]
        return df[present].fillna(0).values if present else np.zeros((len(df), 1))

    def fit(self, df: pd.DataFrame) -> None:
        X = self._extract_features(df)
        y = (df["rt_actual"] - df["sgdfnet_pred"]).values
        self.model.fit(X, y)
        self._trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained")
        X = self._extract_features(df)
        residual = self.model.predict(X)
        return df["sgdfnet_pred"].values + residual


class MLPResidualModel:
    """Baseline 8: Small MLP residual model."""

    def __init__(self, hidden=(64, 32)):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden, activation="relu",
            solver="adam", max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )
        self._trained = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "is_weekend",
            "rt_lag_24h", "rt_lag_48h", "rt_mean_24h", "rt_std_24h",
            "delta_lag_24h", "delta_lag_48h",
            "previous_day_delta_mean_24h", "previous_day_delta_std_24h",
            "sgdfnet_residual_lag_24h", "sgdfnet_residual_mean_7d",
        ]
        present = [c for c in feature_cols if c in df.columns]
        return df[present].fillna(0).values if present else np.zeros((len(df), 1))

    def fit(self, df: pd.DataFrame) -> None:
        X = self._extract_features(df)
        y = (df["rt_actual"] - df["sgdfnet_pred"]).values
        self.model.fit(X, y)
        self._trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained")
        X = self._extract_features(df)
        residual = self.model.predict(X)
        return df["sgdfnet_pred"].values + residual
