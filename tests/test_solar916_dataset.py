"""Tests for Solar916 dataset and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.deep_sgdf_delta.solar916_features import (
    ALL_CANDIDATE_FEATURES,
    build_solar916_features,
    detect_raw_columns,
)


def _make_raw_df(n_days: int = 14, start_date: str = "2026-02-01") -> pd.DataFrame:
    """Create a minimal raw DataFrame mimicking the Shandong PMOS data."""
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows = []
    for d in dates:
        for h in range(1, 25):
            ts = d + pd.Timedelta(hours=h - 1) if h < 24 else d + pd.Timedelta(days=1)
            rows.append({
                "时刻": ts,
                "日前电价": 100.0 + np.random.randn() * 20,
                "实时电价": 110.0 + np.random.randn() * 30,
                "光伏总加预测值": max(0, 50 * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0,
                "风电总加预测值": 30 + np.random.randn() * 10,
                "新能源总加预测值": 60 + np.random.randn() * 15,
                "竞价空间预测值": 200 + np.random.randn() * 50,
                "直调负荷预测值": 500 + np.random.randn() * 50,
            })
    return pd.DataFrame(rows)


def _make_sgdfnet_predictions(df_916: pd.DataFrame) -> pd.DataFrame:
    """Create mock SGDFNet predictions aligned with 9_16 data."""
    preds = df_916[["business_day", "hour_business"]].copy()
    preds["teacher_pred"] = preds["hour_business"] * 3.0 + 50.0
    return preds


class TestDetectRawColumns:
    def test_detects_chinese_columns(self):
        df = pd.DataFrame({"光伏总加预测值": [1], "风电总加预测值": [2]})
        detected = detect_raw_columns(df)
        assert "forecast_solar" in detected
        assert "forecast_wind" in detected

    def test_empty_for_unknown(self):
        df = pd.DataFrame({"unknown_col": [1]})
        detected = detect_raw_columns(df)
        assert len(detected) == 0


class TestBuildSolar916Features:
    def test_basic_output_columns(self):
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        raw = _make_raw_df(n_days=14)
        df = add_business_time_columns(raw, timestamp_col="时刻")
        df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
        df_916 = df[df["period"] == "9_16"].copy()

        result, info = build_solar916_features(df_916)

        assert "sgdfnet_residual" in result.columns
        assert "hour_business" in result.columns
        assert "weekday" in result.columns
        assert "month" in result.columns
        assert "net_load" in result.columns
        assert info["n_samples"] == len(df_916)

    def test_with_sgdfnet_predictions(self):
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        raw = _make_raw_df(n_days=14)
        df = add_business_time_columns(raw, timestamp_col="时刻")
        df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
        df_916 = df[df["period"] == "9_16"].copy()

        sgdf_preds = _make_sgdfnet_predictions(df_916)
        result, info = build_solar916_features(df_916, sgdfnet_predictions=sgdf_preds)

        assert result["sgdfnet_pred"].notna().sum() > 0
        assert result["sgdfnet_residual"].notna().sum() > 0
        # residual = rt_price - sgdfnet_pred
        valid = result.dropna(subset=["rt_price", "sgdfnet_pred"])
        expected_residual = valid["rt_price"] - valid["sgdfnet_pred"]
        np.testing.assert_allclose(
            valid["sgdfnet_residual"].values,
            expected_residual.values,
            atol=1e-6,
        )

    def test_only_9_16_hours(self):
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        raw = _make_raw_df(n_days=7)
        df = add_business_time_columns(raw, timestamp_col="时刻")
        df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
        df_916 = df[df["period"] == "9_16"].copy()

        result, _ = build_solar916_features(df_916)
        assert set(result["hour_business"].unique()).issubset(set(range(9, 17)))

    def test_missing_features_recorded(self):
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        # Create data without solar/wind columns
        raw = pd.DataFrame({
            "时刻": pd.date_range("2026-02-01", periods=48, freq="h"),
            "日前电价": 100.0,
            "实时电价": 110.0,
        })
        df = add_business_time_columns(raw, timestamp_col="时刻")
        df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
        df_916 = df[df["period"] == "9_16"].copy()

        _, info = build_solar916_features(df_916)
        assert "forecast_solar" in info["missing_features"]
        assert "forecast_wind" in info["missing_features"]

    def test_lag_features(self):
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        raw = _make_raw_df(n_days=14)
        df = add_business_time_columns(raw, timestamp_col="时刻")
        df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
        df_916 = df[df["period"] == "9_16"].copy()

        result, _ = build_solar916_features(df_916)
        assert "delta_lag_24" in result.columns
        assert "delta_lag_168" in result.columns
        assert "residual_lag_24" in result.columns
        assert "residual_lag_168" in result.columns

    def test_rolling_features(self):
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        raw = _make_raw_df(n_days=14)
        df = add_business_time_columns(raw, timestamp_col="时刻")
        df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
        df_916 = df[df["period"] == "9_16"].copy()

        result, _ = build_solar916_features(df_916)
        assert "rolling_residual_mean_7d" in result.columns
        assert "rolling_residual_std_7d" in result.columns
        assert "same_hour_residual_mean_7d" in result.columns
        assert "same_hour_residual_std_7d" in result.columns

    def test_no_future_actuals_leak(self):
        """Ensure no rt_actual-based features except the target itself."""
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        raw = _make_raw_df(n_days=14)
        df = add_business_time_columns(raw, timestamp_col="时刻")
        df = df.rename(columns={"日前电价": "da_price", "实时电价": "rt_price"})
        df_916 = df[df["period"] == "9_16"].copy()

        result, _ = build_solar916_features(df_916)
        # Lag features should be based on past deltas/residuals, not future rt
        # delta = rt - da, so delta_lag uses past rt (which is ok since it's lagged)
        # But we should not have any feature that directly uses current rt_actual
        # except the target column
        feature_cols = [c for c in ALL_CANDIDATE_FEATURES if c in result.columns]
        # None of the feature columns should be rt_actual itself
        assert "rt_actual" not in feature_cols
        assert "rt_price" not in feature_cols


class TestSolar916Dataset:
    def test_build_dataset(self, tmp_path):
        from models.deep_sgdf_delta.solar916_dataset import build_solar916_dataset

        # Create a temp data file
        raw = _make_raw_df(n_days=14)
        data_path = str(tmp_path / "test_data.xlsx")
        raw.to_excel(data_path, index=False)

        df, info = build_solar916_dataset(
            data_path=data_path,
            start_date="2026-02-01",
            end_date="2026-02-14",
            output_dir=str(tmp_path / "output"),
        )

        assert len(df) > 0
        assert "sgdfnet_residual" in df.columns
        assert "rt_actual" in df.columns
        assert "business_day" in df.columns
        assert info["n_samples"] == len(df)
        # Check output files
        assert (tmp_path / "output" / "dataset.csv").exists()
        assert (tmp_path / "output" / "feature_manifest.json").exists()

    def test_only_916_in_dataset(self, tmp_path):
        from models.deep_sgdf_delta.solar916_dataset import build_solar916_dataset

        raw = _make_raw_df(n_days=14)
        data_path = str(tmp_path / "test_data.xlsx")
        raw.to_excel(data_path, index=False)

        df, _ = build_solar916_dataset(
            data_path=data_path,
            start_date="2026-02-01",
            end_date="2026-02-14",
        )

        assert set(df["hour_business"].unique()).issubset(set(range(9, 17)))
        assert (df["period"] == "9_16").all()
