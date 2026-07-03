"""Tests for realtime_column_mapping.py.

Covers:
- Chinese column detection and renaming.
- Edge case: duplicate mappings (e.g. 日前电价 and 日前价格 both → da_anchor).
- Edge case: partial Chinese column presence.
- Unmapped Chinese column reporting.
"""
from __future__ import annotations

import pandas as pd
import pytest

from models.deep_sgdf_delta.realtime_column_mapping import (
    CN_CORE,
    CN_FORECAST_MAP,
    EN_TO_CN,
    _build_rename_map,
    audit_chinese_column_mapping,
    rename_chinese_columns,
)


class TestRenameChineseColumns:
    """Core renaming functionality."""

    def test_rename_core_columns(self):
        """Basic core columns are renamed correctly."""
        df = pd.DataFrame({
            "时刻": pd.date_range("2025-06-01", periods=24, freq="h"),
            "日前电价": [300.0] * 24,
            "实时电价": [310.0] * 24,
        })
        result = rename_chinese_columns(df)
        assert "ds" in result.columns
        assert "da_anchor" in result.columns
        assert "rt_actual" in result.columns
        assert "时刻" not in result.columns
        assert result["da_anchor"].iloc[0] == 300.0

    def test_duplicate_da_anchor_mapping(self):
        """When both 日前电价 and 日前价格 exist, one maps to forecast_price."""
        df = pd.DataFrame({
            "时刻": pd.date_range("2025-06-01", periods=24, freq="h"),
            "日前电价": [300.0] * 24,
            "日前价格": [305.0] * 24,
            "实时电价": [310.0] * 24,
        })
        result = rename_chinese_columns(df)
        assert "da_anchor" in result.columns
        assert "forecast_price" in result.columns
        # Both should have different values (from different source cols)
        assert result["da_anchor"].iloc[0] == 300.0
        assert result["forecast_price"].iloc[0] == 305.0

    def test_forecast_columns_renamed(self):
        """Forecast Chinese columns are renamed."""
        df = pd.DataFrame({
            "时刻": pd.date_range("2025-06-01", periods=24, freq="h"),
            "日前电价": [300.0] * 24,
            "实时电价": [310.0] * 24,
            "风电总加预测值": [500.0] * 24,
            "光伏总加预测值": [300.0] * 24,
            "新能源总加预测值": [800.0] * 24,
        })
        result = rename_chinese_columns(df)
        assert "wind_forecast" in result.columns
        assert "solar_forecast" in result.columns
        assert "renewable_forecast" in result.columns
        assert result["wind_forecast"].iloc[0] == 500.0

    def test_partial_chinese_columns(self):
        """Only known Chinese columns are renamed; others left unchanged."""
        df = pd.DataFrame({
            "时刻": pd.date_range("2025-06-01", periods=24, freq="h"),
            "日前电价": [300.0] * 24,
            "实时电价": [310.0] * 24,
            "未知中文列": [1.0] * 24,
            "english_col": [2.0] * 24,
        })
        result = rename_chinese_columns(df)
        assert "ds" in result.columns
        assert "unknown_col" not in result.columns  # unmapped stays
        assert "未知中文列" in result.columns  # unmapped
        assert "english_col" in result.columns

    def test_already_english(self):
        """DataFrame with English names is unchanged."""
        df = pd.DataFrame({
            "ds": pd.date_range("2025-06-01", periods=24, freq="h"),
            "da_anchor": [300.0] * 24,
            "rt_actual": [310.0] * 24,
        })
        result = rename_chinese_columns(df)
        assert list(result.columns) == ["ds", "da_anchor", "rt_actual"]


class TestBuildRenameMap:
    """Internal rename map builder."""

    def test_no_duplicate_en_names(self):
        """No duplicate English names in the rename map."""
        df = pd.DataFrame({
            "时刻": [1],
            "日前电价": [2],
            "日前价格": [3],
            "实时价格": [4],
            "实时电价": [5],
        })
        rename_map = _build_rename_map(df)
        en_values = list(rename_map.values())
        # da_anchor should appear only once
        assert en_values.count("da_anchor") == 1, (
            f"da_anchor appears {en_values.count('da_anchor')} times"
        )


class TestAuditChineseColumnMapping:
    """Audit functionality."""

    def test_audit_finds_mapped_and_unmapped(self):
        """Audit reports mapped, unmapped, and missing English columns."""
        df = pd.DataFrame({
            "时刻": [1],
            "日前电价": [2],
            "实时电价": [3],
            "未知列A": [4],
        })
        audit = audit_chinese_column_mapping(df)
        assert audit["n_mapped"] >= 3  # 时刻, 日前电价, 实时电价
        assert "未知列A" in audit["unmapped_cn_columns"]
        assert audit["n_unmapped"] == 1

    def test_audit_reports_missing_english(self):
        """Audit lists English names that have CN variants but were not found."""
        df = pd.DataFrame({
            "时刻": [1],
            "日前电价": [2],
        })
        audit = audit_chinese_column_mapping(df)
        # Should list some forecast columns as "known_english_not_found"
        assert len(audit["known_english_not_found"]) > 0
