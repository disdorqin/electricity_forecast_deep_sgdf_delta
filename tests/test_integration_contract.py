"""Tests for the integration contract module."""
from __future__ import annotations

import pandas as pd
import pytest


def test_online_pack_columns_defined():
    from models.deep_sgdf_delta.integration_contract import ONLINE_PACK_COLUMNS
    assert len(ONLINE_PACK_COLUMNS) == 14
    assert "trend_pred" in ONLINE_PACK_COLUMNS
    assert "y_true" not in ONLINE_PACK_COLUMNS


def test_eval_extra_columns():
    from models.deep_sgdf_delta.integration_contract import EVAL_EXTRA_COLUMNS
    assert "y_true" in EVAL_EXTRA_COLUMNS
    assert "residual_for_spike_module" in EVAL_EXTRA_COLUMNS
    assert "residual_for_negative_module" in EVAL_EXTRA_COLUMNS


def test_hour_to_period():
    from models.deep_sgdf_delta.integration_contract import hour_to_period
    assert hour_to_period(1) == "1_8"
    assert hour_to_period(8) == "1_8"
    assert hour_to_period(9) == "9_16"
    assert hour_to_period(16) == "9_16"
    assert hour_to_period(17) == "17_24"
    assert hour_to_period(24) == "17_24"
    assert hour_to_period(0) == "unknown"
    assert hour_to_period(25) == "unknown"


def test_validate_online_pack_valid():
    from models.deep_sgdf_delta.integration_contract import (
        validate_online_pack, ONLINE_PACK_COLUMNS,
    )
    df = pd.DataFrame({
        "business_day": pd.to_datetime(["2026-03-01", "2026-03-02"]),
        "hour_business": [1, 9],
        "period": ["1_8", "9_16"],
        "ds": pd.to_datetime(["2026-03-01 01:00", "2026-03-02 09:00"]),
        "trend_pred": [100.0, 200.0],
        "trend_model_name": ["v2_day_tcn", "v2_day_tcn"],
        "trend_confidence": [0.9, 0.8],
        "deep_rt_pred": [100.0, 200.0],
        "sgdfnet_pred": [105.0, 195.0],
        "blend_pred": [102.0, 198.0],
        "da_anchor": [90.0, 180.0],
        "normal_trend_flag": [1, 1],
        "high_price_bucket_flag": [0, 0],
        "negative_bucket_flag": [0, 0],
    })
    is_valid, errors = validate_online_pack(df)
    assert is_valid, f"Expected valid, got errors: {errors}"


def test_validate_online_pack_missing_columns():
    from models.deep_sgdf_delta.integration_contract import validate_online_pack
    df = pd.DataFrame({"business_day": [1], "hour_business": [1]})
    is_valid, errors = validate_online_pack(df)
    assert not is_valid
    assert any("Missing" in e for e in errors)


def test_validate_online_pack_no_y_true():
    from models.deep_sgdf_delta.integration_contract import validate_online_pack
    df = pd.DataFrame({
        "business_day": pd.to_datetime(["2026-03-01"]),
        "hour_business": [1],
        "period": ["1_8"],
        "ds": pd.to_datetime(["2026-03-01 01:00"]),
        "trend_pred": [100.0],
        "trend_model_name": ["test"],
        "trend_confidence": [1.0],
        "deep_rt_pred": [100.0],
        "sgdfnet_pred": [100.0],
        "blend_pred": [100.0],
        "da_anchor": [90.0],
        "normal_trend_flag": [1],
        "high_price_bucket_flag": [0],
        "negative_bucket_flag": [0],
        "y_true": [105.0],  # Should NOT be in online pack
    })
    is_valid, errors = validate_online_pack(df)
    assert not is_valid
    assert any("eval columns" in e for e in errors)


def test_strip_eval_columns():
    from models.deep_sgdf_delta.integration_contract import strip_eval_columns
    df = pd.DataFrame({
        "trend_pred": [100.0],
        "y_true": [105.0],
        "residual_for_spike_module": [5.0],
        "residual_for_negative_module": [5.0],
        "da_anchor": [90.0],
    })
    stripped = strip_eval_columns(df)
    assert "y_true" not in stripped.columns
    assert "residual_for_spike_module" not in stripped.columns
    assert "trend_pred" in stripped.columns
    assert "da_anchor" in stripped.columns


def test_add_eval_columns():
    from models.deep_sgdf_delta.integration_contract import add_eval_columns
    df = pd.DataFrame({
        "trend_pred": [100.0, 200.0],
        "y_true": [105.0, 210.0],
    })
    result = add_eval_columns(df)
    assert "residual_for_spike_module" in result.columns
    assert "residual_for_negative_module" in result.columns
    assert result["residual_for_spike_module"].iloc[0] == pytest.approx(5.0)
    assert result["residual_for_negative_module"].iloc[1] == pytest.approx(10.0)


def test_add_eval_columns_requires_y_true():
    from models.deep_sgdf_delta.integration_contract import add_eval_columns
    df = pd.DataFrame({"trend_pred": [100.0]})
    with pytest.raises(ValueError, match="y_true"):
        add_eval_columns(df)


def test_build_online_pack_row():
    from models.deep_sgdf_delta.integration_contract import build_online_pack_row
    row = build_online_pack_row(
        business_day="2026-03-01",
        hour_business=9,
        ds="2026-03-01 09:00",
        trend_pred=200.0,
        trend_model_name="v2_day_tcn",
        trend_confidence=0.95,
        deep_rt_pred=200.0,
        sgdfnet_pred=195.0,
        blend_pred=198.0,
        da_anchor=180.0,
    )
    assert row["period"] == "9_16"
    assert row["trend_pred"] == 200.0
    assert row["normal_trend_flag"] == 1
    assert row["high_price_bucket_flag"] == 0
    assert row["negative_bucket_flag"] == 0
