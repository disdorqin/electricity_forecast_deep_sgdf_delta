"""Base Prediction Adapter for Ledger-1 Shadow Replay.

Loads and standardizes base predictions from various sources:
1. DA anchor baseline (fallback, marked as non-production)
2. Optional base prediction CSV files
3. SGDFNet/fusion/TimesFM/TimeMixer predictions (if available)

Output standard fields:
- business_day, hour_business, target_month, ds
- base_pred, base_model_name, base_source
- optional y_true (for eval mode)

Uses business_time.py for proper time alignment.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

from models.deep_sgdf_delta.business_time import add_business_time_columns


@dataclass
class BasePredictionLoadResult:
    """Result from loading base predictions."""
    df: pd.DataFrame
    source: str  # "DA_ANCHOR_BASELINE", "BASE_PREDICTION_FILE", etc.
    model_name: str  # "da_anchor", "sgdfnet", "fusion", etc.
    production_candidate: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def load_da_anchor_baseline(
    data_path: str | Path,
    target_months: list[str],
) -> BasePredictionLoadResult:
    """Load DA anchor baseline from Shandong PMOS data.

    DA anchor = day-ahead clearing price (usually the 'price' column).
    This is a fallback option, NOT a production baseline.

    Args:
        data_path: Path to Shandong PMOS hourly CSV.
        target_months: List of target months (e.g., ["2026-01", "2026-02"]).

    Returns:
        BasePredictionLoadResult with DA anchor baseline.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data (try multiple encodings)
    df = None
    for encoding in ["utf-8", "gbk", "gb2312", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise RuntimeError(f"Failed to read {data_path} with any encoding")
    
    # Clean column names (remove whitespace, invisible chars)
    df.columns = df.columns.str.strip()
    
    # Find price column (try common names, including Chinese)
    price_col = None
    for col in ["price", "Price", "clearing_price", "da_price", "日前电价"]:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(
            f"Cannot find price column in {data_path}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Find timestamp column (try common names, including Chinese)
    ts_col = None
    for col in ["ds", "timestamp", "date", "time", "时刻"]:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        raise ValueError(
            f"Cannot find timestamp column in {data_path}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Rename for consistency
    df = df.rename(columns={ts_col: "ds", price_col: "base_pred"})
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Keep actual price as y_true for evaluation
    df["y_true"] = df["base_pred"].copy()
    
    # Add business time columns
    df = add_business_time_columns(df, timestamp_col="ds")
    
    # Filter to target months
    df["year_month"] = df["business_day"].dt.strftime("%Y-%m")
    target_months_set = set(target_months)
    df = df[df["year_month"].isin(target_months_set)].copy()
    
    if len(df) == 0:
        raise ValueError(
            f"No data found for target months {target_months}. "
            f"Available months: {sorted(df['year_month'].unique())}"
        )
    
    # Add target_month column (same as business_day's month)
    df["target_month"] = df["business_day"].dt.strftime("%Y-%m")
    
    # Select standard output columns
    output_cols = [
        "business_day", "hour_business", "target_month", "ds",
        "base_pred", "y_true",
    ]
    
    result_df = df[output_cols].copy()
    result_df["base_model_name"] = "da_anchor"
    result_df["base_source"] = "DA_ANCHOR_BASELINE"
    
    # Check uniqueness
    key_cols = ["business_day", "hour_business", "target_month"]
    if result_df[key_cols].duplicated().any():
        dup_count = result_df[key_cols].duplicated().sum()
        raise ValueError(
            f"Duplicate keys found in DA anchor baseline: {dup_count} duplicates. "
            f"business_day + hour_business + target_month must be unique."
        )
    
    warnings = [
        "This is a DA anchor baseline (fallback), NOT a production baseline.",
        "Marked as production_candidate=false.",
        "Use only for guardrail sensitivity testing, not for production evaluation.",
    ]
    
    return BasePredictionLoadResult(
        df=result_df,
        source="DA_ANCHOR_BASELINE",
        model_name="da_anchor",
        production_candidate=False,
        warnings=warnings,
        metadata={
            "n_samples": len(result_df),
            "target_months": target_months,
            "price_column_used": price_col,
        },
    )


def load_base_prediction_file(
    file_path: str | Path,
    base_model_name: Optional[str] = None,
) -> BasePredictionLoadResult:
    """Load base predictions from a CSV file.

    Supports various input column names:
    - ds / timestamp
    - business_day
    - hour_business
    - target_month
    - da_anchor / base_pred / y_pred / rt_pred
    - sgdfnet_pred / timesfm_pred / timemixer_pred / lightgbm_pred
    - actual / rt_actual / y_true

    Args:
        file_path: Path to base prediction CSV.
        base_model_name: Optional model name override.

    Returns:
        BasePredictionLoadResult with standardized base predictions.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Base prediction file not found: {file_path}")

    df = pd.read_csv(file_path)
    
    # Auto-detect columns
    column_mapping = _detect_base_prediction_columns(df.columns)
    
    # Rename to standard names
    rename_map = {}
    for standard_name, possible_names in column_mapping.items():
        for col in possible_names:
            if col in df.columns:
                rename_map[col] = standard_name
                break
    
    df = df.rename(columns=rename_map)
    
    # Ensure required columns exist
    required_cols = ["base_pred"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Cannot find base prediction column. "
                f"Tried: {column_mapping['base_pred']}. "
                f"Available columns: {list(df.columns)}"
            )
    
    # Add business time columns if not present
    if "business_day" not in df.columns or "hour_business" not in df.columns:
        if "ds" not in df.columns:
            raise ValueError(
                "Cannot add business time columns: 'ds' column not found. "
                "Provide 'ds' or 'business_day' + 'hour_business'."
            )
        df = add_business_time_columns(df, timestamp_col="ds")
    
    # Add target_month if not present
    if "target_month" not in df.columns:
        if "business_day" in df.columns:
            df["target_month"] = pd.to_datetime(df["business_day"]).dt.strftime("%Y-%m")
        else:
            raise ValueError(
                "Cannot infer target_month: provide 'target_month' column or 'business_day'."
            )
    
    # Determine base_model_name
    if base_model_name is None:
        if "base_model_name" in df.columns:
            base_model_name = df["base_model_name"].iloc[0]
        else:
            base_model_name = "unknown"
    
    # Determine base_source
    if "base_source" in df.columns:
        base_source = df["base_source"].iloc[0]
    else:
        base_source = "BASE_PREDICTION_FILE"
    
    # Add base_model_name and base_source columns if not present
    if "base_model_name" not in df.columns:
        df["base_model_name"] = base_model_name
    if "base_source" not in df.columns:
        df["base_source"] = base_source
    
    # Select standard output columns
    output_cols = [
        "business_day", "hour_business", "target_month", "ds",
        "base_pred", "base_model_name", "base_source",
    ]
    
    # Add optional y_true if available
    if "y_true" in df.columns:
        output_cols.append("y_true")
    
    # Ensure all columns exist
    available_cols = df.columns.tolist()
    output_cols = [col for col in output_cols if col in available_cols]
    
    result_df = df[output_cols].copy()
    
    # Check uniqueness
    key_cols = ["business_day", "hour_business", "target_month"]
    if result_df[key_cols].duplicated().any():
        dup_count = result_df[key_cols].duplicated().sum()
        raise ValueError(
            f"Duplicate keys found in base prediction file: {dup_count} duplicates. "
            f"business_day + hour_business + target_month must be unique."
        )
    
    # Determine production_candidate
    production_candidate = base_source != "DA_ANCHOR_BASELINE"
    
    return BasePredictionLoadResult(
        df=result_df,
        source=base_source,
        model_name=base_model_name,
        production_candidate=production_candidate,
        warnings=[],
        metadata={
            "n_samples": len(result_df),
            "file_path": str(file_path),
        },
    )


def _detect_base_prediction_columns(columns: list[str]) -> dict:
    """Detect column names for base prediction data.

    Args:
        columns: List of column names in the DataFrame.

    Returns:
        Dict mapping standard names to lists of possible column names.
    """
    columns_lower = [c.lower() for c in columns]
    
    mapping = {
        "ds": ["ds", "timestamp", "date", "time", "DateTime", "datetime"],
        "business_day": ["business_day", "business_day", "date"],
        "hour_business": ["hour_business", "hour", "hour_of_day"],
        "target_month": ["target_month", "target_month", "month"],
        "base_pred": [
            "base_pred", "da_anchor", "y_pred", "rt_pred",
            "prediction", "forecast", "price_forecast",
        ],
        "y_true": ["y_true", "actual", "rt_actual", "price", "true_price"],
        "base_model_name": ["base_model_name", "model_name", "model"],
        "base_source": ["base_source", "source"],
    }
    
    return mapping


class BasePredictionAdapter:
    """Adapter for loading and standardizing base predictions.

    Usage:
        adapter = BasePredictionAdapter()
        result = adapter.load(
            data_path="data/shandong_pmos_hourly.csv",
            target_months=["2026-01", "2026-02"],
            base_prediction_file=None,  # Use DA anchor fallback
        )
    """

    def __init__(self):
        self.last_result: Optional[BasePredictionLoadResult] = None

    def load(
        self,
        data_path: Optional[str | Path] = None,
        target_months: Optional[list[str]] = None,
        base_prediction_file: Optional[str | Path] = None,
        base_model_name: Optional[str] = None,
    ) -> BasePredictionLoadResult:
        """Load base predictions from specified source.

        Priority:
        1. base_prediction_file (if provided)
        2. DA anchor baseline (if data_path and target_months provided)

        Args:
            data_path: Path to Shandong PMOS data (for DA anchor fallback).
            target_months: List of target months.
            base_prediction_file: Path to base prediction CSV.
            base_model_name: Optional model name override.

        Returns:
            BasePredictionLoadResult.
        """
        if base_prediction_file is not None:
            result = load_base_prediction_file(
                file_path=base_prediction_file,
                base_model_name=base_model_name,
            )
        elif data_path is not None and target_months is not None:
            result = load_da_anchor_baseline(
                data_path=data_path,
                target_months=target_months,
            )
        else:
            raise ValueError(
                "Must provide either base_prediction_file or "
                "(data_path + target_months) for DA anchor fallback."
            )
        
        self.last_result = result
        return result

    def validate(self, result: Optional[BasePredictionLoadResult] = None) -> list[str]:
        """Validate loaded base predictions.

        Args:
            result: BasePredictionLoadResult to validate. If None, uses last_result.

        Returns:
            List of validation errors (empty if valid).
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            return ["No base prediction loaded. Call load() first."]
        
        errors = []
        
        # Check required columns
        required_cols = [
            "business_day", "hour_business", "target_month",
            "base_pred", "base_model_name", "base_source",
        ]
        
        for col in required_cols:
            if col not in result.df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check uniqueness
        key_cols = ["business_day", "hour_business", "target_month"]
        if all(col in result.df.columns for col in key_cols):
            if result.df[key_cols].duplicated().any():
                dup_count = result.df[key_cols].duplicated().sum()
                errors.append(f"Duplicate keys: {dup_count} duplicates in {key_cols}")
        
        # Check for NaN in base_pred
        if "base_pred" in result.df.columns:
            nan_count = result.df["base_pred"].isna().sum()
            if nan_count > 0:
                errors.append(f"NaN in base_pred: {nan_count} rows")
        
        return errors
