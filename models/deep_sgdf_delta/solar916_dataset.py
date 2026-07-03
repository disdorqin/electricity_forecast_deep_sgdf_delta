"""Solar916 dataset builder.

Constructs the training/evaluation dataset for the 9_16 residual correction
specialist. Only includes hours 9-16 (business hours 9-16).

Each row contains:
  business_day, hour_business, ds, period, rt_actual, da_anchor,
  sgdfnet_pred, sgdfnet_residual

Plus all features from solar916_features.py.

Usage:
    from models.deep_sgdf_delta.solar916_dataset import build_solar916_dataset

    ds = build_solar916_dataset(
        data_path="path/to/shandong_pmos_hourly.xlsx",
        sgdfnet_predictions_path="path/to/sgdfnet_predictions.csv",
        start_date="2026-01-01",
        end_date="2026-03-31",
    )
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.deep_sgdf_delta.business_time import add_business_time_columns
from models.deep_sgdf_delta.solar916_features import (
    build_solar916_features,
    write_feature_manifest,
)

logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw electricity data with encoding fallback."""
    p = Path(path)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return pd.read_csv(p, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot read {p}")


def load_sgdfnet_predictions(path: str) -> pd.DataFrame:
    """Load SGDFNet predictions CSV."""
    p = Path(path)
    if not p.exists():
        logger.warning("SGDFNet predictions not found at %s", p)
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


def build_solar916_dataset(
    data_path: str,
    sgdfnet_predictions: Optional[pd.DataFrame] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the Solar916 dataset.

    Parameters
    ----------
    data_path : str
        Path to raw data (xlsx/csv).
    sgdfnet_predictions : pd.DataFrame, optional
        Pre-loaded SGDFNet predictions. If None, attempts to load from
        teacher adapter.
    start_date, end_date : str, optional
        Date range filter (on business_day).
    output_dir : str, optional
        If provided, writes dataset.csv and feature_manifest.json here.

    Returns
    -------
    (df, info) where info contains metadata about the dataset.
    """
    # Load raw data
    raw_df = load_raw_data(data_path)
    logger.info("Raw data: %d rows", len(raw_df))

    # Detect timestamp column
    ts_col = None
    for c in ["时刻", "timestamp", "time", "ds"]:
        if c in raw_df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("No timestamp column found in raw data")

    # Apply business time alignment
    df = add_business_time_columns(raw_df, timestamp_col=ts_col)

    # Detect price columns
    da_col = None
    rt_col = None
    for c in ["日前电价", "da_price", "dayahead"]:
        if c in df.columns:
            da_col = c
            break
    for c in ["实时电价", "rt_price", "realtime"]:
        if c in df.columns:
            rt_col = c
            break
    if da_col is None or rt_col is None:
        raise ValueError("Cannot find price columns")

    df = df.rename(columns={da_col: "da_price", rt_col: "rt_price"})
    df["da_price"] = pd.to_numeric(df["da_price"], errors="coerce")
    df["rt_price"] = pd.to_numeric(df["rt_price"], errors="coerce")

    # Date filter
    if start_date:
        df = df[df["business_day"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["business_day"] <= pd.Timestamp(end_date)]

    # Load SGDFNet predictions if not provided
    if sgdfnet_predictions is None:
        try:
            from models.deep_sgdf_delta.teacher_adapters import sgdfnet_teacher
            sgdf_df = sgdfnet_teacher.load_predictions()
        except Exception:
            sgdf_df = None
    else:
        sgdf_df = sgdfnet_predictions

    # Phase 8: Build features on FULL dataset BEFORE filtering to 9_16.
    # This is critical for correct lag/rolling features — they need the
    # full 24-hour context to compute same-hour previous-day lookups.
    logger.info("Building features on full dataset: %d rows", len(df))
    df_full, feat_info = build_solar916_features(df, sgdfnet_predictions=sgdf_df)

    # NOW filter to 9_16 only
    df_916 = df_full[df_full["period"] == "9_16"].copy()
    logger.info("9_16 segment after feature build: %d rows", len(df_916))

    # Rename for clarity
    df_916 = df_916.rename(columns={"rt_price": "rt_actual"})

    # Ensure required columns exist
    required_cols = [
        "business_day", "hour_business", "ds", "period",
        "rt_actual", "da_price", "sgdfnet_pred", "sgdfnet_residual",
    ]
    for col in required_cols:
        if col not in df_916.columns:
            if col == "da_price":
                df_916[col] = df_916.get("da_anchor", 0.0)
            elif col == "sgdfnet_pred":
                df_916[col] = np.nan
            elif col == "sgdfnet_residual":
                df_916[col] = np.nan
            else:
                df_916[col] = np.nan

    # Add da_anchor alias
    if "da_anchor" not in df_916.columns:
        df_916["da_anchor"] = df_916["da_price"]

    # Build info
    info = {
        "n_samples": len(df_916),
        "date_range": [
            str(df_916["business_day"].min().date()) if len(df_916) > 0 else None,
            str(df_916["business_day"].max().date()) if len(df_916) > 0 else None,
        ],
        "missing_features": feat_info.get("missing_features", []),
        "feature_columns": feat_info.get("feature_columns", []),
        "n_sgdfnet_aligned": int(df_916["sgdfnet_pred"].notna().sum()),
    }

    # Write outputs
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        df_916.to_csv(out / "dataset.csv", index=False, encoding="utf-8-sig")
        logger.info("Dataset written to %s (%d rows)", out / "dataset.csv", len(df_916))

        write_feature_manifest(feat_info, out / "feature_manifest.json")

    return df_916, info
