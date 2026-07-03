"""TimeMixer teacher adapter.

TimeMixer provides multiscale decomposition predictions.
Loads from existing output or checkpoint.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

TEACHER_NAME = "timemixer"

def check_availability(source_repo_root: Optional[str] = None) -> str:
    """Check if TimeMixer predictions are available."""
    if source_repo_root:
        base = Path(source_repo_root)
    else:
        base = Path(__file__).resolve().parent.parent.parent.parent / "electricity_forecast_model2.0_exp"
    
    # Check for TimeMixer output
    tm_outputs = base / "outputs" / "timemixer"
    if tm_outputs.is_dir():
        return "available"
    # Also check src directory for TimeMixer code
    tm_src = base / "src" / "timemixer"
    if tm_src.is_dir():
        return "missing_checkpoint"
    return "unavailable"

def load_predictions(
    source_repo_root: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    experiment_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load TimeMixer predictions from existing output."""
    if source_repo_root:
        base = Path(source_repo_root)
    else:
        base = Path(__file__).resolve().parent.parent.parent.parent / "electricity_forecast_model2.0_exp"
    
    candidates = []
    if experiment_dir:
        candidates.append(Path(experiment_dir))
    
    for pattern_dir in [
        base / "outputs" / "timemixer",
        base / "outputs" / "TimeMixer",
    ]:
        if pattern_dir.is_dir():
            for csv_file in pattern_dir.rglob("*.csv"):
                candidates.append(csv_file)
    
    for pred_path in candidates:
        if pred_path.exists() and pred_path.is_file():
            try:
                df = pd.read_csv(pred_path, encoding="utf-8-sig")
                return _standardize(df, start_date, end_date)
            except Exception as exc:
                logger.warning("Failed to load TimeMixer predictions from %s: %s", pred_path, exc)
    
    logger.info("TimeMixer predictions not found — teacher will be marked unavailable")
    return None

def _standardize(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Standardize TimeMixer output to teacher format."""
    result = df.copy()
    
    for c in ("rt_pred", "y_pred", "prediction", "forecast"):
        if c in result.columns:
            result["teacher_pred"] = result[c]
            break
    
    if "teacher_pred" not in result.columns:
        return None
    
    if "da_anchor" not in result.columns:
        for c in ("da_price", "日前电价"):
            if c in result.columns:
                result["da_anchor"] = result[c]
                break
        if "da_anchor" not in result.columns:
            result["da_anchor"] = 0.0
    
    result["teacher_delta_pred"] = result["teacher_pred"] - result["da_anchor"]
    result["teacher_name"] = TEACHER_NAME
    result["teacher_available"] = True
    result["teacher_source"] = "timemixer_output"
    
    # Ensure business_day / hour using unified module
    if "business_day" not in result.columns or "hour_business" not in result.columns:
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        ts_col = None
        for c in ("ds", "timestamp", "时刻"):
            if c in result.columns:
                ts_col = c
                break
        if ts_col:
            result = add_business_time_columns(result, timestamp_col=ts_col)
    
    if "hour_business" not in result.columns:
        result["hour_business"] = 1
    
    if "period" not in result.columns and "hour_business" in result.columns:
        from models.deep_sgdf_delta.business_time import compute_period
        result["period"] = result["hour_business"].astype(int).apply(compute_period)
    
    if start_date and "business_day" in result.columns:
        result["business_day"] = pd.to_datetime(result["business_day"])
        result = result[result["business_day"] >= pd.Timestamp(start_date)]
    if end_date and "business_day" in result.columns:
        result = result[result["business_day"] <= pd.Timestamp(end_date)]
    
    keep = ["business_day", "hour_business", "period", "teacher_name",
            "teacher_pred", "teacher_delta_pred", "teacher_available", "teacher_source", "da_anchor"]
    keep = [c for c in keep if c in result.columns]
    return result[keep].copy()
