"""RT916 teacher adapter.

RT916 is a spike/sudden-change model. It serves as an offline teacher
for high-volatility hours. Predictions are loaded from existing output.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

TEACHER_NAME = "rt916"

def check_availability(source_repo_root: Optional[str] = None) -> str:
    """Check if RT916 predictions are available."""
    if source_repo_root:
        base = Path(source_repo_root)
    else:
        base = Path(__file__).resolve().parent.parent.parent.parent / "electricity_forecast_model2.0_exp"
    
    # Check for RT916 output directories
    rt_outputs = base / "outputs" / "RT916_SpikeMarketLab"
    if rt_outputs.is_dir():
        return "available"
    return "missing_checkpoint"

def load_predictions(
    source_repo_root: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    experiment_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load RT916 predictions from existing output."""
    if source_repo_root:
        base = Path(source_repo_root)
    else:
        base = Path(__file__).resolve().parent.parent.parent.parent / "electricity_forecast_model2.0_exp"
    
    candidates = []
    if experiment_dir:
        candidates.append(Path(experiment_dir))
    
    # Search common RT916 output locations
    for pattern_dir in [
        base / "outputs" / "RT916_SpikeMarketLab",
        base / "outputs" / "rt916",
    ]:
        if pattern_dir.is_dir():
            for csv_file in pattern_dir.rglob("predictions*.csv"):
                candidates.append(csv_file)
            for csv_file in pattern_dir.rglob("*fused*.csv"):
                candidates.append(csv_file)
    
    for pred_path in candidates:
        if pred_path.exists() and pred_path.is_file():
            try:
                df = pd.read_csv(pred_path, encoding="utf-8-sig")
                return _standardize(df, start_date, end_date)
            except Exception as exc:
                logger.warning("Failed to load RT916 predictions from %s: %s", pred_path, exc)
    
    logger.info("RT916 predictions not found — teacher will be marked unavailable")
    return None

def _standardize(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Standardize RT916 output to teacher format."""
    result = df.copy()
    
    # Map prediction column
    for c in ("rt_pred", "y_pred", "rt_hat", "forecast", "prediction"):
        if c in result.columns:
            result["teacher_pred"] = result[c]
            break
    
    if "teacher_pred" not in result.columns:
        return None
    
    if "da_anchor" not in result.columns:
        for c in ("da_price", "日前电价", "dayahead"):
            if c in result.columns:
                result["da_anchor"] = result[c]
                break
        if "da_anchor" not in result.columns:
            result["da_anchor"] = 0.0
    
    result["teacher_delta_pred"] = result["teacher_pred"] - result["da_anchor"]
    result["teacher_name"] = TEACHER_NAME
    result["teacher_available"] = True
    result["teacher_source"] = "rt916_output"
    
    # Ensure business_day / hour
    if "business_day" not in result.columns:
        for c in ("ds", "timestamp", "时刻"):
            if c in result.columns:
                ts = pd.to_datetime(result[c])
                result["business_day"] = ts.dt.normalize()
                h = ts.dt.hour
                mask = h == 0
                result["hour_business"] = h
                result.loc[mask, "hour_business"] = 24
                result.loc[mask, "business_day"] = result.loc[mask, "business_day"] - pd.Timedelta(days=1)
                break
    
    if "hour_business" not in result.columns:
        for c in ("hour", "target_hour", "hour_business"):
            if c in result.columns:
                result["hour_business"] = result[c]
                break
    
    h = result["hour_business"].astype(int)
    result["period"] = pd.cut(h, bins=[0, 8, 16, 24], labels=["1_8", "9_16", "17_24"], include_lowest=True).astype(str)
    
    # Date filter
    if start_date and "business_day" in result.columns:
        result["business_day"] = pd.to_datetime(result["business_day"])
        result = result[result["business_day"] >= pd.Timestamp(start_date)]
    if end_date and "business_day" in result.columns:
        result = result[result["business_day"] <= pd.Timestamp(end_date)]
    
    keep = ["business_day", "hour_business", "period", "teacher_name",
            "teacher_pred", "teacher_delta_pred", "teacher_available", "teacher_source", "da_anchor"]
    keep = [c for c in keep if c in result.columns]
    return result[keep].copy()
