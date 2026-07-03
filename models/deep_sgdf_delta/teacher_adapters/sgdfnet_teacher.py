"""SGDFNet teacher adapter.

Loads SGDFNet predictions from existing experiment output or runs the
Protocol B cutoff experiment to generate them.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

TEACHER_NAME = "sgdfnet"

def check_availability(sgdfnet_root: Optional[str] = None) -> str:
    """Return 'available', 'missing_checkpoint', or 'unavailable'."""
    from models.deep_sgdf_delta.sgdfnet_bridge import find_sgdfnet_root
    try:
        root = find_sgdfnet_root(sgdfnet_root)
        if (root / "src" / "sgdfnet").is_dir():
            return "available"
        return "unavailable"
    except Exception:
        return "unavailable"

def load_predictions(
    source_repo_root: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    experiment_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load SGDFNet predictions from existing experiment output.
    
    Returns DataFrame with columns:
      business_day, hour_business, period, ds, teacher_pred, teacher_delta_pred, da_anchor
    
    Or None if not available.
    """
    # Try to find existing predictions
    candidates = []
    if experiment_dir:
        candidates.append(Path(experiment_dir) / "predictions.csv")
    
    if source_repo_root:
        base = Path(source_repo_root)
    else:
        # Try sibling path
        base = Path(__file__).resolve().parent.parent.parent.parent / "electricity_forecast_model2.0_exp"
    
    # Search in common output locations
    exp_base = base / "outputs" / "RT916_SpikeMarketLab" / "cutoff_recovery_experiments"
    if exp_base.is_dir():
        for d in sorted(exp_base.iterdir(), reverse=True):
            if d.is_dir():
                pred_file = d / "predictions.csv"
                if pred_file.exists():
                    candidates.append(pred_file)
    
    for pred_path in candidates:
        if pred_path.exists():
            try:
                df = pd.read_csv(pred_path, encoding="utf-8-sig")
                return _standardize(df, start_date, end_date)
            except Exception as exc:
                logger.warning("Failed to load SGDFNet predictions from %s: %s", pred_path, exc)
    
    logger.info("SGDFNet predictions not found — teacher will be marked unavailable")
    return None

def _standardize(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Standardize SGDFNet output to teacher format."""
    result = df.copy()
    
    # Map column names
    col_map = {}
    for c in ("business_day", "ds"):
        if c in result.columns:
            col_map[c] = c
    if "hour_business" in result.columns:
        col_map["hour_business"] = "hour_business"
    elif "target_hour" in result.columns:
        result["hour_business"] = result["target_hour"]
        col_map["hour_business"] = "hour_business"
    elif "hour" in result.columns:
        result["hour_business"] = result["hour"]
        col_map["hour_business"] = "hour_business"
    
    # Prediction columns
    for c in ("y_pred", "rt_hat", "rt_pred"):
        if c in result.columns:
            result["teacher_pred"] = result[c]
            break
    for c in ("delta_pred", "delta_hat"):
        if c in result.columns:
            result["teacher_delta_pred"] = result[c]
            break
    if "da_anchor" not in result.columns:
        result["da_anchor"] = 0.0
    if "teacher_delta_pred" not in result.columns and "teacher_pred" in result.columns:
        result["teacher_delta_pred"] = result["teacher_pred"] - result["da_anchor"]
    
    result["teacher_name"] = TEACHER_NAME
    result["teacher_available"] = True
    result["teacher_source"] = "sgdfnet_experiment"
    
    # Period
    if "period" not in result.columns and "hour_business" in result.columns:
        h = result["hour_business"].astype(int)
        result["period"] = pd.cut(h, bins=[0, 8, 16, 24], labels=["1_8", "9_16", "17_24"], include_lowest=True).astype(str)
    
    # Date filter
    if start_date and "business_day" in result.columns:
        result["business_day"] = pd.to_datetime(result["business_day"])
        result = result[result["business_day"] >= pd.Timestamp(start_date)]
    if end_date and "business_day" in result.columns:
        result = result[result["business_day"] <= pd.Timestamp(end_date)]
    
    keep = ["business_day", "hour_business", "period", "ds", "teacher_name",
            "teacher_pred", "teacher_delta_pred", "teacher_available", "teacher_source", "da_anchor"]
    keep = [c for c in keep if c in result.columns]
    return result[keep].copy()
