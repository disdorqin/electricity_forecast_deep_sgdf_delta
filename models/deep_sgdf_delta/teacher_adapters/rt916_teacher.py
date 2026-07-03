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

    # Only model_packages contains actual RT916 predictions;
    # cutoff_recovery_experiments contains SGDFNet files — do NOT count it.
    for sub in ["model_packages", ""]:
        d = base / "outputs" / "RT916_SpikeMarketLab" / sub if sub else base / "outputs" / "rt916"
        if d.is_dir():
            for pat in ("预测结果*.csv", "*fused*.csv", "predictions_rt*.csv"):
                if any(d.rglob(pat)):
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

    # Search model_packages/ for actual RT916 predictions.
    # DO NOT search cutoff_recovery_experiments/ — those are SGDFNet files.
    for pattern_dir in [
        base / "outputs" / "RT916_SpikeMarketLab" / "model_packages",
        base / "outputs" / "rt916",
    ]:
        if pattern_dir.is_dir():
            # RT916 files: 预测结果*.csv (Chinese) or predictions_rt*.csv
            for csv_file in pattern_dir.rglob("预测结果*.csv"):
                candidates.append(csv_file)
            for csv_file in pattern_dir.rglob("predictions_rt*.csv"):
                candidates.append(csv_file)
            for csv_file in pattern_dir.rglob("*fused*.csv"):
                candidates.append(csv_file)

    # Prefer the file with the most rows (best coverage)
    candidates.sort(reverse=True)

    best_result = None
    best_count = 0
    for pred_path in candidates:
        if pred_path.exists() and pred_path.is_file():
            for enc in ("utf-8-sig", "gbk"):
                try:
                    df = pd.read_csv(pred_path, encoding=enc)
                    result = _standardize(df, start_date, end_date)
                    if result is not None and len(result) > best_count:
                        best_result = result
                        best_count = len(result)
                        logger.info("RT916 candidate: %s (%d rows, enc=%s)", pred_path.name, len(result), enc)
                except Exception as exc:
                    logger.warning("Failed to load RT916 predictions from %s (enc=%s): %s", pred_path, enc, exc)

    if best_result is not None and best_count > 0:
        logger.info("Selected RT916 predictions with %d rows", best_count)
        return best_result

    logger.info("RT916 predictions not found — teacher will be marked unavailable")
    return None

def _standardize(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Standardize RT916 output to teacher format."""
    result = df.copy()
    
    # Map prediction column (include Chinese names from RT916 output)
    for c in ("rt_pred", "y_pred", "rt_hat", "forecast", "prediction",
              "预测实时电价"):
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
        for c in ("hour", "target_hour", "hour_business"):
            if c in result.columns:
                result["hour_business"] = result[c]
                break
    
    if "period" not in result.columns and "hour_business" in result.columns:
        from models.deep_sgdf_delta.business_time import compute_period
        result["period"] = result["hour_business"].astype(int).apply(compute_period)
    
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
