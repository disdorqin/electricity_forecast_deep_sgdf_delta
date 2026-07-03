"""Unified teacher registry for TrendKnight-X.

Manages SGDFNet, RT916, and TimeMixer teachers. Each teacher can be:
  - available: predictions loaded successfully
  - missing_checkpoint: code exists but no predictions found
  - unavailable: teacher not installed or not applicable

Teachers are used for:
  1. Residual distillation (student learns from teacher residuals)
  2. Ensemble blending (weighted combination of teacher + student)
  3. Confidence calibration (disagreement between teachers signals uncertainty)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

TEACHER_NAMES = ["sgdfnet", "rt916", "timemixer"]

@dataclass
class TeacherStatus:
    name: str
    availability: str  # 'available', 'missing_checkpoint', 'unavailable'
    n_predictions: int = 0
    source_path: Optional[str] = None
    error: Optional[str] = None

@dataclass
class TeacherRegistry:
    """Registry that holds all teacher predictions."""
    teachers: dict[str, TeacherStatus] = field(default_factory=dict)
    predictions: dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)
    
    def __post_init__(self):
        for name in TEACHER_NAMES:
            if name not in self.teachers:
                self.teachers[name] = TeacherStatus(name=name, availability="unavailable")
            if name not in self.predictions:
                self.predictions[name] = None
    
    def load_teacher(
        self,
        name: str,
        source_repo_root: Optional[str] = None,
        sgdfnet_root: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        experiment_dir: Optional[str] = None,
    ) -> TeacherStatus:
        """Load a single teacher's predictions."""
        if name == "sgdfnet":
            from models.deep_sgdf_delta.teacher_adapters.sgdfnet_teacher import (
                check_availability, load_predictions,
            )
            status = TeacherStatus(name=name, availability=check_availability(sgdfnet_root))
            if status.availability == "available":
                try:
                    df = load_predictions(source_repo_root, start_date, end_date, experiment_dir)
                    if df is not None and not df.empty:
                        self.predictions[name] = df
                        status.n_predictions = len(df)
                    else:
                        status.availability = "missing_checkpoint"
                except Exception as exc:
                    status.availability = "unavailable"
                    status.error = str(exc)
        
        elif name == "rt916":
            from models.deep_sgdf_delta.teacher_adapters.rt916_teacher import (
                check_availability, load_predictions,
            )
            status = TeacherStatus(name=name, availability=check_availability(source_repo_root))
            if status.availability in ("available", "missing_checkpoint"):
                try:
                    df = load_predictions(source_repo_root, start_date, end_date, experiment_dir)
                    if df is not None and not df.empty:
                        self.predictions[name] = df
                        status.n_predictions = len(df)
                        status.availability = "available"
                    else:
                        status.availability = "missing_checkpoint"
                except Exception as exc:
                    status.availability = "unavailable"
                    status.error = str(exc)
        
        elif name == "timemixer":
            from models.deep_sgdf_delta.teacher_adapters.timemixer_teacher import (
                check_availability, load_predictions,
            )
            status = TeacherStatus(name=name, availability=check_availability(source_repo_root))
            if status.availability in ("available", "missing_checkpoint"):
                try:
                    df = load_predictions(source_repo_root, start_date, end_date, experiment_dir)
                    if df is not None and not df.empty:
                        self.predictions[name] = df
                        status.n_predictions = len(df)
                        status.availability = "available"
                    else:
                        status.availability = "missing_checkpoint"
                except Exception as exc:
                    status.availability = "unavailable"
                    status.error = str(exc)
        else:
            status = TeacherStatus(name=name, availability="unavailable", error=f"Unknown teacher: {name}")
        
        self.teachers[name] = status
        logger.info("Teacher '%s': %s (%d predictions)", name, status.availability, status.n_predictions)
        return status
    
    def load_all(
        self,
        teachers: Optional[list[str]] = None,
        source_repo_root: Optional[str] = None,
        sgdfnet_root: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, TeacherStatus]:
        """Load all requested teachers."""
        if teachers is None:
            teachers = TEACHER_NAMES
        for name in teachers:
            self.load_teacher(name, source_repo_root, sgdfnet_root, start_date, end_date)
        return dict(self.teachers)
    
    def get_merged_predictions(self) -> Optional[pd.DataFrame]:
        """Merge all available teacher predictions into one table.
        
        Returns DataFrame with columns:
          business_day, hour_business, period, teacher_name, teacher_pred,
          teacher_delta_pred, teacher_available, teacher_source
        """
        dfs = []
        for name in TEACHER_NAMES:
            df = self.predictions.get(name)
            if df is not None and not df.empty:
                dfs.append(df)
        
        if not dfs:
            return None
        
        return pd.concat(dfs, ignore_index=True)
    
    def get_wide_predictions(self) -> Optional[pd.DataFrame]:
        """Return teacher predictions in wide format (one row per hour, columns per teacher).
        
        Columns: business_day, hour_business, period,
                 sgdfnet_pred, rt916_pred, timemixer_pred,
                 sgdfnet_available, rt916_available, timemixer_available
        """
        merged = self.get_merged_predictions()
        if merged is None or merged.empty:
            return None
        
        # Pivot to wide format
        wide = merged.pivot_table(
            index=["business_day", "hour_business"],
            columns="teacher_name",
            values=["teacher_pred", "teacher_available"],
            aggfunc="first",
        ).reset_index()
        
        # Flatten column names
        wide.columns = [
            f"{t}_{c}" if c != "" else c
            for c, t in wide.columns
        ]
        # Fix: the pivot creates MultiIndex columns
        wide.columns = ["_".join(col).strip("_") for col in wide.columns]
        
        # Rename to expected format
        rename_map = {}
        for name in TEACHER_NAMES:
            for prefix in [f"teacher_pred_{name}", f"{name}_teacher_pred"]:
                if prefix in wide.columns:
                    rename_map[prefix] = f"{name}_pred"
            for prefix in [f"teacher_available_{name}", f"{name}_teacher_available"]:
                if prefix in wide.columns:
                    rename_map[prefix] = f"{name}_available"
        
        wide = wide.rename(columns=rename_map)
        
        # Add period
        if "period" not in wide.columns and "hour_business" in wide.columns:
            h = wide["hour_business"].astype(int)
            wide["period"] = pd.cut(h, bins=[0, 8, 16, 24], labels=["1_8", "9_16", "17_24"], include_lowest=True).astype(str)
        
        return wide
    
    def summary(self) -> dict:
        """Return a summary dict of teacher statuses."""
        return {
            name: {
                "availability": status.availability,
                "n_predictions": status.n_predictions,
                "error": status.error,
            }
            for name, status in self.teachers.items()
        }
