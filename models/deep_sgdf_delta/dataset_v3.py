"""DaySequenceDatasetV3 for TrendKnight-X v3.

Extends the V2 day-level dataset with optional teacher prediction features.
Each sample is one full business_day with 24 hours.

Produces (in addition to V2 outputs):
  - teacher_pred_24:   [num_teachers, 24]  per-teacher delta predictions
  - teacher_mask_24:   [num_teachers]      1 where teacher is available

Supports train / val / predict modes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset_v2 import (
    DaySequenceDataset,
    collate_fn_v2,
    _resolve_feature_columns,
    _get_feature_config,
)

# ── SGDFNet integration via bridge (lazy) ────────────────────────────
from models.deep_sgdf_delta import sgdfnet_bridge as _bridge


def _resolve_feature_columns(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Keep only feature columns that actually exist in the frame."""
    return [c for c in feature_cols if c in frame.columns]


# Lazy-evaluated default config (access triggers SGDFNet resolution)
class _LazyDefaultConfig:
    _instance = None
    def __getattr__(self, name):
        if _LazyDefaultConfig._instance is None:
            _LazyDefaultConfig._instance = _get_feature_config()
        return getattr(_LazyDefaultConfig._instance, name)
    def __repr__(self):
        return repr(_get_feature_config())

DEFAULT_FEATURE_CONFIG = _LazyDefaultConfig()


logger = logging.getLogger(__name__)


# ── Teacher alignment helper ─────────────────────────────────────────

def _align_teacher_predictions(
    teacher_df: pd.DataFrame,
    business_days: list[pd.Timestamp],
    num_teachers: int,
    teacher_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align teacher predictions to [num_days, num_teachers, 24] arrays.

    Args:
        teacher_df: DataFrame with columns:
            business_day, hour (1-24), and one column per teacher named
            ``{teacher_name}_delta_pred`` or ``teacher_pred_{teacher_name}``.
        business_days: Ordered list of business days to align to.
        num_teachers: Number of teacher slots.
        teacher_names: Names of teachers (used to find columns). If None,
            auto-detected from DataFrame columns.

    Returns:
        teacher_pred: [num_days, num_teachers, 24] float32 array
        teacher_mask: [num_days, num_teachers] float32 array (1 = available)
    """
    num_days = len(business_days)
    teacher_pred = np.zeros((num_days, num_teachers, 24), dtype=np.float32)
    teacher_mask = np.zeros((num_days, num_teachers), dtype=np.float32)

    if teacher_df is None or teacher_df.empty:
        return teacher_pred, teacher_mask

    # Auto-detect teacher names from columns
    if teacher_names is None:
        teacher_names = []
        for col in teacher_df.columns:
            if col.endswith("_delta_pred"):
                name = col.replace("_delta_pred", "")
                teacher_names.append(name)
            elif col.startswith("teacher_pred_"):
                name = col.replace("teacher_pred_", "")
                teacher_names.append(name)
        teacher_names = teacher_names[:num_teachers]

    if not teacher_names:
        return teacher_pred, teacher_mask

    # Build day->index mapping
    day_to_idx = {d: i for i, d in enumerate(business_days)}

    # Ensure business_day is datetime
    tdf = teacher_df.copy()
    if "business_day" in tdf.columns:
        tdf["business_day"] = pd.to_datetime(tdf["business_day"]).dt.normalize()

    # Determine hour column
    hour_col = None
    for candidate in ["hour", "target_hour", "hour_business"]:
        if candidate in tdf.columns:
            hour_col = candidate
            break
    if hour_col is None:
        logger.warning("No hour column found in teacher DataFrame")
        return teacher_pred, teacher_mask

    tdf[hour_col] = tdf[hour_col].astype(int)

    for t_idx, t_name in enumerate(teacher_names):
        if t_idx >= num_teachers:
            break

        # Find the prediction column
        pred_col = None
        for candidate in [f"{t_name}_delta_pred", f"teacher_pred_{t_name}",
                          f"{t_name}_pred", "delta_pred"]:
            if candidate in tdf.columns:
                pred_col = candidate
                break
        if pred_col is None:
            continue

        for _, row in tdf.iterrows():
            bd = row.get("business_day")
            h = int(row[hour_col])
            if bd not in day_to_idx or h < 1 or h > 24:
                continue
            day_idx = day_to_idx[bd]
            hour_idx = h - 1
            val = row[pred_col]
            if pd.notna(val):
                teacher_pred[day_idx, t_idx, hour_idx] = float(val)
                teacher_mask[day_idx, t_idx] = 1.0

    return teacher_pred, teacher_mask


# ── V3 Dataset ───────────────────────────────────────────────────────

class DaySequenceDatasetV3(Dataset):
    """PyTorch Dataset for day-level 24-hour prediction with teacher features.

    Each sample is one business_day with 24 hourly rows.
    Extends V2 dataset with teacher prediction features.

    Args:
        frame: DataFrame with columns from SGDFNet preprocess_dataframe
        feature_cols: list of feature column names to use
        mode: "train", "val", or "predict"
        spike_threshold: threshold for high_price_mask
        teacher_pred_24: [num_days, num_teachers, 24] teacher predictions
        teacher_mask_24: [num_days, num_teachers] teacher availability mask
        num_teachers: number of teacher slots
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        *,
        mode: Literal["train", "val", "predict"] = "train",
        spike_threshold: float = 500.0,
        teacher_pred_24: np.ndarray | None = None,
        teacher_mask_24: np.ndarray | None = None,
        num_teachers: int = 3,
    ):
        self.feature_cols = _resolve_feature_columns(frame, feature_cols)
        self.mode = mode
        self.spike_threshold = spike_threshold
        self.input_dim = len(self.feature_cols)
        self.num_teachers = num_teachers

        # Validate required columns
        required = ["business_day", "target_hour", "segment_id", "da_anchor"]
        for col in required:
            if col not in frame.columns:
                raise ValueError(f"Missing required column: {col}")

        # Prepare frame
        frame = frame.copy()
        frame["business_day"] = pd.to_datetime(frame["business_day"]).dt.normalize()
        frame["target_hour"] = frame["target_hour"].astype(int)

        # Group by business_day
        grouped = frame.groupby("business_day")

        # Build per-day data structures
        self._days: list[pd.Timestamp] = []
        self._day_features: list[np.ndarray] = []       # each [24, input_dim]
        self._day_delta: list[np.ndarray] = []           # each [24]
        self._day_da: list[np.ndarray] = []              # each [24]
        self._day_rt: list[np.ndarray] = []              # each [24]
        self._day_seg: list[np.ndarray] = []             # each [24]
        self._day_valid: list[np.ndarray] = []           # each [24]

        # Teacher data
        self._teacher_pred: list[np.ndarray] = []        # each [num_teachers, 24]
        self._teacher_mask: list[np.ndarray] = []        # each [num_teachers]

        day_idx = 0
        for bd, group in grouped:
            group = group.set_index("target_hour")

            # Skip days with no usable data in train/val mode
            if mode != "predict":
                if "delta_target" not in group.columns:
                    continue
                if group["delta_target"].isna().all():
                    continue

            # Build 24-hour arrays (hours 1-24, index 0 = hour 1)
            feat_arr = np.zeros((24, self.input_dim), dtype=np.float32)
            delta_arr = np.zeros(24, dtype=np.float32)
            da_arr = np.zeros(24, dtype=np.float32)
            rt_arr = np.zeros(24, dtype=np.float32)
            seg_arr = np.zeros(24, dtype=np.int64)
            valid_arr = np.zeros(24, dtype=np.float32)

            for hour in range(1, 25):
                idx = hour - 1
                if hour in group.index:
                    row = group.loc[hour]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    feat_arr[idx] = row[self.feature_cols].values.astype(np.float32)
                    da_arr[idx] = float(row.get("da_anchor", 0.0))
                    seg_arr[idx] = int(row.get("segment_id", 0))
                    valid_arr[idx] = 1.0

                    if mode != "predict":
                        dt_val = row.get("delta_target")
                        if pd.notna(dt_val):
                            delta_arr[idx] = float(dt_val)
                        rt_val = row.get("rt_actual")
                        if pd.notna(rt_val):
                            rt_arr[idx] = float(rt_val)
                        else:
                            rt_arr[idx] = da_arr[idx] + delta_arr[idx]
                    else:
                        rt_arr[idx] = da_arr[idx]

            self._days.append(bd)
            self._day_features.append(feat_arr)
            self._day_delta.append(delta_arr)
            self._day_da.append(da_arr)
            self._day_rt.append(rt_arr)
            self._day_seg.append(seg_arr)
            self._day_valid.append(valid_arr)

            # Teacher data
            if teacher_pred_24 is not None and day_idx < teacher_pred_24.shape[0]:
                self._teacher_pred.append(teacher_pred_24[day_idx])
                self._teacher_mask.append(teacher_mask_24[day_idx])
            else:
                self._teacher_pred.append(
                    np.zeros((num_teachers, 24), dtype=np.float32)
                )
                self._teacher_mask.append(
                    np.zeros(num_teachers, dtype=np.float32)
                )
            day_idx += 1

    def __len__(self) -> int:
        return len(self._days)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feat = self._day_features[idx]          # [24, input_dim]
        delta = self._day_delta[idx]            # [24]
        da = self._day_da[idx]                  # [24]
        rt = self._day_rt[idx]                  # [24]
        seg = self._day_seg[idx]                # [24]
        valid = self._day_valid[idx]            # [24]

        # Majority segment for the day
        valid_bool = valid.astype(bool)
        if valid_bool.sum() > 0:
            seg_counts = np.bincount(seg[valid_bool], minlength=3)
        else:
            seg_counts = np.bincount(seg, minlength=3)
        majority_seg = int(np.argmax(seg_counts))

        # Compute masks
        rt_tensor = torch.from_numpy(rt)
        high_price_mask = (rt_tensor.abs() > self.spike_threshold).float()
        negative_mask = (rt_tensor < 0.0).float()
        normal_mask = ((1.0 - high_price_mask) * (1.0 - negative_mask)
                       * torch.from_numpy(valid)).float()
        seg_916_mask = (torch.from_numpy(seg) == 1).float() * torch.from_numpy(valid)

        # Teacher data
        t_pred = self._teacher_pred[idx]        # [num_teachers, 24]
        t_mask = self._teacher_mask[idx]        # [num_teachers]

        result = {
            "features_24h": torch.from_numpy(feat),            # [24, input_dim]
            "delta_target_24": torch.from_numpy(delta),        # [24]
            "da_anchor_24": torch.from_numpy(da),              # [24]
            "rt_actual_24": rt_tensor,                         # [24]
            "segment_ids_24": torch.from_numpy(seg),           # [24]
            "segment_id": torch.tensor(majority_seg, dtype=torch.long),
            "valid_mask": torch.from_numpy(valid),             # [24]
            "normal_mask": normal_mask,                        # [24]
            "high_price_mask": high_price_mask,                # [24]
            "negative_mask": negative_mask,                    # [24]
            "segment_916_mask": seg_916_mask,                  # [24]
            "teacher_pred_24": torch.from_numpy(t_pred),       # [num_teachers, 24]
            "teacher_mask_24": torch.from_numpy(t_mask),       # [num_teachers]
            "business_day": torch.tensor(
                self._days[idx].timestamp(), dtype=torch.float64
            ),
        }
        return result

    @property
    def business_days(self) -> list[pd.Timestamp]:
        """Return the list of business_days in order."""
        return list(self._days)


# ── Collate function ─────────────────────────────────────────────────

def collate_fn_v3(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader — stacks day-level V3 samples."""
    base = collate_fn_v2(batch)
    base["teacher_pred_24"] = torch.stack([b["teacher_pred_24"] for b in batch])
    base["teacher_mask_24"] = torch.stack([b["teacher_mask_24"] for b in batch])
    return base


# ── Builder functions ────────────────────────────────────────────────

def build_training_datasets_v3(
    raw_df: pd.DataFrame,
    feature_config=None,
    *,
    decision_day: pd.Timestamp,
    val_days: int = 30,
    train_min_days: int = 90,
    teacher_pred_df: pd.DataFrame | None = None,
    num_teachers: int = 3,
    teacher_names: list[str] | None = None,
    rt916_scope_config=None,
) -> tuple[DaySequenceDatasetV3, DaySequenceDatasetV3, list[str]]:
    """Build train and val DaySequenceDatasetV3 for a single decision day.

    Args:
        raw_df: Raw data DataFrame
        feature_config: SGDFNet FeatureConfig (uses default if None)
        decision_day: Walk-forward decision day
        val_days: Number of days for validation
        train_min_days: Minimum training days required
        teacher_pred_df: Optional teacher predictions DataFrame
        num_teachers: Number of teacher slots
        teacher_names: Names of teachers (e.g., ["sgdfnet", "rt916", "timemixer"])
        rt916_scope_config: Optional RT916ScopeConfig for local teacher restriction

    Returns:
        (train_ds, val_ds, feature_cols)
    """
    if feature_config is None:
        feature_config = _get_feature_config()

    frame, feature_cols = _bridge.preprocess_dataframe(raw_df, feature_config)

    # Split by business_day
    val_start = decision_day - pd.Timedelta(days=val_days)
    train_frame = frame[frame["business_day"] < val_start].copy()
    val_frame = frame[
        (frame["business_day"] >= val_start) & (frame["business_day"] < decision_day)
    ].copy()

    if train_frame["business_day"].nunique() < train_min_days:
        raise ValueError(
            f"Insufficient training days: "
            f"{train_frame['business_day'].nunique()} < {train_min_days}"
        )

    # Fill NaN in features with 0
    for col in feature_cols:
        if col in train_frame.columns:
            train_frame[col] = train_frame[col].fillna(0.0)
            val_frame[col] = val_frame[col].fillna(0.0)

    # Build teacher prediction arrays
    train_teacher_pred, train_teacher_mask = None, None
    val_teacher_pred, val_teacher_mask = None, None

    if teacher_pred_df is not None and not teacher_pred_df.empty:
        train_days = sorted(train_frame["business_day"].unique())
        val_days_list = sorted(val_frame["business_day"].unique())

        # Convert to Timestamp if needed
        train_days = [pd.Timestamp(d) for d in train_days]
        val_days_list = [pd.Timestamp(d) for d in val_days_list]

        # Build full teacher arrays for all days, then slice
        all_days = sorted(set(train_days) | set(val_days_list))
        all_tp, all_tm = _align_teacher_predictions(
            teacher_pred_df, all_days, num_teachers,
        )

        # Apply RT916 scope restriction (Phase 5 Task C)
        if rt916_scope_config is not None and teacher_names is not None:
            from .rt916_scope import apply_rt916_scope

            rt916_idx = teacher_names.index("rt916") if "rt916" in teacher_names else -1
            if rt916_idx >= 0:
                # Build context arrays from frame for scope computation
                all_day_set = set(all_days)
                context_rows = []
                for bd in all_days:
                    day_frame = frame[frame["business_day"] == bd]
                    day_rt = np.zeros(24, dtype=np.float32)
                    day_seg = np.zeros(24, dtype=np.int64)
                    day_delta = np.zeros(24, dtype=np.float32)
                    day_da = np.zeros(24, dtype=np.float32)
                    for _, r in day_frame.iterrows():
                        h = int(r.get("target_hour", 0)) - 1
                        if 0 <= h < 24:
                            day_rt[h] = float(r.get("rt_actual", 0.0))
                            day_seg[h] = int(r.get("segment_id", 0))
                            day_delta[h] = float(r.get("delta_target", 0.0))
                            day_da[h] = float(r.get("da_anchor", 0.0))
                    context_rows.append((day_rt, day_seg, day_delta, day_da))

                if context_rows:
                    ctx_rt = np.stack([r[0] for r in context_rows])
                    ctx_seg = np.stack([r[1] for r in context_rows])
                    ctx_delta = np.stack([r[2] for r in context_rows])
                    ctx_da = np.stack([r[3] for r in context_rows])

                    # SGDFNet predictions (teacher index 0)
                    sgd_pred = all_tp[:, 0, :] if num_teachers > 0 else None

                    all_tp, all_tm, scope_stats = apply_rt916_scope(
                        all_tp, all_tm, teacher_names, rt916_idx,
                        rt_actual=ctx_rt, segment_ids=ctx_seg,
                        delta_true=ctx_delta,
                        sgdfnet_pred=sgd_pred, da_anchor=ctx_da,
                        config=rt916_scope_config,
                    )
                    logger.info("RT916 scope stats: %s", scope_stats)

        day_to_all_idx = {d: i for i, d in enumerate(all_days)}

        # Train slice
        train_indices = [day_to_all_idx[d] for d in train_days if d in day_to_all_idx]
        if train_indices:
            train_teacher_pred = all_tp[train_indices]
            train_teacher_mask = all_tm[train_indices]

        # Val slice
        val_indices = [day_to_all_idx[d] for d in val_days_list if d in day_to_all_idx]
        if val_indices:
            val_teacher_pred = all_tp[val_indices]
            val_teacher_mask = all_tm[val_indices]

    train_ds = DaySequenceDatasetV3(
        train_frame, feature_cols, mode="train",
        teacher_pred_24=train_teacher_pred,
        teacher_mask_24=train_teacher_mask,
        num_teachers=num_teachers,
    )
    val_ds = DaySequenceDatasetV3(
        val_frame, feature_cols, mode="val",
        teacher_pred_24=val_teacher_pred,
        teacher_mask_24=val_teacher_mask,
        num_teachers=num_teachers,
    )

    return train_ds, val_ds, feature_cols


def build_predict_dataset_v3(
    raw_df: pd.DataFrame,
    feature_config=None,
    *,
    target_day: pd.Timestamp,
    visible_frame: pd.DataFrame | None = None,
    teacher_pred_df: pd.DataFrame | None = None,
    num_teachers: int = 3,
) -> tuple[DaySequenceDatasetV3, list[str]]:
    """Build a prediction DaySequenceDatasetV3 for a single target day.

    Args:
        raw_df: Raw data DataFrame
        feature_config: SGDFNet FeatureConfig (uses default if None)
        target_day: Day to predict
        visible_frame: Cutoff-safe frame for feature computation
        teacher_pred_df: Optional teacher predictions DataFrame
        num_teachers: Number of teacher slots

    Returns:
        (dataset, feature_cols)
    """
    if feature_config is None:
        feature_config = _get_feature_config()

    if visible_frame is not None:
        frame, feature_cols = _bridge.preprocess_dataframe(
            visible_frame, feature_config, rt_history_col="visible_rt_anchor",
        )
    else:
        frame, feature_cols = _bridge.preprocess_dataframe(raw_df, feature_config)

    target_frame = frame[frame["business_day"] == target_day].copy()
    if target_frame.empty:
        raise ValueError(f"No rows found for target_day={target_day}")

    for col in feature_cols:
        if col in target_frame.columns:
            target_frame[col] = target_frame[col].fillna(0.0)

    # Build teacher arrays
    teacher_pred, teacher_mask_arr = None, None
    if teacher_pred_df is not None and not teacher_pred_df.empty:
        days = [pd.Timestamp(target_day)]
        teacher_pred, teacher_mask_arr = _align_teacher_predictions(
            teacher_pred_df, days, num_teachers,
        )

    ds = DaySequenceDatasetV3(
        target_frame, feature_cols, mode="predict",
        teacher_pred_24=teacher_pred,
        teacher_mask_24=teacher_mask_arr,
        num_teachers=num_teachers,
    )
    return ds, feature_cols
