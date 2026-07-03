"""DaySequenceDataset for DeepSGDFDeltaV2.

Each sample is one full business_day with 24 feature rows (hour 1-24).
Produces:
  - features_24h:    [24, input_dim]  per-hour feature vectors
  - delta_target_24: [24]             per-hour delta targets
  - da_anchor_24:    [24]             per-hour day-ahead anchor prices
  - rt_actual_24:    [24]             per-hour realtime actual prices
  - segment_ids_24:  [24]             per-hour segment IDs (0/1/2)
  - segment_id:      scalar           majority segment for the day
  - valid_mask:      [24]             1 where hour is present, 0 if padded
  - normal_mask:     [24]             not high-price and not negative
  - high_price_mask: [24]             |rt_actual| > 500
  - negative_mask:   [24]             rt_actual < 0
  - segment_916_mask:[24]             segment_id == 1

Supports train / val / predict modes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── SGDFNet integration via bridge (lazy) ────────────────────────────
from models.deep_sgdf_delta import sgdfnet_bridge as _bridge


def _get_feature_config():
    """Lazily create the default FeatureConfig via the bridge."""
    return _bridge.FeatureConfig(
        include_forecast_columns=True,
        include_actual_history_columns=False,
        use_visible_actual_history=True,
        include_delta_history_features=True,
        include_tf_moving_average_features=False,
        include_static_group_graph_features=False,
        include_weekly_history_features=False,
        include_forecast_residual_history_features=False,
        include_segment_local_stats=False,
        include_forecast_pressure_interactions=False,
        include_calendar_features=True,
        include_engineered_forecast_features=True,
    )


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


def _resolve_feature_columns(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Keep only feature columns that actually exist in the frame."""
    return [c for c in feature_cols if c in frame.columns]


class DaySequenceDataset(Dataset):
    """PyTorch Dataset for day-level 24-hour prediction.

    Each sample is one business_day with 24 hourly rows.
    Missing hours are zero-padded with valid_mask=0.

    Args:
        frame: DataFrame with columns from SGDFNet preprocess_dataframe
        feature_cols: list of feature column names to use
        mode: "train", "val", or "predict"
        spike_threshold: threshold for high_price_mask
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        *,
        mode: Literal["train", "val", "predict"] = "train",
        spike_threshold: float = 500.0,
    ):
        self.feature_cols = _resolve_feature_columns(frame, feature_cols)
        self.mode = mode
        self.spike_threshold = spike_threshold
        self.input_dim = len(self.feature_cols)

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

        for bd, group in grouped:
            group = group.set_index("target_hour")

            # Skip days with no usable data in predict mode
            if mode != "predict":
                if "delta_target" not in group.columns:
                    continue
                # Need at least 1 valid delta target
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
                idx = hour - 1  # 0-based index in the 24-element array
                if hour in group.index:
                    row = group.loc[hour]
                    # Handle duplicate hours (take first)
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
                            # rt_actual = da_anchor + delta_target
                            rt_arr[idx] = da_arr[idx] + delta_arr[idx]
                    else:
                        rt_arr[idx] = da_arr[idx]  # placeholder

            self._days.append(bd)
            self._day_features.append(feat_arr)
            self._day_delta.append(delta_arr)
            self._day_da.append(da_arr)
            self._day_rt.append(rt_arr)
            self._day_seg.append(seg_arr)
            self._day_valid.append(valid_arr)

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
        seg_counts = np.bincount(seg[valid.astype(bool).astype(int)] if valid.sum() > 0 else seg, minlength=3)
        majority_seg = int(np.argmax(seg_counts))

        # Compute masks
        rt_tensor = torch.from_numpy(rt)
        high_price_mask = (rt_tensor.abs() > self.spike_threshold).float()
        negative_mask = (rt_tensor < 0.0).float()
        normal_mask = ((1.0 - high_price_mask) * (1.0 - negative_mask) * torch.from_numpy(valid)).float()
        seg_916_mask = (torch.from_numpy(seg) == 1).float() * torch.from_numpy(valid)

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

def collate_fn_v2(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader — stacks day-level samples."""
    return {
        "features_24h": torch.stack([b["features_24h"] for b in batch]),
        "delta_target_24": torch.stack([b["delta_target_24"] for b in batch]),
        "da_anchor_24": torch.stack([b["da_anchor_24"] for b in batch]),
        "rt_actual_24": torch.stack([b["rt_actual_24"] for b in batch]),
        "segment_ids_24": torch.stack([b["segment_ids_24"] for b in batch]),
        "segment_id": torch.stack([b["segment_id"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
        "normal_mask": torch.stack([b["normal_mask"] for b in batch]),
        "high_price_mask": torch.stack([b["high_price_mask"] for b in batch]),
        "negative_mask": torch.stack([b["negative_mask"] for b in batch]),
        "segment_916_mask": torch.stack([b["segment_916_mask"] for b in batch]),
        "business_day": torch.stack([b["business_day"] for b in batch]),
    }


# ── Builder functions ────────────────────────────────────────────────

def build_training_datasets_v2(
    raw_df: pd.DataFrame,
    feature_config: FeatureConfig,
    *,
    decision_day: pd.Timestamp,
    val_days: int = 30,
    train_min_days: int = 90,
) -> tuple[DaySequenceDataset, DaySequenceDataset, list[str]]:
    """Build train and val DaySequenceDatasets for a single decision day.

    Returns (train_ds, val_ds, feature_cols).
    """
    frame, feature_cols = _bridge.preprocess_dataframe(raw_df, feature_config)

    # Split by business_day
    val_start = decision_day - pd.Timedelta(days=val_days)
    train_frame = frame[frame["business_day"] < val_start].copy()
    val_frame = frame[
        (frame["business_day"] >= val_start) & (frame["business_day"] < decision_day)
    ].copy()

    if train_frame["business_day"].nunique() < train_min_days:
        raise ValueError(
            f"Insufficient training days: {train_frame['business_day'].nunique()} < {train_min_days}"
        )

    # Fill NaN in features with 0
    for col in feature_cols:
        if col in train_frame.columns:
            train_frame[col] = train_frame[col].fillna(0.0)
            val_frame[col] = val_frame[col].fillna(0.0)

    train_ds = DaySequenceDataset(train_frame, feature_cols, mode="train")
    val_ds = DaySequenceDataset(val_frame, feature_cols, mode="val")

    return train_ds, val_ds, feature_cols


def build_predict_dataset_v2(
    raw_df: pd.DataFrame,
    feature_config: FeatureConfig,
    *,
    target_day: pd.Timestamp,
    visible_frame: pd.DataFrame | None = None,
) -> tuple[DaySequenceDataset, list[str]]:
    """Build a prediction DaySequenceDataset for a single target day.

    Uses the visible frame (cutoff-safe) for feature computation.
    """
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

    ds = DaySequenceDataset(target_frame, feature_cols, mode="predict")
    return ds, feature_cols
