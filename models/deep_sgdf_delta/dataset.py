"""Dataset for DeepSGDFDelta.

Builds sequence windows from SGDFNet's cutoff-safe feature frame.
Each sample is a (sequence, target) pair where:
  - sequence = past `window_days` days of feature rows for the same business_day hour
  - target = delta_target for the prediction hour on the target day

Key constraints:
  - business_day + hour_business alignment (not ds.date())
  - Only uses cutoff-safe features from SGDFNet Protocol B
  - No leakage of post-cutoff realtime prices
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── SGDFNet integration ──────────────────────────────────────────────
# We import SGDFNet's data contract to reuse feature engineering
_MAIN_PROJECT = Path(__file__).resolve().parent.parent.parent.parent
_SGDFNET_SRC = _MAIN_PROJECT / "SGDFNet" / "src"
if str(_SGDFNET_SRC) not in sys.path:
    sys.path.insert(0, str(_SGDFNET_SRC))

from sgdfnet.data_contract import (  # noqa: E402
    TIMESTAMP_COL,
    DA_COL,
    RT_COL,
    FeatureConfig,
    add_business_time_columns,
    load_dataset,
    preprocess_dataframe,
)


# ── Feature columns that are always available ────────────────────────
# These are the columns produced by SGDFNet's preprocess_dataframe with
# the production config (cutoff_recovery_2026_diag_a_prune_actualside.yaml)
DEFAULT_FEATURE_CONFIG = FeatureConfig(
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


def _resolve_feature_columns(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Keep only feature columns that actually exist in the frame."""
    return [c for c in feature_cols if c in frame.columns]


class DeltaSequenceDataset(Dataset):
    """PyTorch Dataset for delta prediction with sequence windows.

    Each sample:
      x: (window_days, num_features) — past days' feature vectors at same hour
      y_delta: scalar — delta_target for current hour
      segment_id: int — 0/1/2 for 1_8/9_16/17_24
      da_anchor: scalar — day-ahead price for current hour
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        window_days: int = 7,
        *,
        mode: Literal["train", "val", "predict"] = "train",
    ):
        self.feature_cols = _resolve_feature_columns(frame, feature_cols)
        self.window_days = window_days
        self.mode = mode

        # Validate required columns
        required = ["business_day", "target_hour", "segment_id", "da_anchor"]
        for col in required:
            if col not in frame.columns:
                raise ValueError(f"Missing required column: {col}")

        # Build the feature matrix per (business_day, target_hour)
        frame = frame.copy()
        frame["business_day"] = pd.to_datetime(frame["business_day"]).dt.normalize()
        frame["target_hour"] = frame["target_hour"].astype(int)

        # Drop rows with NaN in critical columns
        critical = self.feature_cols + ["delta_target", "business_day", "target_hour"]
        critical = [c for c in critical if c in frame.columns]
        self._full_frame = frame

        # Build lookup: (business_day, target_hour) -> row index
        self._index_map: dict[tuple[pd.Timestamp, int], int] = {}
        for idx, row in frame.iterrows():
            key = (row["business_day"], row["target_hour"])
            self._index_map[key] = idx

        # Build valid sample list (must have enough history)
        unique_days = sorted(frame["business_day"].unique())
        self._day_set = set(unique_days)

        # Collect valid indices
        self._samples: list[int] = []
        for i, (_, row) in enumerate(frame.iterrows()):
            bd = row["business_day"]
            th = row["target_hour"]
            # Check if we have enough history
            if self._has_history(bd, th, window_days):
                if mode != "predict" and pd.isna(row.get("delta_target")):
                    continue
                self._samples.append(frame.index.get_loc(frame.index[i]) if not isinstance(frame.index[i], int) else i)

        # Store frame values as numpy for fast access
        self._feat_values = frame[self.feature_cols].values.astype(np.float32)
        self._delta_target = frame["delta_target"].values.astype(np.float32) if "delta_target" in frame.columns else None
        self._segment_ids = frame["segment_id"].values.astype(np.int64)
        self._da_anchor = frame["da_anchor"].values.astype(np.float32)
        self._bd_values = frame["business_day"].values
        self._hour_values = frame["target_hour"].values.astype(np.int64)

        # Build day-index mapping for fast history lookup
        self._day_to_row_offset: dict[pd.Timestamp, list[tuple[int, int]]] = {}
        for i in range(len(frame)):
            bd = pd.Timestamp(self._bd_values[i])
            th = int(self._hour_values[i])
            if bd not in self._day_to_row_offset:
                self._day_to_row_offset[bd] = []
            self._day_to_row_offset[bd].append((th, i))

    def _has_history(self, business_day: pd.Timestamp, target_hour: int, window_days: int) -> bool:
        """Check if we have at least `window_days` prior days with this hour."""
        count = 0
        for d in range(1, window_days + 2):
            lookback_day = business_day - pd.Timedelta(days=d)
            if lookback_day in self._day_to_row_offset:
                count += 1
                if count >= window_days:
                    return True
        return count >= window_days

    def _get_history_sequence(self, business_day: pd.Timestamp, target_hour: int) -> np.ndarray:
        """Get feature sequence for past `window_days` at the same target_hour."""
        seq = []
        for d in range(self.window_days, 0, -1):
            lookback_day = business_day - pd.Timedelta(days=d)
            hour_rows = self._day_to_row_offset.get(lookback_day, [])
            matched = [row_idx for (th, row_idx) in hour_rows if th == target_hour]
            if matched:
                seq.append(self._feat_values[matched[0]])
            else:
                seq.append(np.zeros(len(self.feature_cols), dtype=np.float32))
        return np.stack(seq, axis=0)  # (window_days, num_features)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame_idx = self._samples[idx]
        bd = pd.Timestamp(self._bd_values[frame_idx])
        th = int(self._hour_values[frame_idx])

        features = self._get_history_sequence(bd, th)
        result = {
            "features": torch.from_numpy(features),
            "segment_id": torch.tensor(self._segment_ids[frame_idx], dtype=torch.long),
            "da_anchor": torch.tensor(self._da_anchor[frame_idx], dtype=torch.float32),
            "hour": torch.tensor(th, dtype=torch.long),
        }
        if self._delta_target is not None:
            result["delta_target"] = torch.tensor(self._delta_target[frame_idx], dtype=torch.float32)
            # rt_actual = da_anchor + delta_target
            result["rt_actual"] = result["da_anchor"] + result["delta_target"]
        return result


def build_training_datasets(
    raw_df: pd.DataFrame,
    feature_config: FeatureConfig,
    *,
    decision_day: pd.Timestamp,
    val_days: int = 30,
    window_days: int = 7,
    train_min_rows: int = 2160,
) -> tuple[DeltaSequenceDataset, DeltaSequenceDataset, list[str]]:
    """Build train and val datasets for a single decision day.

    Returns (train_ds, val_ds, feature_cols).
    """
    # Use SGDFNet's feature engineering
    frame, feature_cols = preprocess_dataframe(raw_df, feature_config)

    # Split by business_day
    val_start = decision_day - pd.Timedelta(days=val_days)
    train_frame = frame[frame["business_day"] < val_start].copy()
    val_frame = frame[(frame["business_day"] >= val_start) & (frame["business_day"] < decision_day)].copy()

    if len(train_frame) < train_min_rows:
        raise ValueError(
            f"Insufficient training data: {len(train_frame)} rows < {train_min_rows} minimum"
        )

    # Fill NaN in features with 0 for deep model
    for col in feature_cols:
        if col in train_frame.columns:
            train_frame[col] = train_frame[col].fillna(0.0)
            val_frame[col] = val_frame[col].fillna(0.0)

    train_ds = DeltaSequenceDataset(train_frame, feature_cols, window_days, mode="train")
    val_ds = DeltaSequenceDataset(val_frame, feature_cols, window_days, mode="val")

    return train_ds, val_ds, feature_cols


def build_predict_dataset(
    raw_df: pd.DataFrame,
    feature_config: FeatureConfig,
    *,
    target_day: pd.Timestamp,
    window_days: int = 7,
    visible_frame: pd.DataFrame | None = None,
) -> tuple[DeltaSequenceDataset, list[str]]:
    """Build a prediction dataset for a single target day.

    Uses the visible frame (cutoff-safe) for feature computation.
    """
    if visible_frame is not None:
        frame, feature_cols = preprocess_dataframe(
            visible_frame, feature_config,
            rt_history_col="visible_rt_anchor",
        )
    else:
        frame, feature_cols = preprocess_dataframe(raw_df, feature_config)

    target_frame = frame[frame["business_day"] == target_day].copy()
    if target_frame.empty:
        raise ValueError(f"No rows found for target_day={target_day}")

    for col in feature_cols:
        if col in target_frame.columns:
            target_frame[col] = target_frame[col].fillna(0.0)

    ds = DeltaSequenceDataset(target_frame, feature_cols, window_days, mode="predict")
    return ds, feature_cols
