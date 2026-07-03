"""Unified realtime dataset for TrendKnightRT (final version).

Provides a PyTorch ``Dataset`` that produces full-day 24-hour samples
aligned by ``business_day`` / ``hour_business`` and two training targets:

- **delta_target**    = rt_actual - da_anchor
- **residual_target** = rt_actual - sgdfnet_pred

Walk-forward splitting:

    train  = all business_days before (target_month - val_days)
    val    = last *val_days* days of the training window
    test   = target_month

No leakage: the target-hour actual and any future-hour actuals are
forbidden in the feature columns.

NaN features are filled with 0.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .business_time import add_business_time_columns
from .realtime_feature_contract import (
    ALL_FEATURES,
    REQUIRED_FEATURES,
    OPTIONAL_FEATURES,
    FEATURE_VERSION,
    build_feature_manifest,
    check_leakage,
    get_period,
    validate_features,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _resolve_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[str]:
    """Return the intersection of *feature_cols* and *frame.columns*.

    If *feature_cols* is ``None``, defaults to ``ALL_FEATURES`` from the
    realtime feature contract.
    """
    if feature_cols is None:
        feature_cols = list(ALL_FEATURES)
    return [c for c in feature_cols if c in frame.columns]


def _ensure_da_anchor(frame: pd.DataFrame) -> pd.DataFrame:
    """Make sure a ``da_anchor`` column exists.

    If only ``forecast_price`` is present, it is copied to ``da_anchor``.
    """
    if "da_anchor" not in frame.columns:
        if "forecast_price" in frame.columns:
            frame = frame.copy()
            frame["da_anchor"] = frame["forecast_price"]
        else:
            raise ValueError(
                "DataFrame must contain either 'da_anchor' or 'forecast_price'."
            )
    return frame


def _ensure_sgdfnet_pred(
    frame: pd.DataFrame,
    *,
    allow_fallback: bool = False,
) -> pd.DataFrame:
    """Ensure ``sgdfnet_pred`` exists.

    Args:
        frame: Input DataFrame.
        allow_fallback: If ``True``, missing ``sgdfnet_pred`` values are
            filled from ``da_anchor``.  If ``False`` (default), missing
            ``sgdfnet_pred`` raises ``ValueError``.

    Returns:
        DataFrame with ``sgdfnet_pred`` column guaranteed to be present.

    Raises:
        ValueError: If ``sgdfnet_pred`` is missing or has NaN values and
            *allow_fallback* is ``False``.
    """
    frame = frame.copy()
    n_missing_total = 0
    fallback_used = False

    if "sgdfnet_pred" not in frame.columns:
        if allow_fallback:
            if "da_anchor" in frame.columns:
                frame["sgdfnet_pred"] = frame["da_anchor"]
                fallback_used = True
                n_missing_total = len(frame)
                logger.warning(
                    "sgdfnet_pred not found — fallback to da_anchor (%d rows). "
                    "This is only acceptable for smoke/predict runs.",
                    n_missing_total,
                )
            else:
                raise ValueError(
                    "Cannot fallback: neither sgdfnet_pred nor da_anchor present."
                )
        else:
            raise ValueError(
                "Missing sgdfnet_pred for formal training. "
                "Provide SGDFNet predictions or use --allow-sgdfnet-fallback "
                "for smoke only."
            )
    else:
        # Column exists — check for NaN
        mask = frame["sgdfnet_pred"].isna()
        n_nan = int(mask.sum())
        if n_nan > 0:
            if allow_fallback:
                frame.loc[mask, "sgdfnet_pred"] = frame.loc[mask, "da_anchor"]
                fallback_used = True
                n_missing_total = n_nan
                logger.warning(
                    "sgdfnet_pred has %d NaN rows — fallback to da_anchor. "
                    "This is only acceptable for smoke/predict runs.",
                    n_nan,
                )
            else:
                raise ValueError(
                    f"sgdfnet_pred has {n_nan} NaN values in formal training. "
                    "Provide complete SGDFNet predictions or use "
                    "--allow-sgdfnet-fallback for smoke only."
                )

    # Store coverage metadata on the frame
    total_rows = len(frame)
    n_present = total_rows - n_missing_total
    coverage_pct = (n_present / total_rows * 100) if total_rows > 0 else 0.0
    frame.attrs["sgdfnet_fallback_used"] = fallback_used
    frame.attrs["sgdfnet_coverage"] = coverage_pct

    return frame


# ── Dataset ──────────────────────────────────────────────────────────

class RealtimeDayDataset(Dataset):
    """PyTorch Dataset for TrendKnightRT day-level 24-hour samples.

    Each sample is one ``business_day`` with 24 hourly rows (hour_business
    1-24).  The dataset produces:

    - ``features_24h``       : ``[24, feature_dim]`` float32
    - ``delta_target_24``    : ``[24]`` float32  (rt_actual - da_anchor)
    - ``residual_target_24`` : ``[24]`` float32  (rt_actual - sgdfnet_pred)
    - ``mask_24``            : ``[24]`` float32  (1 = valid hour)
    - ``period_24``          : ``[24]`` int64    (0="1_8", 1="9_16", 2="17_24")
    - ``segment_id``         : scalar int64      (majority segment for the day)
    - ``da_anchor_24``       : ``[24]`` float32
    - ``sgdfnet_pred_24``    : ``[24]`` float32
    - ``hour_ids``           : ``[24]`` int64    (1-24)
    - ``business_day``       : scalar float64    (unix timestamp)

    Args:
        frame: pandas DataFrame.  Must contain at minimum ``ds``,
            ``rt_actual`` (or be in predict mode), and ``da_anchor`` /
            ``forecast_price``.  Business-time columns are added
            automatically if absent.
        feature_cols: Feature column names to use.  ``None`` defaults
            to ``ALL_FEATURES`` from the contract.
        mode: ``"train"``, ``"val"``, ``"test"``, or ``"predict"``.
        cutoff_hour: Leakage cutoff hour (default 15).
        allow_sgdfnet_fallback: If ``True``, missing ``sgdfnet_pred``
            is filled from ``da_anchor``.  Only for smoke/predict runs.

    Raises:
        ValueError: If required columns are missing (checked via the
            feature contract).
    """

    # Period string -> integer encoding
    _PERIOD_MAP = {"1_8": 0, "9_16": 1, "17_24": 2}

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str] | None = None,
        *,
        mode: Literal["train", "val", "test", "predict"] = "train",
        cutoff_hour: int = 15,
        allow_sgdfnet_fallback: bool = False,
    ):
        self.mode = mode
        self.cutoff_hour = cutoff_hour

        # ── Prepare frame ────────────────────────────────────────
        frame = frame.copy()

        # Add business-time alignment if not already present
        if "business_day" not in frame.columns or "hour_business" not in frame.columns:
            frame = add_business_time_columns(frame, timestamp_col="ds")

        frame = _ensure_da_anchor(frame)
        frame = _ensure_sgdfnet_pred(frame, allow_fallback=allow_sgdfnet_fallback)

        # Ensure numeric types
        frame["business_day"] = pd.to_datetime(frame["business_day"]).dt.normalize()
        frame["hour_business"] = frame["hour_business"].astype(int)

        # Resolve feature columns
        self.feature_cols = _resolve_feature_columns(frame, feature_cols)
        self.input_dim = len(self.feature_cols)

        # Validate required features (warn only — we still proceed)
        missing = validate_features(frame)
        if missing:
            logger.warning(
                "[%s] Missing required features (will be zero-filled): %s",
                mode, missing,
            )

        # Fill NaN in feature columns with 0
        for col in self.feature_cols:
            if col in frame.columns:
                frame[col] = frame[col].fillna(0.0)

        # ── Build per-day arrays ─────────────────────────────────
        self._days: list[pd.Timestamp] = []
        self._day_features: list[np.ndarray] = []      # [24, input_dim]
        self._day_delta: list[np.ndarray] = []          # [24]
        self._day_residual: list[np.ndarray] = []       # [24]
        self._day_mask: list[np.ndarray] = []           # [24]
        self._day_period: list[np.ndarray] = []         # [24]
        self._day_da: list[np.ndarray] = []             # [24]
        self._day_sgdfnet: list[np.ndarray] = []        # [24]
        self._day_hour_ids: list[np.ndarray] = []       # [24]
        self._day_seg: list[np.ndarray] = []            # [24]

        grouped = frame.groupby("business_day")
        for bd, group in grouped:
            # In train/val/test mode, skip days without usable targets
            if mode != "predict":
                if "rt_actual" not in group.columns:
                    continue
                if group["rt_actual"].isna().all():
                    continue

            feat_arr = np.zeros((24, self.input_dim), dtype=np.float32)
            delta_arr = np.zeros(24, dtype=np.float32)
            residual_arr = np.zeros(24, dtype=np.float32)
            mask_arr = np.zeros(24, dtype=np.float32)
            period_arr = np.zeros(24, dtype=np.int64)
            da_arr = np.zeros(24, dtype=np.float32)
            sgdfnet_arr = np.zeros(24, dtype=np.float32)
            hour_arr = np.arange(1, 25, dtype=np.int64)
            seg_arr = np.zeros(24, dtype=np.int64)

            # Index group by hour_business for fast lookup
            group_indexed = group.set_index("hour_business")

            for hour in range(1, 25):
                idx = hour - 1
                if hour not in group_indexed.index:
                    continue

                row = group_indexed.loc[hour]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                # Features
                feat_vals = []
                for col in self.feature_cols:
                    if col in frame.columns:
                        val = row.get(col, 0.0)
                        feat_vals.append(float(val) if pd.notna(val) else 0.0)
                    else:
                        feat_vals.append(0.0)
                feat_arr[idx] = np.array(feat_vals, dtype=np.float32)

                # Anchors and predictions
                da_val = float(row.get("da_anchor", 0.0))
                sgd_val = float(row.get("sgdfnet_pred", da_val))
                da_arr[idx] = da_val if pd.notna(da_val) else 0.0
                sgdfnet_arr[idx] = sgd_val if pd.notna(sgd_val) else da_arr[idx]

                # Period
                period_str = get_period(hour)
                period_arr[idx] = self._PERIOD_MAP.get(period_str, 0)

                # Segment (if available)
                seg_val = row.get("segment_id", 0)
                seg_arr[idx] = int(seg_val) if pd.notna(seg_val) else 0

                # Mask
                mask_arr[idx] = 1.0

                # Targets (only in non-predict modes)
                if mode != "predict" and "rt_actual" in group.columns:
                    rt_val = row.get("rt_actual")
                    if pd.notna(rt_val):
                        rt = float(rt_val)
                        delta_arr[idx] = rt - da_arr[idx]
                        residual_arr[idx] = rt - sgdfnet_arr[idx]

            self._days.append(bd)
            self._day_features.append(feat_arr)
            self._day_delta.append(delta_arr)
            self._day_residual.append(residual_arr)
            self._day_mask.append(mask_arr)
            self._day_period.append(period_arr)
            self._day_da.append(da_arr)
            self._day_sgdfnet.append(sgdfnet_arr)
            self._day_hour_ids.append(hour_arr)
            self._day_seg.append(seg_arr)

    # ── PyTorch interface ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._days)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feat = self._day_features[idx]
        delta = self._day_delta[idx]
        residual = self._day_residual[idx]
        mask = self._day_mask[idx]
        period = self._day_period[idx]
        da = self._day_da[idx]
        sgdfnet = self._day_sgdfnet[idx]
        hour_ids = self._day_hour_ids[idx]
        seg = self._day_seg[idx]

        # Majority segment for the day
        valid_bool = mask.astype(bool)
        if valid_bool.sum() > 0:
            seg_counts = np.bincount(seg[valid_bool], minlength=3)
        else:
            seg_counts = np.bincount(seg, minlength=3)
        majority_seg = int(np.argmax(seg_counts))

        return {
            "features_24h": torch.from_numpy(feat),            # [24, feature_dim]
            "delta_target_24": torch.from_numpy(delta),        # [24]
            "residual_target_24": torch.from_numpy(residual),  # [24]
            "mask_24": torch.from_numpy(mask),                 # [24]
            "period_24": torch.from_numpy(period),             # [24]
            "segment_id": torch.tensor(majority_seg, dtype=torch.long),
            "da_anchor_24": torch.from_numpy(da),              # [24]
            "sgdfnet_pred_24": torch.from_numpy(sgdfnet),      # [24]
            "hour_ids": torch.from_numpy(hour_ids),            # [24]
            "business_day": torch.tensor(
                self._days[idx].timestamp(), dtype=torch.float64
            ),
        }

    # ── Properties ───────────────────────────────────────────────

    @property
    def business_days(self) -> list[pd.Timestamp]:
        """Ordered list of business days in this dataset."""
        return list(self._days)

    @property
    def n_days(self) -> int:
        """Number of business days."""
        return len(self._days)


# ── Collate function ─────────────────────────────────────────────────

def collate_fn_final(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader — stacks day-level samples."""
    return {
        "features_24h": torch.stack([b["features_24h"] for b in batch]),
        "delta_target_24": torch.stack([b["delta_target_24"] for b in batch]),
        "residual_target_24": torch.stack([b["residual_target_24"] for b in batch]),
        "mask_24": torch.stack([b["mask_24"] for b in batch]),
        "period_24": torch.stack([b["period_24"] for b in batch]),
        "segment_id": torch.stack([b["segment_id"] for b in batch]),
        "da_anchor_24": torch.stack([b["da_anchor_24"] for b in batch]),
        "sgdfnet_pred_24": torch.stack([b["sgdfnet_pred_24"] for b in batch]),
        "hour_ids": torch.stack([b["hour_ids"] for b in batch]),
        "business_day": torch.stack([b["business_day"] for b in batch]),
    }


# ── Builder: training datasets ───────────────────────────────────────

def build_training_datasets_final(
    raw_df: pd.DataFrame,
    *,
    target_month: str,
    feature_cols: list[str] | None = None,
    val_days: int = 30,
    train_min_days: int = 90,
    cutoff_hour: int = 15,
    allow_sgdfnet_fallback: bool = False,
) -> tuple[RealtimeDayDataset, RealtimeDayDataset, RealtimeDayDataset, dict]:
    """Build train / val / test datasets with walk-forward split.

    Splitting logic:

    1. Parse *target_month* (e.g. ``"2024-06"``) to determine the test
       window: all business_days whose month matches *target_month*.
    2. Everything before the test window is candidate training data.
    3. The last *val_days* calendar days of the training candidate
       become the validation set; the rest is the training set.
    4. The test set contains exactly the *target_month* days.

    Args:
        raw_df: Raw hourly DataFrame.  Must contain ``ds`` and
            ``rt_actual``.  ``da_anchor`` or ``forecast_price`` is also
            required.
        target_month: Target month string, e.g. ``"2024-06"`` or
            ``"2024-06-01"``.
        feature_cols: Feature columns to use.  ``None`` defaults to
            ``ALL_FEATURES`` from the contract.
        val_days: Number of calendar days for validation (default 30).
        train_min_days: Minimum training days required.  Raises
            ``ValueError`` if not met.
        cutoff_hour: Leakage cutoff hour (default 15).

    Returns:
        ``(train_ds, val_ds, test_ds, manifest)`` where *manifest* is a
        dictionary from :func:`build_feature_manifest`.

    Raises:
        ValueError: If insufficient training data or missing columns.
    """
    # ── Prepare frame ────────────────────────────────────────────
    frame = raw_df.copy()
    frame = add_business_time_columns(frame, timestamp_col="ds")
    frame = _ensure_da_anchor(frame)
    frame = _ensure_sgdfnet_pred(frame, allow_fallback=allow_sgdfnet_fallback)

    # ── Determine split boundaries ───────────────────────────────
    target_ts = pd.Timestamp(target_month)
    if target_ts.day == 1 and len(str(target_month)) <= 7:
        # "2024-06" → test is June 2024
        test_start = target_ts
        test_end = test_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    else:
        # Treat as exact date — test is that single day
        test_start = target_ts.normalize()
        test_end = test_start + pd.Timedelta(days=1)

    # All data before the test window
    pre_test = frame[frame["business_day"] < test_start].copy()

    # Validation starts val_days before test_start
    val_start = test_start - pd.Timedelta(days=val_days)

    train_frame = pre_test[pre_test["business_day"] < val_start].copy()
    val_frame = pre_test[
        (pre_test["business_day"] >= val_start)
        & (pre_test["business_day"] < test_start)
    ].copy()
    test_frame = frame[
        (frame["business_day"] >= test_start)
        & (frame["business_day"] < test_end)
    ].copy()

    # ── Validate training size ───────────────────────────────────
    n_train_days = train_frame["business_day"].nunique()
    if n_train_days < train_min_days:
        raise ValueError(
            f"Insufficient training days: {n_train_days} < {train_min_days}. "
            f"Consider lowering train_min_days or providing more data."
        )

    # ── Resolve feature columns (use train frame as reference) ───
    resolved_cols = _resolve_feature_columns(train_frame, feature_cols)
    if not resolved_cols:
        # Fallback: try val + test frames
        resolved_cols = _resolve_feature_columns(frame, feature_cols)

    # ── Fill NaN in feature columns with 0 ───────────────────────
    for ds_frame in (train_frame, val_frame, test_frame):
        for col in resolved_cols:
            if col in ds_frame.columns:
                ds_frame[col] = ds_frame[col].fillna(0.0)

    # ── Leakage check ────────────────────────────────────────────
    leakage_ok = check_leakage(frame, cutoff_hour=cutoff_hour)
    if not leakage_ok:
        logger.warning("Leakage detected in the full frame — proceeding with caution.")

    # ── Build datasets ───────────────────────────────────────────
    train_ds = RealtimeDayDataset(
        train_frame, resolved_cols, mode="train", cutoff_hour=cutoff_hour,
        allow_sgdfnet_fallback=allow_sgdfnet_fallback,
    )
    val_ds = RealtimeDayDataset(
        val_frame, resolved_cols, mode="val", cutoff_hour=cutoff_hour,
        allow_sgdfnet_fallback=allow_sgdfnet_fallback,
    )
    test_ds = RealtimeDayDataset(
        test_frame, resolved_cols, mode="test", cutoff_hour=cutoff_hour,
        allow_sgdfnet_fallback=allow_sgdfnet_fallback,
    )

    # ── Build manifest ───────────────────────────────────────────
    target_columns = ["delta_target", "residual_target"]
    all_days = frame["business_day"].dropna()
    date_range = (all_days.min(), all_days.max()) if len(all_days) > 0 else None

    manifest = build_feature_manifest(
        feature_columns=resolved_cols,
        target_columns=target_columns,
        date_range=date_range,
        n_days=int(frame["business_day"].nunique()),
    )
    manifest.update({
        "n_rows": len(frame),
        "n_train_days": train_ds.n_days,
        "n_val_days": val_ds.n_days,
        "n_test_days": test_ds.n_days,
        "target_month": str(target_month),
        "val_days": val_days,
        "cutoff_hour": cutoff_hour,
        "leakage_checks_passed": leakage_ok,
        "sgdfnet_fallback_used": frame.attrs.get("sgdfnet_fallback_used", False),
        "sgdfnet_coverage": frame.attrs.get("sgdfnet_coverage", 0.0),
    })

    logger.info(
        "Training datasets built: train=%d days, val=%d days, test=%d days",
        train_ds.n_days, val_ds.n_days, test_ds.n_days,
    )

    return train_ds, val_ds, test_ds, manifest


# ── Builder: prediction dataset ──────────────────────────────────────

def build_predict_dataset_final(
    raw_df: pd.DataFrame,
    *,
    target_day: pd.Timestamp | str,
    feature_cols: list[str] | None = None,
    cutoff_hour: int = 15,
    allow_sgdfnet_fallback: bool = True,
) -> tuple[RealtimeDayDataset, dict]:
    """Build a prediction dataset for a single target business day.

    Args:
        raw_df: Raw hourly DataFrame containing the target day's data.
            Must contain ``ds``.  ``da_anchor`` / ``forecast_price`` is
            required; ``sgdfnet_pred`` is optional (falls back to
            ``da_anchor``).
        target_day: The business day to predict.  Strings are parsed
            via ``pd.Timestamp``.
        feature_cols: Feature columns to use.  ``None`` defaults to
            ``ALL_FEATURES``.
        cutoff_hour: Leakage cutoff hour (default 15).

    Returns:
        ``(dataset, manifest)`` — *dataset* is a
        :class:`RealtimeDayDataset` in ``"predict"`` mode; *manifest* is
        a metadata dictionary.

    Raises:
        ValueError: If no rows are found for *target_day*.
    """
    target_day = pd.Timestamp(target_day).normalize()

    frame = raw_df.copy()
    frame = add_business_time_columns(frame, timestamp_col="ds")
    frame = _ensure_da_anchor(frame)
    frame = _ensure_sgdfnet_pred(frame, allow_fallback=allow_sgdfnet_fallback)

    # Filter to the target day
    target_frame = frame[frame["business_day"] == target_day].copy()
    if target_frame.empty:
        raise ValueError(
            f"No rows found for target_day={target_day}. "
            f"Available business_days: "
            f"{sorted(frame['business_day'].unique())[:5]}..."
        )

    # Resolve features
    resolved_cols = _resolve_feature_columns(target_frame, feature_cols)
    if not resolved_cols:
        resolved_cols = _resolve_feature_columns(frame, feature_cols)

    # Fill NaN
    for col in resolved_cols:
        if col in target_frame.columns:
            target_frame[col] = target_frame[col].fillna(0.0)

    # Leakage check
    leakage_ok = check_leakage(target_frame, cutoff_hour=cutoff_hour)

    ds = RealtimeDayDataset(
        target_frame, resolved_cols, mode="predict", cutoff_hour=cutoff_hour,
        allow_sgdfnet_fallback=allow_sgdfnet_fallback,
    )

    manifest = build_feature_manifest(
        feature_columns=resolved_cols,
        target_columns=[],
        date_range=(target_day, target_day),
        n_days=1,
    )
    manifest.update({
        "n_rows": len(target_frame),
        "target_day": str(target_day.date()),
        "cutoff_hour": cutoff_hour,
        "leakage_checks_passed": leakage_ok,
        "mode": "predict",
        "sgdfnet_fallback_used": target_frame.attrs.get("sgdfnet_fallback_used", False),
        "sgdfnet_coverage": target_frame.attrs.get("sgdfnet_coverage", 0.0),
    })

    logger.info(
        "Prediction dataset built: target_day=%s, %d hours, %d features",
        target_day.date(), len(target_frame), len(resolved_cols),
    )

    return ds, manifest
