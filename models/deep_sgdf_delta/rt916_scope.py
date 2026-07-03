"""RT916 teacher scope restriction — local teacher only.

RT916 has weaker overall performance than SGDFNet (sMAPE 33.76 vs 26.61).
It must NOT participate in teacher distillation for normal low-volatility hours.

RT916 is only enabled when at least one of:
  - period is 9_16 or 17_24
  - high_price_bucket (|rt_actual| > spike_threshold)
  - high_volatility_bucket (|delta_true| > volatility_threshold)
  - SGDFNet historical error is high AND RT916 historical error is low

If RT916 local performance (on allowed hours) is also weak, auto-disable it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RT916ScopeConfig:
    """Configuration for RT916 teacher scope restriction."""

    enabled: bool = True
    mode: str = "high_volatility_only"
    allowed_periods: list[str] = field(default_factory=lambda: ["9_16", "17_24"])
    spike_threshold: float = 500.0
    volatility_threshold: float = 100.0
    min_disagreement_with_sgdfnet: float = 50.0
    max_global_weight: float = 0.15
    auto_disable_if_local_smape_worse_than_sgdfnet: bool = True


def _period_from_hour(hours: np.ndarray) -> np.ndarray:
    """Map hour (1-24) to period string.

    Returns array of period strings: '1_8', '9_16', '17_24'.
    """
    periods = np.full(len(hours), "", dtype=object)
    periods[(hours >= 1) & (hours <= 8)] = "1_8"
    periods[(hours >= 9) & (hours <= 16)] = "9_16"
    periods[(hours >= 17) & (hours <= 24)] = "17_24"
    return periods


def compute_scope_mask(
    rt_actual: np.ndarray,
    segment_ids: np.ndarray,
    delta_true: np.ndarray,
    sgdfnet_pred: np.ndarray | None = None,
    da_anchor: np.ndarray | None = None,
    config: RT916ScopeConfig | None = None,
) -> np.ndarray:
    """Compute hour-level boolean mask: True where RT916 is allowed.

    Args:
        rt_actual: [num_days, 24] actual realtime prices
        segment_ids: [num_days, 24] segment IDs (0=1_8, 1=9_16, 2=17_24)
        delta_true: [num_days, 24] actual delta (rt - da)
        sgdfnet_pred: [num_days, 24] SGDFNet delta predictions (optional)
        da_anchor: [num_days, 24] day-ahead anchor prices (optional)
        config: RT916ScopeConfig (uses default if None)

    Returns:
        scope_mask: [num_days, 24] bool array — True where RT916 is allowed
    """
    if config is None:
        config = RT916ScopeConfig()

    num_days, _ = rt_actual.shape
    scope_mask = np.zeros((num_days, 24), dtype=bool)

    if not config.enabled:
        # If disabled, allow everywhere (no restriction)
        scope_mask[:] = True
        return scope_mask

    # Rule 1: allowed periods
    period_map = {"1_8": 0, "9_16": 1, "17_24": 2}
    allowed_seg_ids = set()
    for p in config.allowed_periods:
        if p in period_map:
            allowed_seg_ids.add(period_map[p])
    if allowed_seg_ids:
        for sid in allowed_seg_ids:
            scope_mask |= (segment_ids == sid)

    # Rule 2: high price bucket
    scope_mask |= (np.abs(rt_actual) > config.spike_threshold)

    # Rule 3: high volatility bucket
    scope_mask |= (np.abs(delta_true) > config.volatility_threshold)

    # Rule 4: SGDFNet error high AND RT916 error low (disagreement)
    if sgdfnet_pred is not None and da_anchor is not None:
        sgdfnet_rt = sgdfnet_pred + da_anchor
        sgdfnet_error = np.abs(sgdfnet_rt - rt_actual)
        # Use delta error as proxy for RT916 error (we don't have RT916 pred here)
        # This rule is applied later when RT916 predictions are available
        # For now, mark hours where SGDFNet error is very high
        high_sgdfnet_error = sgdfnet_error > config.min_disagreement_with_sgdfnet
        scope_mask |= high_sgdfnet_error

    return scope_mask


def apply_rt916_scope(
    teacher_pred: np.ndarray,
    teacher_mask: np.ndarray,
    teacher_names: list[str],
    rt916_idx: int,
    rt_actual: np.ndarray,
    segment_ids: np.ndarray,
    delta_true: np.ndarray,
    sgdfnet_pred: np.ndarray | None = None,
    da_anchor: np.ndarray | None = None,
    config: RT916ScopeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply RT916 scope restriction to teacher arrays.

    Modifies teacher_pred in-place: sets RT916 predictions to NaN for
    disallowed hours. The loss function's isfinite check will then
    exclude these hours from distillation.

    Args:
        teacher_pred: [num_days, num_teachers, 24] teacher predictions
        teacher_mask: [num_days, num_teachers] teacher availability
        teacher_names: list of teacher names (e.g., ["sgdfnet", "rt916", "timemixer"])
        rt916_idx: index of RT916 in teacher arrays
        rt_actual: [num_days, 24] actual realtime prices
        segment_ids: [num_days, 24] segment IDs
        delta_true: [num_days, 24] actual delta
        sgdfnet_pred: [num_days, 24] SGDFNet delta predictions (optional)
        da_anchor: [num_days, 24] day-ahead anchor (optional)
        config: RT916ScopeConfig

    Returns:
        (teacher_pred, teacher_mask, stats_dict)
    """
    if config is None:
        config = RT916ScopeConfig()

    stats = {
        "enabled": config.enabled,
        "total_rt916_hours": 0,
        "allowed_hours": 0,
        "blocked_hours": 0,
        "auto_disabled": False,
    }

    if not config.enabled:
        return teacher_pred, teacher_mask, stats

    if rt916_idx < 0 or rt916_idx >= teacher_pred.shape[1]:
        logger.warning("RT916 index %d out of range [0, %d)", rt916_idx, teacher_pred.shape[1])
        return teacher_pred, teacher_mask, stats

    if "rt916" not in teacher_names:
        logger.info("RT916 not in teacher names %s — scope not applied", teacher_names)
        return teacher_pred, teacher_mask, stats

    # Compute scope mask
    scope_mask = compute_scope_mask(
        rt_actual, segment_ids, delta_true,
        sgdfnet_pred=sgdfnet_pred, da_anchor=da_anchor, config=config,
    )

    # Count RT916 available hours
    rt916_available = teacher_mask[:, rt916_idx] > 0  # [num_days]
    total_rt916_hours = int(rt916_available.sum()) * 24  # rough upper bound

    # Actually count hours where RT916 has finite predictions
    rt916_finite = np.isfinite(teacher_pred[:, rt916_idx, :])  # [num_days, 24]
    total_rt916_hours = int(rt916_finite.sum())

    # Block disallowed hours: set teacher_pred to NaN
    blocked = (~scope_mask) & rt916_finite  # hours where RT916 is finite but not allowed
    num_blocked = int(blocked.sum())
    num_allowed = total_rt916_hours - num_blocked

    # Use np.where to avoid boolean indexing issues with 3D arrays
    teacher_pred[:, rt916_idx, :] = np.where(
        blocked, np.nan, teacher_pred[:, rt916_idx, :]
    )

    # Update teacher_mask: if all hours are blocked for a day, set mask to 0
    rt916_any_finite = np.isfinite(teacher_pred[:, rt916_idx, :]).any(axis=1)  # [num_days]
    teacher_mask[~rt916_any_finite, rt916_idx] = 0.0

    stats["total_rt916_hours"] = total_rt916_hours
    stats["allowed_hours"] = num_allowed
    stats["blocked_hours"] = num_blocked

    logger.info(
        "RT916 scope: %d/%d hours allowed (%d blocked)",
        num_allowed, total_rt916_hours, num_blocked,
    )

    if num_allowed == 0:
        logger.warning("RT916 scope blocked ALL hours — auto-disabling RT916 teacher")
        teacher_mask[:, rt916_idx] = 0.0
        teacher_pred[:, rt916_idx, :] = np.nan
        stats["auto_disabled"] = True

    return teacher_pred, teacher_mask, stats


def evaluate_rt916_local_quality(
    teacher_pred: np.ndarray,
    teacher_mask: np.ndarray,
    teacher_names: list[str],
    rt916_idx: int,
    rt_actual: np.ndarray,
    da_anchor: np.ndarray | None = None,
    teacher_pred_kind: str = "delta",
    sgdfnet_pred: np.ndarray | None = None,
    sgdfnet_idx: int = 0,
) -> dict:
    """Evaluate RT916 quality on its allowed hours only.

    Args:
        teacher_pred: [num_days, num_teachers, 24] teacher predictions
        teacher_mask: [num_days, num_teachers] teacher availability
        teacher_names: list of teacher names
        rt916_idx: index of RT916 in teacher arrays
        rt_actual: [num_days, 24] actual realtime prices
        da_anchor: [num_days, 24] day-ahead anchor prices (required if teacher_pred_kind="delta")
        teacher_pred_kind: "rt" if teacher_pred is RT price, "delta" if teacher_pred is delta
        sgdfnet_pred: [num_days, 24] SGDFNet delta predictions (for comparison)
        sgdfnet_idx: index of SGDFNet in teacher arrays

    Returns dict with:
      - rt916_local_smape: sMAPE_floor50 on allowed hours (RT price space)
      - sgdfnet_local_smape: SGDFNet sMAPE on same hours
      - rt916_is_better: bool
      - recommendation: "keep" | "disable"
    """
    if rt916_idx < 0 or rt916_idx >= teacher_pred.shape[1]:
        return {"rt916_local_smape": float("inf"), "recommendation": "disable"}

    # Find hours where RT916 is available
    rt916_finite = np.isfinite(teacher_pred[:, rt916_idx, :])
    if rt916_finite.sum() == 0:
        return {"rt916_local_smape": float("inf"), "recommendation": "disable"}

    # Extract RT916 predictions and convert to RT price space
    rt916_pred_vals = teacher_pred[:, rt916_idx, :][rt916_finite]
    rt_actual_vals = rt_actual[rt916_finite]

    if teacher_pred_kind == "rt":
        # Teacher predictions are already RT prices
        rt916_rt_pred = rt916_pred_vals
    elif teacher_pred_kind == "delta":
        # Teacher predictions are delta; convert to RT price
        if da_anchor is None:
            logger.warning("da_anchor required for teacher_pred_kind='delta', using zeros")
            da_anchor_vals = np.zeros_like(rt916_pred_vals)
        else:
            da_anchor_vals = da_anchor[rt916_finite]
        rt916_rt_pred = da_anchor_vals + rt916_pred_vals
    else:
        raise ValueError(f"teacher_pred_kind must be 'rt' or 'delta', got '{teacher_pred_kind}'")

    # Compute sMAPE_floor50 in RT price space
    floor = 50.0
    rt916_rt_clipped = np.clip(np.abs(rt916_rt_pred), floor, None)
    rt_actual_clipped = np.clip(np.abs(rt_actual_vals), floor, None)
    rt916_smape = float(np.mean(
        200.0 * np.abs(rt916_rt_clipped - rt_actual_clipped)
        / (np.abs(rt916_rt_clipped) + np.abs(rt_actual_clipped) + 1e-6)
    ))

    result = {
        "rt916_local_smape": rt916_smape,
        "rt916_allowed_hours": int(rt916_finite.sum()),
        "teacher_pred_kind": teacher_pred_kind,
        "recommendation": "keep",
    }

    # Compare with SGDFNet on same hours
    if sgdfnet_pred is not None and sgdfnet_idx < teacher_pred.shape[1]:
        sgdfnet_finite = np.isfinite(teacher_pred[:, sgdfnet_idx, :])
        both_finite = rt916_finite & sgdfnet_finite
        if both_finite.sum() > 0:
            sgd_delta_vals = teacher_pred[:, sgdfnet_idx, :][both_finite]
            rt_actual_both = rt_actual[both_finite]

            # Convert SGDFNet delta to RT price
            if da_anchor is not None:
                da_anchor_both = da_anchor[both_finite]
                sgd_rt_pred = da_anchor_both + sgd_delta_vals
            else:
                sgd_rt_pred = sgd_delta_vals  # fallback

            sgd_rt_clipped = np.clip(np.abs(sgd_rt_pred), floor, None)
            rt_actual_both_clipped = np.clip(np.abs(rt_actual_both), floor, None)
            sgdfnet_smape = float(np.mean(
                200.0 * np.abs(sgd_rt_clipped - rt_actual_both_clipped)
                / (np.abs(sgd_rt_clipped) + np.abs(rt_actual_both_clipped) + 1e-6)
            ))
            result["sgdfnet_local_smape"] = sgdfnet_smape
            result["rt916_is_better"] = rt916_smape < sgdfnet_smape

            if rt916_smape > sgdfnet_smape * 1.2:  # 20% worse
                result["recommendation"] = "disable"
                logger.warning(
                    "RT916 local sMAPE %.2f is 20%%+ worse than SGDFNet %.2f — recommend disable",
                    rt916_smape, sgdfnet_smape,
                )

    return result
