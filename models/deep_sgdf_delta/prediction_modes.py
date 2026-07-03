"""Prediction mode definitions — Phase 9.

Defines two prediction modes:
  FULL_DAY:  Predict all 24 hours of business_day D at once.
             No D-day actuals allowed.
  INTRADAY:  Predict remaining hours of business_day D after cutoff_hour.
             D-day actuals up to cutoff_hour are allowed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PredictionMode(str, Enum):
    """Prediction mode enum."""
    FULL_DAY = "FULL_DAY"
    INTRADAY = "INTRADAY"


@dataclass
class FeatureVisibilityRequest:
    """Request to validate feature visibility for a single prediction."""
    mode: PredictionMode
    business_day: pd.Timestamp
    target_hour: int  # hour_business of the hour being predicted
    feature_timestamp: pd.Timestamp  # timestamp of the feature value
    cutoff_hour: Optional[int] = None  # only for INTRADAY mode


def validate_feature_visibility(
    mode: PredictionMode | str,
    business_day: pd.Timestamp | str,
    target_hour: int,
    feature_timestamp: pd.Timestamp | str,
    cutoff_hour: Optional[int] = None,
) -> dict:
    """Validate whether a feature is visible (legal) for a given prediction.

    Parameters
    ----------
    mode : PredictionMode or str
        FULL_DAY or INTRADAY.
    business_day : pd.Timestamp or str
        The business day being predicted.
    target_hour : int
        The hour_business being predicted (1-24).
    feature_timestamp : pd.Timestamp or str
        The timestamp of the feature value.
    cutoff_hour : int, optional
        For INTRADAY mode: the cutoff hour (hours <= cutoff are observed).

    Returns
    -------
    dict with keys:
        - valid: bool
        - reason: str (why invalid, or "ok" if valid)
        - mode: str
        - business_day: str
        - target_hour: int
    """
    mode = PredictionMode(mode) if isinstance(mode, str) else mode
    bd = pd.Timestamp(business_day)
    ft = pd.Timestamp(feature_timestamp)

    result = {
        "valid": True,
        "reason": "ok",
        "mode": mode.value,
        "business_day": str(bd.date()),
        "target_hour": target_hour,
    }

    # Determine the feature's business_day and hour_business
    # Timestamp D 00:00 → business_day D-1, hour 24
    # Timestamp D HH:00 (HH>=1) → business_day D, hour HH
    if ft.hour == 0:
        feat_bd = ft.date() - pd.Timedelta(days=1)
        feat_hb = 24
    else:
        feat_bd = ft.date()
        feat_hb = ft.hour

    # Rule: feature must be from before the target hour on the same day,
    # or from a previous day entirely.
    if isinstance(feat_bd, pd.Timestamp):
        feat_bd_date = feat_bd.date()
    else:
        feat_bd_date = feat_bd

    bd_date = bd.date() if hasattr(bd, 'date') else pd.Timestamp(bd).date()

    if mode == PredictionMode.FULL_DAY:
        # FULL_DAY: no D-day actuals allowed at all
        if feat_bd_date == bd_date:
            # Same business day — only allowed if hour_business < target_hour
            # But for FULL_DAY, we predict ALL hours at once before any occur.
            # So NO same-day features are allowed.
            result["valid"] = False
            result["reason"] = (
                f"FULL_DAY mode: feature from same business_day {bd_date} "
                f"(hour {feat_hb}) is not allowed — no D-day actuals permitted"
            )
            return result
        elif feat_bd_date > bd_date:
            result["valid"] = False
            result["reason"] = (
                f"Feature from future business_day {feat_bd_date} "
                f"(target is {bd_date})"
            )
            return result

    elif mode == PredictionMode.INTRADAY:
        if cutoff_hour is None:
            result["valid"] = False
            result["reason"] = "INTRADAY mode requires cutoff_hour"
            return result

        if feat_bd_date == bd_date:
            # Same business day: only hours <= cutoff_hour are allowed
            if feat_hb > cutoff_hour:
                result["valid"] = False
                result["reason"] = (
                    f"INTRADAY cutoff_hour={cutoff_hour}: feature from "
                    f"hour {feat_hb} is after cutoff — not yet observed"
                )
                return result
            # Also: must be before target_hour (can't use future hour to predict past)
            if feat_hb >= target_hour:
                result["valid"] = False
                result["reason"] = (
                    f"Feature hour {feat_hb} >= target hour {target_hour} — "
                    f"cannot use future/present to predict past"
                )
                return result
        elif feat_bd_date > bd_date:
            result["valid"] = False
            result["reason"] = (
                f"Feature from future business_day {feat_bd_date} "
                f"(target is {bd_date})"
            )
            return result

    return result


def validate_intraday_cutoff(
    cutoff_hour: int,
    target_hour: int,
) -> bool:
    """Quick check: is target_hour reachable from cutoff_hour?

    For INTRADAY mode, target must be > cutoff_hour (predicting future hours).
    """
    return target_hour > cutoff_hour
