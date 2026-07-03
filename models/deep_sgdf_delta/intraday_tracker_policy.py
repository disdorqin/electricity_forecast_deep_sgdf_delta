"""Intraday Tracker Policy — Phase 10.

Cutoff gating policy for the IntradayResidualTracker.
Decides whether to apply correction, shadow-only, or disable entirely,
based on mode, cutoff_hour, confidence, residual statistics, and risk flags.

Policy decisions:
  DISABLED    — tracker not used at all (fusion_weight = 0)
  SHADOW_ONLY — tracker runs but correction not applied (fusion_weight = 0)
  LOW_WEIGHT  — tracker applied with low weight (fusion_weight = 0.10~0.15)
  HIGH_WEIGHT — tracker applied with higher weight (fusion_weight = 0.20~0.25)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from models.deep_sgdf_delta.intraday_residual_tracker import IntradayResidualState
from models.deep_sgdf_delta.prediction_modes import PredictionMode

logger = logging.getLogger(__name__)


class PolicyDecision(str, Enum):
    DISABLED = "DISABLED"
    SHADOW_ONLY = "SHADOW_ONLY"
    LOW_WEIGHT = "LOW_WEIGHT"
    HIGH_WEIGHT = "HIGH_WEIGHT"


@dataclass
class PolicyConfig:
    """Configuration for the intraday tracker policy."""
    # Minimum cutoff hour to allow any correction
    min_cutoff_for_correction: int = 12
    # Minimum cutoff hour to allow shadow mode (below this → disabled)
    min_cutoff_for_shadow: int = 10
    # Minimum confidence for LOW_WEIGHT
    min_confidence_low: float = 0.35
    # Minimum confidence for HIGH_WEIGHT
    min_confidence_high: float = 0.55
    # Maximum residual std for correction (above → shadow_only)
    max_residual_std: float = 180.0
    # Minimum observed hours for any correction
    min_observed_hours: int = 3
    # Fusion weights per decision
    fusion_weight_disabled: float = 0.0
    fusion_weight_shadow: float = 0.0
    fusion_weight_low: float = 0.12
    fusion_weight_high: float = 0.22
    # Negative price risk: if bias is negative and da_anchor < 0
    negative_risk_weight: float = 0.08  # override to this if negative risk detected


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    policy_decision: PolicyDecision
    fusion_weight: float
    shadow_only_flag: bool
    reason: str


def evaluate_policy(
    state: IntradayResidualState,
    mode: PredictionMode,
    config: Optional[PolicyConfig] = None,
    has_negative_risk: bool = False,
) -> PolicyResult:
    """Evaluate the intraday tracker policy.

    Parameters
    ----------
    state : IntradayResidualState
        Residual state from compute_intraday_residual_state.
    mode : PredictionMode
        Current prediction mode (FULL_DAY or INTRADAY).
    config : PolicyConfig, optional
        Policy configuration. Uses defaults if None.
    has_negative_risk : bool
        Whether negative price risk is detected (da_anchor < 0 for target hours).

    Returns
    -------
    PolicyResult with decision, fusion_weight, shadow flag, and reason.
    """
    if config is None:
        config = PolicyConfig()

    # Rule 1: Mode must be INTRADAY
    if mode != PredictionMode.INTRADAY:
        return PolicyResult(
            policy_decision=PolicyDecision.DISABLED,
            fusion_weight=config.fusion_weight_disabled,
            shadow_only_flag=False,
            reason="mode_not_intraday",
        )

    # Rule 2: Not enough observed hours → disabled
    if state.n_observed < config.min_observed_hours:
        return PolicyResult(
            policy_decision=PolicyDecision.DISABLED,
            fusion_weight=config.fusion_weight_disabled,
            shadow_only_flag=False,
            reason=f"n_observed_{state.n_observed}_below_min_{config.min_observed_hours}",
        )

    # Rule 3: Cutoff too low → disabled or shadow
    if state.cutoff_hour < config.min_cutoff_for_shadow:
        return PolicyResult(
            policy_decision=PolicyDecision.DISABLED,
            fusion_weight=config.fusion_weight_disabled,
            shadow_only_flag=False,
            reason=f"cutoff_{state.cutoff_hour}_below_min_shadow_{config.min_cutoff_for_shadow}",
        )

    if state.cutoff_hour < config.min_cutoff_for_correction:
        return PolicyResult(
            policy_decision=PolicyDecision.SHADOW_ONLY,
            fusion_weight=config.fusion_weight_shadow,
            shadow_only_flag=True,
            reason=f"cutoff_{state.cutoff_hour}_below_min_correction_{config.min_cutoff_for_correction}",
        )

    # Rule 4: Low confidence → shadow
    if state.confidence < config.min_confidence_low:
        return PolicyResult(
            policy_decision=PolicyDecision.SHADOW_ONLY,
            fusion_weight=config.fusion_weight_shadow,
            shadow_only_flag=True,
            reason=f"confidence_{state.confidence:.3f}_below_min_{config.min_confidence_low}",
        )

    # Rule 5: High residual std → shadow
    if state.residual_std_today > config.max_residual_std:
        return PolicyResult(
            policy_decision=PolicyDecision.SHADOW_ONLY,
            fusion_weight=config.fusion_weight_shadow,
            shadow_only_flag=True,
            reason=f"residual_std_{state.residual_std_today:.1f}_above_max_{config.max_residual_std}",
        )

    # Rule 6: Negative price risk → low weight
    if has_negative_risk:
        return PolicyResult(
            policy_decision=PolicyDecision.LOW_WEIGHT,
            fusion_weight=config.negative_risk_weight,
            shadow_only_flag=False,
            reason="negative_price_risk",
        )

    # Rule 7: Determine weight based on confidence and cutoff
    if state.confidence >= config.min_confidence_high and state.cutoff_hour >= 14:
        return PolicyResult(
            policy_decision=PolicyDecision.HIGH_WEIGHT,
            fusion_weight=config.fusion_weight_high,
            shadow_only_flag=False,
            reason=f"high_confidence_{state.confidence:.3f}_cutoff_{state.cutoff_hour}",
        )

    # Default: LOW_WEIGHT
    return PolicyResult(
        policy_decision=PolicyDecision.LOW_WEIGHT,
        fusion_weight=config.fusion_weight_low,
        shadow_only_flag=False,
        reason=f"default_low_weight_confidence_{state.confidence:.3f}",
    )
