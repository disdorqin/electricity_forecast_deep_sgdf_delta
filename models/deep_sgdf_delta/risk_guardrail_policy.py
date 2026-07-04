"""Risk Guardrail Policy for Ledger-1 Shadow Replay.

Defines policy config for risk-aware guardrail.

Supported actions:
1. none: No action.
2. alert_only: Only alert, no correction.
3. soft_negative_blend: Blend prediction with negative floor.
4. soft_spike_blend: Blend prediction with spike floor.
5. weight_adjust: Adjust fusion weights (not implemented yet).

Policy config YAML format:
  negative:
    enabled: true
    risk_column: negative_risk_score
    prob_column: negative_prob
    threshold: 0.6
    action: soft_negative_blend
    blend_weight: 0.2
    floor_value: 0

  spike:
    enabled: true
    risk_column: spike_risk_score
    prob_column: spike_prob
    threshold: 0.7
    action: soft_spike_blend
    blend_weight: 0.2
    spike_floor: 500

  delta_supply:
    enabled: true
    down_prob_column: deviation_down_prob
    up_prob_column: deviation_up_prob
    threshold: 0.6
    action: weight_adjust

Reason codes:
  NEGATIVE_HIGH_RISK: Negative risk triggered.
  SPIKE_HIGH_RISK: Spike risk triggered.
  DELTA_SUPPLY_DOWN_RISK: Delta supply down risk triggered.
  DELTA_SUPPLY_UP_RISK: Delta supply up risk triggered.
  NO_TRIGGER: No risk triggered.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class GuardrailAction:
    """Result of applying guardrail policy."""
    action_taken: str  # "none", "alert_only", "soft_negative_blend", "soft_spike_blend", "weight_adjust"
    reason_codes: List[str]
    adjustment_amount: float  # Amount of adjustment (can be 0)
    triggered: bool  # Whether any risk triggered
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailPolicyConfig:
    """Configuration for guardrail policy."""
    # Negative risk config
    negative_enabled: bool = True
    negative_risk_column: str = "negative_risk_score"
    negative_prob_column: str = "negative_prob"
    negative_threshold: float = 0.6
    negative_action: str = "soft_negative_blend"
    negative_blend_weight: float = 0.2
    negative_floor_value: float = 0

    # Spike risk config
    spike_enabled: bool = True
    spike_risk_column: str = "spike_risk_score"
    spike_prob_column: str = "spike_prob"
    spike_threshold: float = 0.7
    spike_action: str = "soft_spike_blend"
    spike_blend_weight: float = 0.2
    spike_floor: float = 500

    # Delta supply risk config
    delta_supply_enabled: bool = True
    delta_supply_down_prob_column: str = "deviation_down_prob"
    delta_supply_up_prob_column: str = "deviation_up_prob"
    delta_supply_threshold: float = 0.6
    delta_supply_action: str = "weight_adjust"

    # General config
    alert_only: bool = False  # If True, only alert, no correction


def load_policy_from_yaml(yaml_path: str | Path) -> GuardrailPolicyConfig:
    """Load guardrail policy from YAML file.

    Args:
        yaml_path: Path to YAML config file.

    Returns:
        GuardrailPolicyConfig.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Policy YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Map YAML to config
    config = GuardrailPolicyConfig()

    # Negative config
    if "negative" in config_dict:
        neg = config_dict["negative"]
        config.negative_enabled = neg.get("enabled", True)
        config.negative_risk_column = neg.get("risk_column", "negative_risk_score")
        config.negative_prob_column = neg.get("prob_column", "negative_prob")
        config.negative_threshold = neg.get("threshold", 0.6)
        config.negative_action = neg.get("action", "soft_negative_blend")
        config.negative_blend_weight = neg.get("blend_weight", 0.2)
        config.negative_floor_value = neg.get("floor_value", 0)

    # Spike config
    if "spike" in config_dict:
        spike = config_dict["spike"]
        config.spike_enabled = spike.get("enabled", True)
        config.spike_risk_column = spike.get("risk_column", "spike_risk_score")
        config.spike_prob_column = spike.get("prob_column", "spike_prob")
        config.spike_threshold = spike.get("threshold", 0.7)
        config.spike_action = spike.get("action", "soft_spike_blend")
        config.spike_blend_weight = spike.get("blend_weight", 0.2)
        config.spike_floor = spike.get("spike_floor", 500)

    # Delta supply config
    if "delta_supply" in config_dict:
        delta = config_dict["delta_supply"]
        config.delta_supply_enabled = delta.get("enabled", True)
        config.delta_supply_down_prob_column = delta.get("down_prob_column", "deviation_down_prob")
        config.delta_supply_up_prob_column = delta.get("up_prob_column", "deviation_up_prob")
        config.delta_supply_threshold = delta.get("threshold", 0.6)
        config.delta_supply_action = delta.get("action", "weight_adjust")

    return config


class RiskGuardrailPolicy:
    """Apply risk-aware guardrail policy to predictions.

    Usage:
        policy = RiskGuardrailPolicy()
        policy.load_config(config)

        result = policy.apply(
            base_pred=100.0,
            risk_scores={
                "negative_risk_score": 0.8,
                "negative_prob": 0.75,
                "spike_risk_score": 0.3,
                "spike_prob": 0.25,
            },
        )

        # result.adjustment_amount is the correction to apply
        adjusted_pred = base_pred + result.adjustment_amount
    """

    def __init__(self, config: Optional[GuardrailPolicyConfig] = None):
        self.config = config or GuardrailPolicyConfig()
        self.last_action: Optional[GuardrailAction] = None

    def load_config(self, config: GuardrailPolicyConfig):
        """Load policy config.

        Args:
            config: GuardrailPolicyConfig.
        """
        self.config = config

    def load_config_from_yaml(self, yaml_path: str | Path):
        """Load policy config from YAML file.

        Args:
            yaml_path: Path to YAML config file.
        """
        self.config = load_policy_from_yaml(yaml_path)

    def apply(
        self,
        base_pred: float,
        risk_scores: Dict[str, float],
        hour: Optional[int] = None,
        verbose: bool = False,
    ) -> GuardrailAction:
        """Apply guardrail policy to a single prediction.

        Args:
            base_pred: Base prediction (e.g., DA anchor, SGDFNet).
            risk_scores: Dict of risk scores/probabilities.
            hour: Optional hour of day (for debugging).
            verbose: If True, print details.

        Returns:
            GuardrailAction with action taken and reason codes.
        """
        reason_codes = []
        adjustment = 0.0
        triggered = False
        details = {}

        # Check negative risk
        if self.config.negative_enabled:
            neg_risk = risk_scores.get(self.config.negative_risk_column, 0.0)
            neg_prob = risk_scores.get(self.config.negative_prob_column, 0.0)

            if neg_risk >= self.config.negative_threshold or neg_prob >= self.config.negative_threshold:
                reason_codes.append("NEGATIVE_HIGH_RISK")
                triggered = True
                details["negative_risk"] = neg_risk
                details["negative_prob"] = neg_prob
                details["negative_threshold"] = self.config.negative_threshold

                if self.config.alert_only:
                    details["action"] = "alert_only"
                elif self.config.negative_action == "soft_negative_blend":
                    # Blend base_pred with floor_value
                    blended = (1 - self.config.negative_blend_weight) * base_pred + \
                              self.config.negative_blend_weight * self.config.negative_floor_value
                    adjustment += (blended - base_pred)
                    details["action"] = "soft_negative_blend"
                    details["blend_weight"] = self.config.negative_blend_weight
                    details["floor_value"] = self.config.negative_floor_value
                elif self.config.negative_action == "none":
                    details["action"] = "none"

        # Check spike risk
        if self.config.spike_enabled:
            spike_risk = risk_scores.get(self.config.spike_risk_column, 0.0)
            spike_prob = risk_scores.get(self.config.spike_prob_column, 0.0)

            if spike_risk >= self.config.spike_threshold or spike_prob >= self.config.spike_threshold:
                reason_codes.append("SPIKE_HIGH_RISK")
                triggered = True
                details["spike_risk"] = spike_risk
                details["spike_prob"] = spike_prob
                details["spike_threshold"] = self.config.spike_threshold

                if self.config.alert_only:
                    details["action"] = "alert_only"
                elif self.config.spike_action == "soft_spike_blend":
                    # Blend base_pred with spike_floor
                    blended = (1 - self.config.spike_blend_weight) * base_pred + \
                              self.config.spike_blend_weight * self.config.spike_floor
                    adjustment += (blended - base_pred)
                    details["action"] = "soft_spike_blend"
                    details["blend_weight"] = self.config.spike_blend_weight
                    details["spike_floor"] = self.config.spike_floor
                elif self.config.spike_action == "none":
                    details["action"] = "none"

        # Check delta supply risk
        if self.config.delta_supply_enabled:
            down_prob = risk_scores.get(self.config.delta_supply_down_prob_column, 0.0)
            up_prob = risk_scores.get(self.config.delta_supply_up_prob_column, 0.0)

            if down_prob >= self.config.delta_supply_threshold:
                reason_codes.append("DELTA_SUPPLY_DOWN_RISK")
                triggered = True
                details["delta_supply_down_prob"] = down_prob
                details["delta_supply_threshold"] = self.config.delta_supply_threshold
                details["action"] = "weight_adjust (down)"

            if up_prob >= self.config.delta_supply_threshold:
                reason_codes.append("DELTA_SUPPLY_UP_RISK")
                triggered = True
                details["delta_supply_up_prob"] = up_prob
                details["delta_supply_threshold"] = self.config.delta_supply_threshold
                details["action"] = "weight_adjust (up)"

        # If no trigger, set reason code
        if not triggered:
            reason_codes.append("NO_TRIGGER")
            details["action"] = "none"

        # Create action result
        action = GuardrailAction(
            action_taken=details.get("action", "none"),
            reason_codes=reason_codes,
            adjustment_amount=adjustment,
            triggered=triggered,
            details=details,
        )

        self.last_action = action

        if verbose:
            print(f"Hour {hour}: base_pred={base_pred:.2f}, adjustment={adjustment:.2f}")
            print(f"  Reason codes: {reason_codes}")
            print(f"  Details: {details}")

        return action

    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        base_pred_col: str = "base_pred",
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Apply guardrail policy to a DataFrame.

        Args:
            df: DataFrame with base_pred and risk score columns.
            base_pred_col: Name of base prediction column.
            verbose: If True, print progress.

        Returns:
            DataFrame with guardrail adjustments.
        """
        result_df = df.copy()

        # Collect risk score columns
        risk_score_cols = [
            self.config.negative_risk_column,
            self.config.negative_prob_column,
            self.config.spike_risk_column,
            self.config.spike_prob_column,
            self.config.delta_supply_down_prob_column,
            self.config.delta_supply_up_prob_column,
        ]

        # Filter to available columns
        available_cols = [col for col in risk_score_cols if col in result_df.columns]

        # Apply policy row by row
        actions = []
        adjusted_preds = []

        for idx, row in result_df.iterrows():
            base_pred = row[base_pred_col]

            # Extract risk scores
            risk_scores = {}
            for col in available_cols:
                risk_scores[col] = row[col]

            # Get hour for debugging
            hour = row.get("hour_business", None)

            # Apply policy
            action = self.apply(
                base_pred=base_pred,
                risk_scores=risk_scores,
                hour=hour,
                verbose=verbose,
            )

            actions.append(action)
            adjusted_preds.append(base_pred + action.adjustment_amount)

        # Add results to DataFrame
        result_df["risk_adjusted_pred"] = adjusted_preds
        result_df["guardrail_triggered"] = [a.triggered for a in actions]
        result_df["guardrail_action"] = [a.action_taken for a in actions]
        result_df["guardrail_adjustment"] = [a.adjustment_amount for a in actions]
        result_df["guardrail_reason_codes"] = [",".join(a.reason_codes) for a in actions]

        return result_df
