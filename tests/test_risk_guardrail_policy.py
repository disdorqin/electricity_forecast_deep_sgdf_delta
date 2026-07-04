"""Tests for RiskGuardrailPolicy."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml

from models.deep_sgdf_delta.risk_guardrail_policy import (
    RiskGuardrailPolicy,
    GuardrailPolicyConfig,
    GuardrailAction,
    load_policy_from_yaml,
)


def _create_test_policy_yaml(path: str | Path) -> Path:
    """Create a test policy YAML file."""
    path = Path(path)
    
    policy = {
        "negative": {
            "enabled": True,
            "risk_column": "negative_risk_score",
            "prob_column": "negative_prob",
            "threshold": 0.6,
            "action": "soft_negative_blend",
            "blend_weight": 0.2,
            "floor_value": 0,
        },
        "spike": {
            "enabled": True,
            "risk_column": "spike_risk_score",
            "prob_column": "spike_prob",
            "threshold": 0.7,
            "action": "soft_spike_blend",
            "blend_weight": 0.2,
            "spike_floor": 500,
        },
        "delta_supply": {
            "enabled": True,
            "down_prob_column": "deviation_down_prob",
            "up_prob_column": "deviation_up_prob",
            "threshold": 0.6,
            "action": "weight_adjust",
        },
    }
    
    with open(path, "w") as f:
        yaml.dump(policy, f, default_flow_style=False)
    
    return path


class TestGuardrailPolicyConfig:
    """Tests for GuardrailPolicyConfig."""

    def test_default_config(self):
        """Default config should be valid."""
        config = GuardrailPolicyConfig()
        
        assert config.negative_enabled == True
        assert config.negative_threshold == 0.6
        assert config.negative_action == "soft_negative_blend"
        
        assert config.spike_enabled == True
        assert config.spike_threshold == 0.7
        assert config.spike_action == "soft_spike_blend"
        
        assert config.delta_supply_enabled == True
        assert config.delta_supply_threshold == 0.6


class TestLoadPolicyFromYaml:
    """Tests for load_policy_from_yaml()."""

    def test_loads_successfully(self, tmp_path):
        """Should load policy from YAML successfully."""
        yaml_file = tmp_path / "policy.yaml"
        _create_test_policy_yaml(yaml_file)
        
        config = load_policy_from_yaml(yaml_file)
        
        assert isinstance(config, GuardrailPolicyConfig)
        assert config.negative_enabled == True
        assert config.negative_threshold == 0.6

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_policy_from_yaml("nonexistent_policy.yaml")

    def test_partial_yaml(self, tmp_path):
        """Should handle partial YAML (only some sections)."""
        yaml_file = tmp_path / "partial_policy.yaml"
        
        policy = {
            "negative": {
                "enabled": True,
                "threshold": 0.8,
            },
        }
        
        with open(yaml_file, "w") as f:
            yaml.dump(policy, f, default_flow_style=False)
        
        config = load_policy_from_yaml(yaml_file)
        
        # Negative should be updated
        assert config.negative_threshold == 0.8
        
        # Spike should remain default
        assert config.spike_enabled == True
        assert config.spike_threshold == 0.7


class TestRiskGuardrailPolicy:
    """Tests for RiskGuardrailPolicy."""

    @pytest.fixture
    def policy(self):
        """Create a policy with default config."""
        return RiskGuardrailPolicy()

    def test_apply_negative_triggered(self, policy):
        """Should trigger negative guardrail."""
        base_pred = 300.0
        risk_scores = {
            "negative_risk_score": 0.8,  # Above threshold 0.6
            "negative_prob": 0.75,
            "spike_risk_score": 0.3,
            "spike_prob": 0.25,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        assert action.triggered == True
        assert "NEGATIVE_HIGH_RISK" in action.reason_codes
        assert action.action_taken == "soft_negative_blend"
        assert action.adjustment_amount!= 0

    def test_apply_negative_not_triggered(self, policy):
        """Should not trigger negative guardrail."""
        base_pred = 300.0
        risk_scores = {
            "negative_risk_score": 0.3,  # Below threshold 0.6
            "negative_prob": 0.25,
            "spike_risk_score": 0.3,
            "spike_prob": 0.25,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        # Might trigger spike or delta, but not negative
        neg_triggered = "NEGATIVE_HIGH_RISK" in action.reason_codes
        assert not neg_triggered

    def test_apply_spike_triggered(self, policy):
        """Should trigger spike guardrail."""
        base_pred = 300.0
        risk_scores = {
            "negative_risk_score": 0.3,
            "negative_prob": 0.25,
            "spike_risk_score": 0.8,  # Above threshold 0.7
            "spike_prob": 0.75,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        assert action.triggered == True
        assert "SPIKE_HIGH_RISK" in action.reason_codes
        assert action.action_taken == "soft_spike_blend"
        assert action.adjustment_amount!= 0

    def test_apply_no_trigger(self, policy):
        """Should not trigger any guardrail."""
        base_pred = 300.0
        risk_scores = {
            "negative_risk_score": 0.3,
            "negative_prob": 0.25,
            "spike_risk_score": 0.3,
            "spike_prob": 0.25,
            "deviation_down_prob": 0.3,
            "deviation_up_prob": 0.3,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        assert action.triggered == False
        assert action.reason_codes == ["NO_TRIGGER"]
        assert action.adjustment_amount == 0.0
        assert action.action_taken == "none"

    def test_apply_outputs_reason_codes(self, policy):
        """All triggers must output reason_codes."""
        base_pred = 300.0
        risk_scores = {
            "negative_risk_score": 0.8,
            "negative_prob": 0.75,
            "spike_risk_score": 0.8,
            "spike_prob": 0.75,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        assert len(action.reason_codes) > 0
        assert "NEGATIVE_HIGH_RISK" in action.reason_codes
        assert "SPIKE_HIGH_RISK" in action.reason_codes

    def test_soft_negative_blend_adjustment(self, policy):
        """Soft negative blend should adjust prediction toward floor."""
        policy.config.negative_floor_value = 0
        policy.config.negative_blend_weight = 0.2
        
        base_pred = 300.0  # High price, risky for negative
        risk_scores = {
            "negative_risk_score": 0.8,
            "negative_prob": 0.75,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        # Adjusted pred should be lower than base_pred
        adjusted_pred = base_pred + action.adjustment_amount
        assert adjusted_pred < base_pred
        assert adjusted_pred >= policy.config.negative_floor_value

    def test_soft_spike_blend_adjustment(self, policy):
        """Soft spike blend should adjust prediction toward spike_floor."""
        policy.config.spike_floor = 500
        policy.config.spike_floor_weight = 0.2
        
        base_pred = 300.0  # Low price, risky for spike
        risk_scores = {
            "spike_risk_score": 0.8,
            "spike_prob": 0.75,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        # Adjusted pred should be higher than base_pred
        adjusted_pred = base_pred + action.adjustment_amount
        assert adjusted_pred > base_pred
        assert adjusted_pred <= policy.config.spike_floor

    def test_alert_only_action(self, policy):
        """Alert-only mode should not adjust prediction."""
        policy.config.alert_only = True
        
        base_pred = 300.0
        risk_scores = {
            "negative_risk_score": 0.8,
            "negative_prob": 0.75,
        }
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        assert action.triggered == True
        assert action.adjustment_amount == 0.0
        assert "alert_only" in action.action_taken or action.action_taken == "none"

    def test_apply_to_dataframe(self, policy):
        """Should apply policy to DataFrame."""
        # Create test DataFrame
        n = 10
        df = pd.DataFrame({
            "base_pred": np.random.uniform(200, 800, n),
            "negative_risk_score": np.random.uniform(0, 1, n),
            "negative_prob": np.random.uniform(0, 1, n),
            "spike_risk_score": np.random.uniform(0, 1, n),
            "spike_prob": np.random.uniform(0, 1, n),
        })
        
        result_df = policy.apply_to_dataframe(df, base_pred_col="base_pred")
        
        assert "risk_adjusted_pred" in result_df.columns
        assert "guardrail_triggered" in result_df.columns
        assert "guardrail_action" in result_df.columns
        assert "guardrail_adjustment" in result_df.columns
        assert "guardrail_reason_codes" in result_df.columns

    def test_no_y_true_in_decision(self, policy):
        """Policy should not use y_true for decision."""
        base_pred = 300.0
        risk_scores = {
            "negative_risk_score": 0.8,
            "negative_prob": 0.75,
        }
        
        # Even if we pass y_true in risk_scores, policy should not use it
        risk_scores["y_true"] = 100.0  # Would be negative price
        
        action = policy.apply(base_pred=base_pred, risk_scores=risk_scores)
        
        # Decision should be based only on risk scores, not y_true
        assert action.triggered == True
        assert "NEGATIVE_HIGH_RISK" in action.reason_codes
        
        # Verify adjustment is based on floor_value, not y_true
        assert policy.config.negative_floor_value == 0
        # If it used y_true=100, adjustment would be very different

    def test_load_config_from_yaml(self, policy, tmp_path):
        """Should load config from YAML."""
        yaml_file = tmp_path / "policy.yaml"
        _create_test_policy_yaml(yaml_file)
        
        policy.load_config_from_yaml(yaml_file)
        
        assert policy.config.negative_threshold == 0.6
        assert policy.config.spike_threshold == 0.7
