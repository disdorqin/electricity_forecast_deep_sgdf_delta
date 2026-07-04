"""Risk Shadow Replay Engine for Ledger-1.

Runs shadow replay of risk-aware guardrail on historical data.

Usage:
    python scripts/run_risk_guardrail_shadow_replay.py \
        --risk-pack reports/local/risk_modules/risk_feature_pack_2026_01_05/risk_feature_pack.csv \
        --risk-pack-manifest reports/local/risk_modules/risk_feature_pack_2026_01_05/manifest.json \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --base-mode da_anchor \
        --target-months 2026-01,2026-02,2026-03,2026-04,2026-05 \
        --out-dir reports/local/ledger_1/shadow_replay_2026_01_05

Policy Sweep:
    negative_threshold: 0.4, 0.5, 0.6, 0.7
    spike_threshold: 0.4, 0.5, 0.6, 0.7
    blend_weight: 0.05, 0.1, 0.2

Outputs:
    input_diagnostics.json
    shadow_metrics.csv
    monthly_metrics.csv
    period_metrics.csv
    bucket_metrics.csv
    risk_trigger_metrics.csv
    decision_log.csv
    policy_sweep.csv
    champion_policy.json
    shadow_replay_report.md
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import itertools
from datetime import datetime

from models.deep_sgdf_delta.base_prediction_adapter import BasePredictionAdapter, BasePredictionLoadResult
from models.deep_sgdf_delta.risk_pack_loader import RiskPackLoader, RiskPackLoadResult
from models.deep_sgdf_delta.risk_guardrail_policy import RiskGuardrailPolicy, GuardrailPolicyConfig
from models.deep_sgdf_delta.metrics import smape_floor50 as canonical_smape_floor50


@dataclass
class ShadowReplayConfig:
    """Configuration for shadow replay."""
    risk_pack_path: str | Path
    manifest_path: Optional[str | Path] = None
    data_path: Optional[str | Path] = None
    base_prediction_file: Optional[str | Path] = None
    base_mode: str = "da_anchor"  # "da_anchor" or "base_prediction_file"
    target_months: List[str] = field(default_factory=list)
    out_dir: str | Path = "reports/local/ledger_1/shadow_replay"
    
    # Policy sweep grid
    negative_thresholds: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7])
    spike_thresholds: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7])
    blend_weights: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    
    # Guardrail config
    negative_floor: float = 0.0
    spike_floor: float = 500.0
    
    # Evaluation
    metrics: List[str] = field(default_factory=lambda: ["sMAPE_floor50", "sMAPE", "MAE", "RMSE"])


@dataclass
class ShadowReplayResult:
    """Result from shadow replay."""
    config: ShadowReplayConfig
    base_pred_result: BasePredictionLoadResult
    risk_pack_result: RiskPackLoadResult
    policy_sweep_df: pd.DataFrame
    champion_policy: Dict[str, Any]
    decision_log: pd.DataFrame
    metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


def run_shadow_replay(config: ShadowReplayConfig) -> ShadowReplayResult:
    """Run shadow replay with policy sweep.

    Args:
        config: ShadowReplayConfig.

    Returns:
        ShadowReplayResult.
    """
    warnings = []
    
    # Step 1: Load base predictions
    adapter = BasePredictionAdapter()
    
    if config.base_mode == "da_anchor":
        if config.data_path is None:
            raise ValueError("data_path required for da_anchor mode")
        
        base_result = adapter.load(
            data_path=config.data_path,
            target_months=config.target_months,
        )
        warnings.extend(base_result.warnings)
    
    elif config.base_mode == "base_prediction_file":
        if config.base_prediction_file is None:
            raise ValueError("base_prediction_file required for base_prediction_file mode")
        
        base_result = adapter.load(
            base_prediction_file=config.base_prediction_file,
        )
    
    else:
        raise ValueError(f"Unknown base_mode: {config.base_mode}")
    
    # Step 2: Load risk pack
    loader = RiskPackLoader()
    risk_result = loader.load(
        risk_pack_path=config.risk_pack_path,
        manifest_path=config.manifest_path,
        online_mode=False,  # Eval mode (we have y_true)
    )
    
    if risk_result.status!= "SUCCESS":
        raise RuntimeError(f"Failed to load risk pack: {risk_result.error_message}")
    
    # Step 3: Merge base predictions and risk pack
    merged_df = _merge_base_and_risk(base_result.df, risk_result.df, out_dir=Path(config.out_dir))
    
    # Step 4: Policy sweep
    policy_sweep_results = []
    
    for neg_thresh, spike_thresh, blend_w in itertools.product(
        config.negative_thresholds,
        config.spike_thresholds,
        config.blend_weights,
    ):
        # Create policy config
        policy_config = GuardrailPolicyConfig(
            negative_threshold=neg_thresh,
            spike_threshold=spike_thresh,
            negative_blend_weight=blend_w,
            spike_blend_weight=blend_w,
            negative_floor_value=config.negative_floor,
            spike_floor=config.spike_floor,
        )
        
        # Apply policy
        policy = RiskGuardrailPolicy(config=policy_config)
        result_df = policy.apply_to_dataframe(
            df=merged_df,
            base_pred_col="base_pred",
        )
        
        # Evaluate
        metrics = _evaluate_guardrail(result_df)
        
        # Record
        policy_sweep_results.append({
            "negative_threshold": neg_thresh,
            "spike_threshold": spike_thresh,
            "blend_weight": blend_w,
            **metrics,
        })
    
    policy_sweep_df = pd.DataFrame(policy_sweep_results)
    
    # Step 5: Select champion policy
    champion_policy = _select_champion_policy(policy_sweep_df)
    
    # Step 6: Run champion policy and generate decision log
    champion_config = _policy_dict_to_config(champion_policy)
    champion_policy_obj = RiskGuardrailPolicy(config=champion_config)
    
    decision_log = champion_policy_obj.apply_to_dataframe(
        df=merged_df,
        base_pred_col="base_pred",
    )
    
    # Step 7: Calculate final metrics
    final_metrics = _evaluate_guardrail(decision_log)
    
    # Add warnings
    if base_result.source == "DA_ANCHOR_BASELINE":
        warnings.append(
            "This is a guardrail sensitivity test using DA anchor baseline, "
            "NOT a production baseline. "
            "Results may not generalize to production."
        )
    
    return ShadowReplayResult(
        config=config,
        base_pred_result=base_result,
        risk_pack_result=risk_result,
        policy_sweep_df=policy_sweep_df,
        champion_policy=champion_policy,
        decision_log=decision_log,
        metrics=final_metrics,
        warnings=warnings,
    )


def _merge_base_and_risk(
    base_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge base predictions and risk pack.

    Args:
        base_df: DataFrame with base predictions.
        risk_df: DataFrame with risk scores.
        out_dir: Output directory for diagnostics (optional).

    Returns:
        Merged DataFrame.
    """
    # Ensure key columns have same dtype
    key_cols = ["business_day", "hour_business", "target_month"]
    
    merged = base_df.copy()
    risk_subset = risk_df.copy()
    
    # Convert business_day to datetime64[ns] for both
    if "business_day" in merged.columns and "business_day" in risk_subset.columns:
        merged["business_day"] = pd.to_datetime(merged["business_day"])
        risk_subset["business_day"] = pd.to_datetime(risk_subset["business_day"])
    
    # Select risk score columns
    risk_cols = [
        "negative_prob", "negative_risk_score",
        "spike_prob", "spike_risk_score",
        "deviation_down_prob", "deviation_up_prob", "deviation_risk_score",
    ]
    
    available_risk_cols = [col for col in risk_cols if col in risk_subset.columns]
    
    # Also include y_true if available (for eval)
    if "y_true" in risk_subset.columns:
        available_risk_cols.append("y_true")
    
    # Merge
    merged = merged.merge(
        risk_subset[key_cols + available_risk_cols],
        on=key_cols,
        how="left",
    )
    
    # Check for missing risk scores
    for col in available_risk_cols:
        if col == "y_true":
            continue
        nan_count = merged[col].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} rows have NaN in {col}")
    
    # Generate input diagnostics
    diagnostics = _generate_input_diagnostics(merged, available_risk_cols)
    
    # Save diagnostics if out_dir provided
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "input_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)
        print(f"Input diagnostics saved to {out_dir / 'input_diagnostics.json'}")
    
    return merged


def _generate_input_diagnostics(
    merged_df: pd.DataFrame,
    risk_cols: List[str],
) -> Dict[str, Any]:
    """Generate input diagnostics after merge.

    Args:
        merged_df: Merged DataFrame.
        risk_cols: List of risk columns.

    Returns:
        Dict with diagnostics.
    """
    diagnostics = {
        "n_rows": len(merged_df),
        "has_y_true": "y_true" in merged_df.columns,
        "y_true_non_null_count": 0,
        "base_pred_non_null_count": 0,
        "risk_cols_non_null_count": {},
    }
    
    # y_true diagnostics
    if "y_true" in merged_df.columns:
        diagnostics["y_true_non_null_count"] = int(merged_df["y_true"].notna().sum())
    
    # base_pred diagnostics
    if "base_pred" in merged_df.columns:
        diagnostics["base_pred_non_null_count"] = int(merged_df["base_pred"].notna().sum())
    
    # Risk columns diagnostics
    for col in risk_cols:
        if col in merged_df.columns and col != "y_true":
            diagnostics["risk_cols_non_null_count"][col] = int(merged_df[col].notna().sum())
    
    return diagnostics


def _evaluate_guardrail(df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate guardrail performance.

    Args:
        df: DataFrame with base_pred, risk_adjusted_pred, and y_true.

    Returns:
        Dict of metrics with fixed schema. Always returns all required columns.
    """
    # Define the full metric schema with default values (NaN)
    metrics = {
        "base_sMAPE_floor50": np.nan,
        "adjusted_sMAPE_floor50": np.nan,
        "sMAPE_floor50_improvement": np.nan,
        "base_sMAPE": np.nan,
        "adjusted_sMAPE": np.nan,
        "sMAPE_improvement": np.nan,
        "base_MAE": np.nan,
        "adjusted_MAE": np.nan,
        "MAE_improvement": np.nan,
        "base_RMSE": np.nan,
        "adjusted_RMSE": np.nan,
        "RMSE_improvement": np.nan,
        "trigger_rate": np.nan,
        "evaluation_status": "MISSING_Y_TRUE",
    }
    
    # Check if y_true is available
    if "y_true" not in df.columns:
        return metrics
    
    y_true = df["y_true"].values
    base_pred = df["base_pred"].values
    risk_adjusted = df["risk_adjusted_pred"].values
    
    # Check for non-null y_true
    valid_mask = np.isfinite(y_true) & np.isfinite(base_pred) & np.isfinite(risk_adjusted)
    
    if not np.any(valid_mask):
        return metrics
    
    y_true_valid = y_true[valid_mask]
    base_pred_valid = base_pred[valid_mask]
    risk_adjusted_valid = risk_adjusted[valid_mask]
    
    # sMAPE_floor50 (using canonical implementation from metrics.py)
    base_smape = canonical_smape_floor50(y_true_valid, base_pred_valid)
    adjusted_smape = canonical_smape_floor50(y_true_valid, risk_adjusted_valid)
    
    metrics["base_sMAPE_floor50"] = base_smape
    metrics["adjusted_sMAPE_floor50"] = adjusted_smape
    metrics["sMAPE_floor50_improvement"] = base_smape - adjusted_smape
    
    # sMAPE
    base_smape_full = _calc_smape(y_true_valid, base_pred_valid)
    adjusted_smape_full = _calc_smape(y_true_valid, risk_adjusted_valid)
    
    metrics["base_sMAPE"] = base_smape_full
    metrics["adjusted_sMAPE"] = adjusted_smape_full
    metrics["sMAPE_improvement"] = base_smape_full - adjusted_smape_full
    
    # MAE
    base_mae = np.mean(np.abs(y_true_valid - base_pred_valid))
    adjusted_mae = np.mean(np.abs(y_true_valid - risk_adjusted_valid))
    
    metrics["base_MAE"] = base_mae
    metrics["adjusted_MAE"] = adjusted_mae
    metrics["MAE_improvement"] = base_mae - adjusted_mae
    
    # RMSE
    base_rmse = np.sqrt(np.mean((y_true_valid - base_pred_valid) ** 2))
    adjusted_rmse = np.sqrt(np.mean((y_true_valid - risk_adjusted_valid) ** 2))
    
    metrics["base_RMSE"] = base_rmse
    metrics["adjusted_RMSE"] = adjusted_rmse
    metrics["RMSE_improvement"] = base_rmse - adjusted_rmse
    
    # Trigger rate
    if "guardrail_triggered" in df.columns:
        trigger_rate = df["guardrail_triggered"].mean()
        metrics["trigger_rate"] = trigger_rate
    
    metrics["evaluation_status"] = "SUCCESS"
    
    return metrics


def _calc_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate sMAPE.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        sMAPE.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2
    
    # Avoid division by zero
    mask = denominator > 0
    smape = np.mean(numerator[mask] / denominator[mask])
    
    return smape * 100  # Convert to percentage


def _select_champion_policy(policy_sweep_df: pd.DataFrame) -> Dict[str, Any]:
    """Select champion policy from sweep results.

    Args:
        policy_sweep_df: DataFrame with sweep results.

    Returns:
        Dict with champion policy config and metrics.
    """
    # Check if sMAPE_floor50_improvement column exists and has non-NaN values
    if "sMAPE_floor50_improvement" not in policy_sweep_df.columns:
        # Fallback to alert_only policy
        return {
            "negative_threshold": 0.5,
            "spike_threshold": 0.5,
            "blend_weight": 0.0,
            "selection_status": "NO_VALID_EVAL_METRIC",
            "metrics": {
                "negative_threshold": 0.5,
                "spike_threshold": 0.5,
                "blend_weight": 0.0,
            },
        }
    
    # Filter out NaN improvements
    valid_df = policy_sweep_df.dropna(subset=["sMAPE_floor50_improvement"])
    
    if len(valid_df) == 0:
        # Fallback to alert_only policy
        return {
            "negative_threshold": 0.5,
            "spike_threshold": 0.5,
            "blend_weight": 0.0,
            "selection_status": "NO_VALID_EVAL_METRIC",
            "metrics": {
                "negative_threshold": 0.5,
                "spike_threshold": 0.5,
                "blend_weight": 0.0,
            },
        }
    
    # Sort by sMAPE_floor50_improvement (descending)
    sorted_df = valid_df.sort_values(
        "sMAPE_floor50_improvement",
        ascending=False,
    )
    
    # Select top policy
    top = sorted_df.iloc[0]
    
    champion = {
        "negative_threshold": top["negative_threshold"],
        "spike_threshold": top["spike_threshold"],
        "blend_weight": top["blend_weight"],
        "selection_status": "SUCCESS",
        "metrics": top.to_dict(),
    }
    
    return champion


def _policy_dict_to_config(policy_dict: Dict[str, Any]) -> GuardrailPolicyConfig:
    """Convert policy dict to GuardrailPolicyConfig.

    Args:
        policy_dict: Dict with policy config.

    Returns:
        GuardrailPolicyConfig.
    """
    return GuardrailPolicyConfig(
        negative_threshold=policy_dict["negative_threshold"],
        spike_threshold=policy_dict["spike_threshold"],
        negative_blend_weight=policy_dict["blend_weight"],
        spike_blend_weight=policy_dict["blend_weight"],
    )


class RiskShadowReplay:
    """Shadow replay engine for risk-aware guardrail.

    Usage:
        engine = RiskShadowReplay()
        result = engine.run(config)
        
        # Export results
        engine.export_results(result, config.out_dir)
    """

    def __init__(self):
        self.last_result: Optional[ShadowReplayResult] = None

    def run(self, config: ShadowReplayConfig) -> ShadowReplayResult:
        """Run shadow replay.

        Args:
            config: ShadowReplayConfig.

        Returns:
            ShadowReplayResult.
        """
        result = run_shadow_replay(config)
        self.last_result = result
        return result

    def export_results(
        self,
        result: Optional[ShadowReplayResult] = None,
        out_dir: Optional[str | Path] = None,
    ):
        """Export shadow replay results.

        Args:
            result: ShadowReplayResult. If None, uses last_result.
            out_dir: Output directory. If None, uses config.out_dir.
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            raise RuntimeError("No result to export. Call run() first.")
        
        if out_dir is None:
            out_dir = result.config.out_dir
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Export policy sweep
        result.policy_sweep_df.to_csv(out_dir / "policy_sweep.csv", index=False)
        
        # Export decision log
        result.decision_log.to_csv(out_dir / "decision_log.csv", index=False)
        
        # Export champion policy
        with open(out_dir / "champion_policy.json", "w") as f:
            json.dump(result.champion_policy, f, indent=2)
        
        # Export metrics
        metrics_df = pd.DataFrame([result.metrics])
        metrics_df.to_csv(out_dir / "shadow_metrics.csv", index=False)
        
        # Export warnings
        if result.warnings:
            with open(out_dir / "warnings.txt", "w") as f:
                for warning in result.warnings:
                    f.write(warning + "\n")
        
        print(f"Results exported to {out_dir}")
