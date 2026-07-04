"""Risk Pack Loader for Ledger-1 Shadow Replay.

Loads and validates risk feature pack from RiskModules-2.5.

Validation checks:
1. risk_feature_version starts with "v1."
2. metric_alignment_status in [PASS, WARN] (FAIL is not allowed)
3. quality_gate_passed = true (if available)
4. No y_true in online mode
5. All probability columns in [0, 1] or NaN
6. module_status columns not all UNKNOWN
7. business_day + hour_business + target_month unique

Output:
  RiskPackLoadResult:
    df: Validated risk feature DataFrame
    manifest: Manifest from JSON
    warnings: List of warnings
    status: Load status
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class RiskPackLoadResult:
    """Result from loading risk feature pack."""
    df: pd.DataFrame
    manifest: dict
    warnings: List[str] = field(default_factory=list)
    status: str = "SUCCESS"
    error_message: Optional[str] = None


# Probability columns that must be in [0, 1] or NaN
PROBABILITY_COLUMNS = [
    "negative_prob",
    "negative_risk_score",
    "spike_prob",
    "spike_risk_score",
    "deviation_down_prob",
    "deviation_up_prob",
    "deviation_risk_score",
]

# Module status columns
MODULE_STATUS_COLUMNS = [
    "negative_module_status",
    "spike_module_status",
    "delta_supply_module_status",
]


def load_risk_pack(
    risk_pack_path: str | Path,
    manifest_path: Optional[str | Path] = None,
    online_mode: bool = True,
) -> RiskPackLoadResult:
    """Load and validate risk feature pack.

    Args:
        risk_pack_path: Path to risk_feature_pack.csv.
        manifest_path: Optional path to manifest.json.
                      If None, tries to find it in same directory.
        online_mode: If True, y_true is not allowed.

    Returns:
        RiskPackLoadResult with validated df and manifest.
    """
    risk_pack_path = Path(risk_pack_path)
    
    if not risk_pack_path.exists():
        return RiskPackLoadResult(
            df=pd.DataFrame(),
            manifest={},
            status="ERROR",
            error_message=f"Risk pack file not found: {risk_pack_path}",
        )
    
    # Load manifest
    if manifest_path is None:
        manifest_path = risk_pack_path.parent / "manifest.json"
    
    manifest = {}
    if Path(manifest_path).exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    
    # Load risk pack
    df = pd.read_csv(risk_pack_path)
    
    warnings = []
    status = "SUCCESS"
    error_message = None
    
    # Validation 1: Check risk_feature_version
    version = manifest.get("risk_feature_version", "")
    if not version.startswith("v1."):
        warnings.append(
            f"risk_feature_version '{version}' does not start with 'v1.'."
            f"Expected format: v1.x.x"
        )
    
    # Validation 2: Check metric_alignment_status
    metric_status = manifest.get("metric_alignment_status", "UNKNOWN")
    if metric_status == "FAIL":
        status = "ERROR"
        error_message = (
            f"metric_alignment_status = FAIL. "
            f"Risk pack cannot be used with FAIL status."
        )
        return RiskPackLoadResult(
            df=df,
            manifest=manifest,
            status=status,
            error_message=error_message,
        )
    elif metric_status == "WARN":
        warnings.append(
            "metric_alignment_status = WARN. "
            "Risk pack has metric alignment warnings."
        )
    
    # Validation 3: Check quality_gate_passed
    quality_gate = manifest.get("quality_gate_passed", None)
    if quality_gate is not None and not quality_gate:
        warnings.append(
            "quality_gate_passed = false. "
            "Risk pack did not pass quality gate."
        )
    
    # Validation 4: Check y_true in online mode
    if online_mode and "y_true" in df.columns:
        status = "ERROR"
        error_message = (
            "online_mode=True but y_true found in risk pack. "
            "y_true is not allowed in online mode."
        )
        return RiskPackLoadResult(
            df=df,
            manifest=manifest,
            status=status,
            error_message=error_message,
        )
    
    # Validation 5: Check probability columns in [0, 1] or NaN
    for col in PROBABILITY_COLUMNS:
        if col in df.columns:
            # Check values
            invalid_mask = ~df[col].between(0, 1) & df[col].notna()
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                warnings.append(
                    f"Column '{col}' has {invalid_count} values outside [0, 1]. "
                    f"Min: {df[col].min()}, Max: {df[col].max()}"
                )
    
    # Validation 6: Check module_status not all UNKNOWN
    for col in MODULE_STATUS_COLUMNS:
        if col in df.columns:
            if (df[col] == "UNKNOWN").all():
                warnings.append(
                    f"Column '{col}' is all UNKNOWN. "
                    f"Module status should not be all UNKNOWN."
                )
    
    # Validation 7: Check key uniqueness
    key_cols = ["business_day", "hour_business", "target_month"]
    if all(col in df.columns for col in key_cols):
        if df[key_cols].duplicated().any():
            dup_count = df[key_cols].duplicated().sum()
            status = "ERROR"
            error_message = (
                f"Duplicate keys found: {dup_count} duplicates in {key_cols}. "
                f"business_day + hour_business + target_month must be unique."
            )
            return RiskPackLoadResult(
                df=df,
                manifest=manifest,
                status=status,
                error_message=error_message,
            )
    else:
        missing_cols = [col for col in key_cols if col not in df.columns]
        status = "ERROR"
        error_message = f"Missing required columns: {missing_cols}"
        return RiskPackLoadResult(
            df=df,
            manifest=manifest,
            status=status,
            error_message=error_message,
        )
    
    return RiskPackLoadResult(
        df=df,
        manifest=manifest,
        warnings=warnings,
        status=status,
        error_message=error_message,
    )


class RiskPackLoader:
    """Loader for risk feature pack.

    Usage:
        loader = RiskPackLoader()
        result = loader.load(
            risk_pack_path="reports/local/risk_modules/risk_feature_pack_2026_01_05/risk_feature_pack.csv",
            manifest_path="reports/local/risk_modules/risk_feature_pack_2026_01_05/manifest.json",
            online_mode=True,
        )
        
        if result.status == "SUCCESS":
            df = result.df
            manifest = result.manifest
    """

    def __init__(self):
        self.last_result: Optional[RiskPackLoadResult] = None

    def load(
        self,
        risk_pack_path: str | Path,
        manifest_path: Optional[str | Path] = None,
        online_mode: bool = True,
    ) -> RiskPackLoadResult:
        """Load and validate risk feature pack.

        Args:
            risk_pack_path: Path to risk_feature_pack.csv.
            manifest_path: Optional path to manifest.json.
            online_mode: If True, y_true is not allowed.

        Returns:
            RiskPackLoadResult.
        """
        result = load_risk_pack(
            risk_pack_path=risk_pack_path,
            manifest_path=manifest_path,
            online_mode=online_mode,
        )
        
        self.last_result = result
        return result

    def validate(self, result: Optional[RiskPackLoadResult] = None) -> list[str]:
        """Validate loaded risk pack.

        Args:
            result: RiskPackLoadResult to validate. If None, uses last_result.

        Returns:
            List of validation errors (empty if valid).
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            return ["No risk pack loaded. Call load() first."]
        
        errors = []
        
        # Check status
        if result.status != "SUCCESS":
            errors.append(f"Load status: {result.status}")
            if result.error_message:
                errors.append(f"Error: {result.error_message}")
        
        # Check required columns
        required_cols = [
            "business_day", "hour_business", "target_month",
        ]
        
        for col in required_cols:
            if col not in result.df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check probability columns
        for col in PROBABILITY_COLUMNS:
            if col in result.df.columns:
                invalid_mask = ~result.df[col].between(0, 1) & result.df[col].notna()
                if invalid_mask.any():
                    errors.append(f"Column '{col}' has values outside [0, 1]")
        
        return errors

    def get_risk_scores(self, result: Optional[RiskPackLoadResult] = None) -> pd.DataFrame:
        """Extract risk scores from loaded risk pack.

        Args:
            result: RiskPackLoadResult. If None, uses last_result.

        Returns:
            DataFrame with risk score columns.
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            raise RuntimeError("No risk pack loaded. Call load() first.")
        
        if result.status != "SUCCESS":
            raise RuntimeError(f"Cannot get risk scores: load status = {result.status}")
        
        # Select risk score columns
        risk_cols = [
            "negative_prob", "negative_risk_score",
            "spike_prob", "spike_risk_score",
            "deviation_down_prob", "deviation_up_prob", "deviation_risk_score",
        ]
        
        available_cols = [col for col in risk_cols if col in result.df.columns]
        
        # Also include key columns
        key_cols = ["business_day", "hour_business", "target_month"]
        output_cols = key_cols + available_cols
        
        return result.df[output_cols].copy()
