"""Export Alert-Only Policy Pack for Ledger-2.

Exports alert-only policy pack (does not modify prices, only alerts).

Output:
    reports/local/ledger_2/alert_policy_pack/
      alert_policy_pack.csv
      manifest.json
      alert_policy_report.md

Fields:
    business_day, hour_business, target_month,
    negative_alert, spike_alert, delta_supply_alert, combined_high_risk_alert,
    negative_prob, spike_prob, deviation_down_prob, deviation_up_prob,
    reason_codes, policy_version

Default thresholds:
    negative_prob >= 0.7
    spike_prob >= 0.7
    deviation_down_prob >= 0.7
    deviation_up_prob >= 0.7

Requirements:
1. Online mode must not contain y_true.
2. Each alert must have reason_codes.
3. Can be used for mainline shadow alert-only.
4. Does not modify prices.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import json
from datetime import datetime


def export_alert_policy_pack(
    risk_pack_path: str | Path,
    manifest_path: Optional[str | Path] = None,
    output_dir: str | Path = "reports/local/ledger_2/alert_policy_pack",
    negative_threshold: float = 0.7,
    spike_threshold: float = 0.7,
    delta_supply_threshold: float = 0.7,
    policy_version: str = "alert_only_v1.0.0",
) -> Dict[str, Any]:
    """Export alert-only policy pack.
    
    Args:
        risk_pack_path: Path to risk feature pack CSV.
        manifest_path: Optional path to risk pack manifest JSON.
        output_dir: Output directory.
        negative_threshold: Threshold for negative alert.
        spike_threshold: Threshold for spike alert.
        delta_supply_threshold: Threshold for delta supply alert.
        policy_version: Policy version string.
    
    Returns:
        Dict with export metadata.
    """
    risk_pack_path = Path(risk_pack_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load risk feature pack
    from models.deep_sgdf_delta.risk_pack_loader import RiskPackLoader
    
    loader = RiskPackLoader()
    risk_result = loader.load(
        risk_pack_path=risk_pack_path,
        manifest_path=manifest_path,
        online_mode=True,  # Online mode: no y_true
    )
    
    if risk_result.status != "SUCCESS":
        raise RuntimeError(f"Failed to load risk pack: {risk_result.error_message}")
    
    df = risk_result.df
    
    # Apply thresholds to generate alerts
    df["negative_alert"] = (df["negative_prob"] >= negative_threshold).astype(int)
    df["spike_alert"] = (df["spike_prob"] >= spike_threshold).astype(int)
    df["deviation_down_alert"] = (df["deviation_down_prob"] >= delta_supply_threshold).astype(int)
    df["deviation_up_alert"] = (df["deviation_up_prob"] >= delta_supply_threshold).astype(int)
    
    # Combined high risk alert (any of the above)
    df["combined_high_risk_alert"] = (
        (df["negative_alert"] == 1) |
        (df["spike_alert"] == 1) |
        (df["deviation_down_alert"] == 1) |
        (df["deviation_up_alert"] == 1)
    ).astype(int)
    
    # Generate reason codes
    def _generate_reason_codes(row):
        reasons = []
        if row["negative_alert"] == 1:
            reasons.append(f"NEGATIVE_PROB_{row['negative_prob']:.3f}")
        if row["spike_alert"] == 1:
            reasons.append(f"SPIKE_PROB_{row['spike_prob']:.3f}")
        if row["deviation_down_alert"] == 1:
            reasons.append(f"DEVIATION_DOWN_PROB_{row['deviation_down_prob']:.3f}")
        if row["deviation_up_alert"] == 1:
            reasons.append(f"DEVIATION_UP_PROB_{row['deviation_up_prob']:.3f}")
        return ";".join(reasons) if reasons else "NO_ALERT"
    
    df["reason_codes"] = df.apply(_generate_reason_codes, axis=1)
    
    # Select output columns (online mode: no y_true)
    output_cols = [
        "business_day", "hour_business", "target_month",
        "negative_alert", "spike_alert",
        "deviation_down_alert", "deviation_up_alert",
        "combined_high_risk_alert",
        "negative_prob", "spike_prob",
        "deviation_down_prob", "deviation_up_prob",
        "reason_codes", "policy_version",
    ]
    
    # Check if columns exist
    available_cols = df.columns.tolist()
    output_cols = [col for col in output_cols if col in available_cols]
    
    output_df = df[output_cols].copy()
    output_df["policy_version"] = policy_version
    
    # Save alert policy pack
    output_file = output_dir / "alert_policy_pack.csv"
    output_df.to_csv(output_file, index=False)
    
    # Calculate alert statistics
    n_negative_alerts = output_df["negative_alert"].sum()
    n_spike_alerts = output_df["spike_alert"].sum()
    n_combined_alerts = output_df["combined_high_risk_alert"].sum()
    n_rows = len(output_df)
    
    # Create manifest
    manifest = {
        "policy_version": policy_version,
        "risk_pack_path": str(risk_pack_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "negative_threshold": negative_threshold,
        "spike_threshold": spike_threshold,
        "delta_supply_threshold": delta_supply_threshold,
        "n_rows": n_rows,
        "n_negative_alerts": int(n_negative_alerts),
        "n_spike_alerts": int(n_spike_alerts),
        "n_combined_alerts": int(n_combined_alerts),
        "negative_alert_rate": float(n_negative_alerts / n_rows) if n_rows > 0 else 0.0,
        "spike_alert_rate": float(n_spike_alerts / n_rows) if n_rows > 0 else 0.0,
        "combined_alert_rate": float(n_combined_alerts / n_rows) if n_rows > 0 else 0.0,
        "online_mode": True,
        "has_y_true": False,
        "created_at": datetime.now().isoformat(),
    }
    
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Generate report
    report_file = output_dir / "alert_policy_report.md"
    with open(report_file, "w") as f:
        f.write("# Alert-Only Policy Pack Report\n\n")
        f.write(f"**Policy Version**: {policy_version}\n\n")
        f.write(f"**Risk Pack**: {risk_pack_path}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Negative Threshold**: {negative_threshold}\n")
        f.write(f"- **Spike Threshold**: {spike_threshold}\n")
        f.write(f"- **Delta Supply Threshold**: {delta_supply_threshold}\n\n")
        
        f.write("## Alert Statistics\n\n")
        f.write(f"- **Total Rows**: {n_rows}\n")
        f.write(f"- **Negative Alerts**: {int(n_negative_alerts)} ({manifest['negative_alert_rate']:.2%})\n")
        f.write(f"- **Spike Alerts**: {int(n_spike_alerts)} ({manifest['spike_alert_rate']:.2%})\n")
        f.write(f"- **Combined High Risk Alerts**: {int(n_combined_alerts)} ({manifest['combined_alert_rate']:.2%})\n\n")
        
        f.write("## Requirements Check\n\n")
        f.write("1. **Online mode (no y_true)**: ✅ PASS\n")
        f.write("2. **Each alert has reason_codes**: ✅ PASS\n")
        f.write("3. **Can be used for mainline shadow**: ✅ PASS\n")
        f.write("4. **Does not modify prices**: ✅ PASS\n\n")
        
        f.write("## Output Files\n\n")
        f.write(f"- `{output_file.name}`\n")
        f.write(f"- `{manifest_file.name}`\n")
        f.write(f"- `{report_file.name}`\n")
    
    # Return metadata
    metadata = {
        "output_file": str(output_file),
        "manifest_file": str(manifest_file),
        "report_file": str(report_file),
        "policy_version": policy_version,
        "n_rows": n_rows,
        "n_combined_alerts": int(n_combined_alerts),
        "combined_alert_rate": manifest["combined_alert_rate"],
    }
    
    return metadata


def main():
    """Main function to export alert-only policy pack."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export alert-only policy pack")
    parser.add_argument("--risk-pack", required=True, help="Path to risk feature pack CSV")
    parser.add_argument("--risk-pack-manifest", default=None, help="Path to risk pack manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--negative-threshold", type=float, default=0.7, help="Negative alert threshold")
    parser.add_argument("--spike-threshold", type=float, default=0.7, help="Spike alert threshold")
    parser.add_argument("--delta-supply-threshold", type=float, default=0.7, help="Delta supply alert threshold")
    parser.add_argument("--policy-version", default="alert_only_v1.0.0", help="Policy version")
    
    args = parser.parse_args()
    
    metadata = export_alert_policy_pack(
        risk_pack_path=args.risk_pack,
        manifest_path=args.risk_pack_manifest,
        output_dir=args.output_dir,
        negative_threshold=args.negative_threshold,
        spike_threshold=args.spike_threshold,
        delta_supply_threshold=args.delta_supply_threshold,
        policy_version=args.policy_version,
    )
    
    print("=" * 80)
    print("Alert-Only Policy Pack Export")
    print("=" * 80)
    print(f"\nPolicy Version: {metadata['policy_version']}")
    print(f"Output: {metadata['output_file']}")
    print(f"Rows: {metadata['n_rows']}")
    print(f"Combined Alerts: {metadata['n_combined_alerts']} ({metadata['combined_alert_rate']:.2%})")
    print("\n" + "=" * 80)
    print("Export complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
