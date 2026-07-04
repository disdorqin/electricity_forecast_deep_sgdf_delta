"""Run risk guardrail shadow replay.

Usage:
    python scripts/run_risk_guardrail_shadow_replay.py \
        --risk-pack reports/local/risk_modules/risk_feature_pack_2026_01_05/risk_feature_pack.csv \
        --risk-pack-manifest reports/local/risk_modules/risk_feature_pack_2026_01_05/manifest.json \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --base-mode da_anchor \
        --target-months 2026-01,2026-02,2026-03,2026-04,2026-05 \
        --out-dir reports/local/ledger_1/shadow_replay_2026_01_05
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deep_sgdf_delta.risk_shadow_replay import (
    RiskShadowReplay,
    ShadowReplayConfig,
)
from models.deep_sgdf_delta.base_prediction_adapter import BasePredictionAdapter
from models.deep_sgdf_delta.risk_pack_loader import RiskPackLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run risk guardrail shadow replay.",
    )
    
    parser.add_argument(
        "--risk-pack",
        type=str,
        required=True,
        help="Path to risk feature pack CSV.",
    )
    
    parser.add_argument(
        "--risk-pack-manifest",
        type=str,
        default=None,
        help="Path to risk pack manifest JSON. If not provided, tries to find in same directory.",
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to Shandong PMOS data (for DA anchor fallback).",
    )
    
    parser.add_argument(
        "--base-predictions",
        type=str,
        default=None,
        help="Path to base prediction CSV (optional, overrides DA anchor).",
    )
    
    parser.add_argument(
        "--base-mode",
        type=str,
        default="da_anchor",
        choices=["da_anchor", "base_prediction_file"],
        help="Base prediction mode.",
    )
    
    parser.add_argument(
        "--target-months",
        type=str,
        required=True,
        help="Comma-separated list of target months (e.g., '2026-01,2026-02').",
    )
    
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    
    parser.add_argument(
        "--negative-thresholds",
        type=str,
        default="0.4,0.5,0.6,0.7",
        help="Comma-separated negative thresholds for policy sweep.",
    )
    
    parser.add_argument(
        "--spike-thresholds",
        type=str,
        default="0.4,0.5,0.6,0.7",
        help="Comma-separated spike thresholds for policy sweep.",
    )
    
    parser.add_argument(
        "--blend-weights",
        type=str,
        default="0.05,0.1,0.2",
        help="Comma-separated blend weights for policy sweep.",
    )
    
    parser.add_argument(
        "--require-y-true",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Require y_true for evaluation (default: true). If true, fail if y_true is missing.",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Parse target months
    target_months = [m.strip() for m in args.target_months.split(",")]
    
    # Parse policy sweep grid
    negative_thresholds = [float(t) for t in args.negative_thresholds.split(",")]
    spike_thresholds = [float(t) for t in args.spike_thresholds.split(",")]
    blend_weights = [float(w) for w in args.blend_weights.split(",")]
    
    # Create config
    config = ShadowReplayConfig(
        risk_pack_path=args.risk_pack,
        manifest_path=args.risk_pack_manifest,
        data_path=args.data_path,
        base_prediction_file=args.base_predictions,
        base_mode=args.base_mode,
        target_months=target_months,
        out_dir=args.out_dir,
        negative_thresholds=negative_thresholds,
        spike_thresholds=spike_thresholds,
        blend_weights=blend_weights,
    )
    
    # Run shadow replay
    print("=" * 80)
    print("Risk Guardrail Shadow Replay")
    print("=" * 80)
    print(f"\nTarget months: {target_months}")
    print(f"Base mode: {args.base_mode}")
    print(f"Risk pack: {args.risk_pack}")
    print(f"Policy sweep grid:")
    print(f"  Negative thresholds: {negative_thresholds}")
    print(f"  Spike thresholds: {spike_thresholds}")
    print(f"  Blend weights: {blend_weights}")
    print()
    
    engine = RiskShadowReplay()
    result = engine.run(config)
    
    # Check if y_true is required but missing
    require_y_true = args.require_y_true.lower() == "true"
    
    if require_y_true:
        input_diag_path = Path(args.out_dir) / "input_diagnostics.json"
        if input_diag_path.exists():
            with open(input_diag_path, "r") as f:
                diagnostics = json.load(f)
            
            if not diagnostics.get("has_y_true") or diagnostics.get("y_true_non_null_count", 0) == 0:
                print("\n" + "=" * 80)
                print("ERROR: y_true is required but missing or all null!")
                print("Input diagnostics:")
                print(json.dumps(diagnostics, indent=2))
                print("=" * 80)
                print("\nTo run without y_true, use --require-y-true false")
                sys.exit(1)
        else:
            print("\nWARNING: input_diagnostics.json not found, cannot verify y_true.")
    
    # Print warnings
    if result.warnings:
        print("WARNINGS:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print()
    
    # Print champion policy
    print("CHAMPION POLICY:")
    print(f"  Negative threshold: {result.champion_policy['negative_threshold']}")
    print(f"  Spike threshold: {result.champion_policy['spike_threshold']}")
    print(f"  Blend weight: {result.champion_policy['blend_weight']}")
    print()
    
    # Print metrics
    print("METRICS:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Export results
    print("Exporting results...")
    engine.export_results(result, out_dir=args.out_dir)
    
    # Generate report
    print("\nGenerating report...")
    _generate_report(result, out_dir=args.out_dir)
    
    print("\nDone!")


def _generate_report(result, out_dir: str | Path):
    """Generate shadow replay report.

    Args:
        result: ShadowReplayResult.
        out_dir: Output directory.
    """
    out_dir = Path(out_dir)
    report_path = out_dir / "shadow_replay_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Shadow Replay Report\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Target months**: {result.config.target_months}\n")
        f.write(f"- **Base mode**: {result.config.base_mode}\n")
        f.write(f"- **Risk pack**: {result.config.risk_pack_path}\n")
        f.write(f"- **Policy sweep grid**:\n")
        f.write(f"  - Negative thresholds: {result.config.negative_thresholds}\n")
        f.write(f"  - Spike thresholds: {result.config.spike_thresholds}\n")
        f.write(f"  - Blend weights: {result.config.blend_weights}\n")
        f.write("\n")
        
        f.write("## Base Prediction Source\n\n")
        f.write(f"- **Source**: {result.base_pred_result.source}\n")
        f.write(f"- **Model name**: {result.base_pred_result.model_name}\n")
        f.write(f"- **Production candidate**: {result.base_pred_result.production_candidate}\n")
        
        if result.base_pred_result.source == "DA_ANCHOR_BASELINE":
            f.write("\n**WARNING**: This is a DA anchor baseline (fallback), NOT a production baseline.\n")
            f.write("This is a guardrail sensitivity test, not a production evaluation.\n")
            f.write("Results may not generalize to production.\n")
        
        f.write("\n")
        
        f.write("## Risk Pack Status\n\n")
        f.write(f"- **Load status**: {result.risk_pack_result.status}\n")
        f.write(f"- **Risk feature version**: {result.risk_pack_result.manifest.get('risk_feature_version', 'UNKNOWN')}\n")
        f.write(f"- **Metric alignment status**: {result.risk_pack_result.manifest.get('metric_alignment_status', 'UNKNOWN')}\n")
        f.write(f"- **Quality gate passed**: {result.risk_pack_result.manifest.get('quality_gate_passed', 'UNKNOWN')}\n")
        
        if result.risk_pack_result.warnings:
            f.write("\n**Warnings**:\n")
            for warning in result.risk_pack_result.warnings:
                f.write(f"- {warning}\n")
        
        f.write("\n")
        
        f.write("## Champion Policy\n\n")
        f.write(f"- **Negative threshold**: {result.champion_policy['negative_threshold']}\n")
        f.write(f"- **Spike threshold**: {result.champion_policy['spike_threshold']}\n")
        f.write(f"- **Blend weight**: {result.champion_policy['blend_weight']}\n")
        f.write("\n")
        
        f.write("## Metrics\n\n")
        f.write("| Metric | Base | Adjusted | Improvement |\n")
        f.write("|---------|-------|------------|--------------|\n")
        
        for key in ["sMAPE_floor50", "sMAPE", "MAE", "RMSE"]:
            base_key = f"base_{key}"
            adjusted_key = f"adjusted_{key}"
            improvement_key = f"{key}_improvement"
            
            if base_key in result.metrics and adjusted_key in result.metrics:
                base_val = result.metrics[base_key]
                adjusted_val = result.metrics[adjusted_key]
                improvement = result.metrics.get(improvement_key, 0.0)
                
                f.write(f"| {key} | {base_val:.4f} | {adjusted_val:.4f} | {improvement:.4f} |\n")
        
        f.write("\n")
        
        f.write("## Policy Sweep Results\n\n")
        f.write("Top 5 policies:\n\n")
        
        top5 = result.policy_sweep_df.head(5)
        f.write("| Negative Thresh | Spike Thresh | Blend Weight | sMAPE_floor50 Improvement |\n")
        f.write("|------------------|----------------|---------------|-----------------------------|\n")
        
        for _, row in top5.iterrows():
            f.write(
                f"| {row['negative_threshold']:.2f} "
                f"| {row['spike_threshold']:.2f} "
                f"| {row['blend_weight']:.2f} "
                f"| {row.get('sMAPE_floor50_improvement', 0.0):.4f} |\n"
            )
        
        f.write("\n")
        
        f.write("## Decision Log\n\n")
        f.write(f"- **Total rows**: {len(result.decision_log)}\n")
        
        if "guardrail_triggered" in result.decision_log.columns:
            trigger_rate = result.decision_log["guardrail_triggered"].mean()
            f.write(f"- **Trigger rate**: {trigger_rate:.2%}\n")
        
        f.write("\n")
        
        f.write("## Conclusion\n\n")
        
        smape_improvement = result.metrics.get("sMAPE_floor50_improvement", 0.0)
        
        if smape_improvement > 0.3:
            f.write("**Guardril improves overall sMAPE.**\n")
            f.write("Recommendation: **PROCEED** to mainline shadow.\n")
        elif smape_improvement > 0.0:
            f.write("**Guardril slightly improves overall sMAPE.**\n")
            f.write("Recommendation: **Consider** for mainline shadow.\n")
        else:
            f.write("**Guardril does NOT improve overall sMAPE.**\n")
            f.write("Recommendation: **DO NOT** proceed to mainline shadow.\n")
        
        f.write("\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `policy_sweep.csv`: Policy sweep results.\n")
        f.write("- `decision_log.csv`: Decision log for champion policy.\n")
        f.write("- `champion_policy.json`: Champion policy configuration.\n")
        f.write("- `shadow_metrics.csv`: Final metrics.\n")
        f.write("- `shadow_play_report.md`: This report.\n")
    
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
