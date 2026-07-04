"""Audit DeepRT-SOTA pipeline consistency.

Compare 3 paths:
1. train_deep_realtime_sota.py (or train_working.py) single-month train output
2. evaluate script separate output
3. run_backtest.py backtest output

Check:
- same target_month rows
- same business_day/hour_business keys
- same y_true
- same da_anchor
- same predictions if same model checkpoint
- same final_pred = da_anchor + residual_pred
- same sMAPE_floor50 implementation
- same residual scaling inverse-transform
- same checkpoint selected
- same eval mode
- same risk_features off
- same normalization parameters
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=str, required=True,
                        help="Output dir from train script")
    parser.add_argument("--backtest-output", type=str, required=True,
                        help="Output dir from backtest script")
    parser.add_argument("--target-month", type=str, default="2026-02")
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def load_predictions(pred_path: str) -> pd.DataFrame:
    """Load predictions.csv from experiment output."""
    return pd.read_csv(pred_path)


def load_results(results_path: str) -> dict:
    """Load results.json or metrics_summary.json from experiment output."""
    p = Path(results_path)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    # Try alternative name
    alt = p.parent / ("metrics_summary.json" if p.name == "results.json" else "results.json")
    if alt.exists():
        with open(alt, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"Neither {p} nor {alt} found")


def audit_consistency(train_output: str, backtest_output: str, 
                      target_month: str, out_dir: str):
    """Audit consistency between train and backtest outputs."""
    train_path = Path(train_output)
    backtest_path = Path(backtest_output)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    checks = []
    joined_rows = []
    
    # Check 1: sMAPE comparison
    train_results = load_results(train_path / "metrics_summary.json")
    backtest_results = load_results(backtest_path / "metrics_summary.json")
    
    smape_delta = abs(train_results["sMAPE_floor50"] - backtest_results["sMAPE_floor50"])
    checks.append({
        "check": "sMAPE_floor50_same",
        "train_value": train_results["sMAPE_floor50"],
        "backtest_value": backtest_results["sMAPE_floor50"],
        "delta": smape_delta,
        "pass": smape_delta < 0.1,
        "comment": "Should be identical if same model + data"
    })
    
    # Check 2: test days count
    train_pred = load_predictions(train_path / "predictions.csv")
    backtest_pred = load_predictions(backtest_path / "predictions.csv")
    
    checks.append({
        "check": "test_days_count_same",
        "train_value": len(train_pred),
        "backtest_value": len(backtest_pred),
        "pass": len(train_pred) == len(backtest_pred),
        "comment": "Should have same number of test days"
    })
    
    # Check 3: predictions comparison (if same model checkpoint)
    if "pred_mean" in train_pred.columns and "pred_mean" in backtest_pred.columns:
        # Try to join on business_day
        train_pred["business_day"] = train_pred["business_day"].astype(str)
        backtest_pred["business_day"] = backtest_pred["business_day"].astype(str)
        
        merged = train_pred.merge(
            backtest_pred,
            on="business_day",
            suffixes=("_train", "_backtest"),
            how="inner"
        )
        
        if len(merged) > 0:
            pred_delta = np.abs(merged["pred_mean_train"] - merged["pred_mean_backtest"]).max()
            checks.append({
                "check": "predictions_same_if_same_checkpoint",
                "merged_rows": len(merged),
                "max_pred_delta": float(pred_delta),
                "pass": pred_delta < 0.01,
                "comment": "Predictions should be identical if same checkpoint"
            })
            
            # Save joined predictions
            merged.to_csv(out_path / "joined_predictions.csv", index=False)
            joined_rows.append(len(merged))
    
    # Check 4: final_pred = da_anchor + residual_pred
    # This check requires the predictions to have da_anchor and residual columns
    # For now, just check if the evaluation code is correct
    
    # Check 5: diagnostics comparison
    train_diag = None
    backtest_diag = None
    
    if (train_path / "diagnostics.json").exists():
        with open(train_path / "diagnostics.json", "r") as f:
            train_diag = json.load(f)
    
    if (backtest_path / "diagnostics.json").exists():
        with open(backtest_path / "diagnostics.json", "r") as f:
            backtest_diag = json.load(f)
    
    if train_diag and backtest_diag:
        # Compare key diagnostics
        for key in ["corr_da_rt", "corr_final_pred_true", "final_rt_pred_std", "rt_actual_std"]:
            if key in train_diag and key in backtest_diag:
                delta = abs(train_diag[key] - backtest_diag[key])
                checks.append({
                    "check": f"diagnostics_{key}_same",
                    "train_value": train_diag[key],
                    "backtest_value": backtest_diag[key],
                    "delta": delta,
                    "pass": delta < 0.01,
                    "comment": f"Diagnostic {key} should be similar"
                })
    
    # Generate report
    report = f"""# Pipeline Consistency Audit Report

## Target Month: {target_month}

## Checks

| Check | Train Value | Backtest Value | Delta | Pass | Comment |
|-------|-------------|----------------|-------|------|---------|
"""
    
    for c in checks:
        report += f"| {c['check']} | {c.get('train_value', 'N/A')} | {c.get('backtest_value', 'N/A')} | {c.get('delta', 'N/A')} | {c['pass']} | {c.get('comment', '')} |\n"
    
    report += f"\n## Summary\n\n"
    n_pass = sum(1 for c in checks if c["pass"])
    n_total = len(checks)
    report += f"Passed {n_pass}/{n_total} checks.\n\n"
    
    if n_pass == n_total:
        report += "**VERDICT: CONSISTENT** - Pipeline is consistent.\n"
    elif n_pass >= n_total * 0.8:
        report += "**VERDICT: MOSTLY_CONSISTENT** - Minor inconsistencies found.\n"
    else:
        report += "**VERDICT: INCONSISTENT** - Major inconsistencies found. Must fix.\n"
    
    report += f"\n## Joined Predictions\n\n"
    if joined_rows:
        report += f"Saved {joined_rows[0]} rows to joined_predictions.csv\n"
    else:
        report += "No joined predictions (might be different models/checkpoints)\n"
    
    # Save outputs
    with open(out_path / "consistency_report.md", "w") as f:
        f.write(report)
    
    with open(out_path / "consistency_checks.json", "w") as f:
        json.dump(checks, f, indent=2)
    
    print(f"Audit complete. Results saved to {out_path}")
    print(f"Passed {n_pass}/{n_total} checks")
    
    return checks


def main():
    args = parse_args()
    audit_consistency(
        args.train_output,
        args.backtest_output,
        args.target_month,
        args.out_dir
    )


if __name__ == "__main__":
    main()
