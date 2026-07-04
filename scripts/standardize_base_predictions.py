"""Base Prediction Standardizer for Ledger-2.

Standardizes base prediction files to unified format.

Input:
    --input <candidate_file>
    --prediction-column <col>
    --actual-column <col optional>
    --model-name <sgdfnet|fusion|timesfm|timemixer|da_anchor>
    --target-months 2026-01,2026-02,2026-03,2026-04,2026-05
    --out-dir reports/local/ledger_2/base_predictions_standardized/<model_name>

Output unified format:
    business_day, hour_business, target_month, ds, base_pred, base_model_name, base_source, optional y_true

Output manifest:
    model_name, source_file, prediction_column, actual_column,
    n_rows, coverage_by_month, oracle_baseline_detected, evaluation_allowed, created_at
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np
from typing import Optional, List
import json
from datetime import datetime


def standardize_base_prediction(
    input_file: str | Path,
    prediction_column: str,
    actual_column: Optional[str] = None,
    model_name: str = "unknown",
    target_months: Optional[List[str]] = None,
    out_dir: str | Path = "reports/local/ledger_2/base_predictions_standardized",
) -> dict:
    """Standardize a base prediction file to unified format.
    
    Args:
        input_file: Path to input base prediction file.
        prediction_column: Column name for predictions.
        actual_column: Optional column name for actuals.
        model_name: Model name (sgdfnet, fusion, timesfm, etc.).
        target_months: List of target months to filter to.
        out_dir: Output directory.
    
    Returns:
        Dict with standardization metadata.
    """
    input_file = Path(input_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input file
    df = None
    for encoding in ["utf-8", "gbk", "gb2312", "latin1"]:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise RuntimeError(f"Failed to read {input_file} with any encoding")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Check prediction column exists
    if prediction_column not in df.columns:
        raise ValueError(
            f"Prediction column '{prediction_column}' not found in {input_file}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Rename prediction column to base_pred
    df = df.rename(columns={prediction_column: "base_pred"})
    
    # Handle actual column
    if actual_column is not None and actual_column in df.columns:
        df = df.rename(columns={actual_column: "y_true"})
    else:
        df["y_true"] = np.nan
    
    # Ensure timestamp column exists
    if "ds" not in df.columns:
        # Try to find timestamp column
        ts_candidates = ["timestamp", "date", "time", "DateTime", "datetime"]
        for col in ts_candidates:
            if col in df.columns:
                df = df.rename(columns={col: "ds"})
                break
        
        if "ds" not in df.columns:
            raise ValueError(
                f"Cannot find timestamp column in {input_file}. "
                f"Available columns: {list(df.columns)}"
            )
    
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Add business time columns if not present
    if "business_day" not in df.columns or "hour_business" not in df.columns:
        from models.deep_sgdf_delta.business_time import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")
    
    # Add target_month if not present
    if "target_month" not in df.columns:
        df["target_month"] = pd.to_datetime(df["business_day"]).dt.strftime("%Y-%m")
    
    # Filter to target months if specified
    if target_months is not None:
        target_months_set = set(target_months)
        df = df[df["target_month"].isin(target_months_set)].copy()
    
    # Add model name and source
    df["base_model_name"] = model_name
    df["base_source"] = "BASE_PREDICTION_FILE"
    
    # Select standard output columns
    output_cols = [
        "business_day", "hour_business", "target_month", "ds",
        "base_pred", "base_model_name", "base_source", "y_true",
    ]
    output_cols = [col for col in output_cols if col in df.columns]
    
    result_df = df[output_cols].copy()
    
    # Oracle baseline detection
    oracle_baseline_detected = False
    evaluation_allowed = True
    
    if "y_true" in result_df.columns:
        valid_mask = result_df["y_true"].notna()
        if valid_mask.sum() > 0:
            base_pred_valid = result_df.loc[valid_mask, "base_pred"].values
            y_true_valid = result_df.loc[valid_mask, "y_true"].values
            if np.allclose(base_pred_valid, y_true_valid, equal_nan=True):
                oracle_baseline_detected = True
                evaluation_allowed = False
    
    # Save standardized file
    output_file = out_dir / "base_predictions.csv"
    result_df.to_csv(output_file, index=False)
    
    # Calculate coverage by month
    coverage_by_month = {}
    if "target_month" in result_df.columns:
        for month in result_df["target_month"].unique():
            coverage_by_month[month] = int((result_df["target_month"] == month).sum())
    
    # Create manifest
    manifest = {
        "model_name": model_name,
        "source_file": str(input_file),
        "prediction_column": prediction_column,
        "actual_column": actual_column,
        "n_rows": len(result_df),
        "coverage_by_month": coverage_by_month,
        "oracle_baseline_detected": oracle_baseline_detected,
        "evaluation_allowed": evaluation_allowed,
        "created_at": datetime.now().isoformat(),
    }
    
    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save metadata
    metadata = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "manifest_file": str(manifest_file),
        "model_name": model_name,
        "n_rows": len(result_df),
        "n_months": len(coverage_by_month),
        "oracle_baseline_detected": oracle_baseline_detected,
        "evaluation_allowed": evaluation_allowed,
    }
    
    return metadata


def main():
    """Main function to standardize base predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize base prediction files")
    parser.add_argument("--input", required=True, help="Input base prediction file")
    parser.add_argument("--prediction-column", required=True, help="Prediction column name")
    parser.add_argument("--actual-column", default=None, help="Actual column name (optional)")
    parser.add_argument("--model-name", required=True, help="Model name (sgdfnet, fusion, timesfm, etc.)")
    parser.add_argument("--target-months", required=True, help="Comma-separated list of target months")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    target_months = args.target_months.split(",")
    
    metadata = standardize_base_prediction(
        input_file=args.input,
        prediction_column=args.prediction_column,
        actual_column=args.actual_column,
        model_name=args.model_name,
        target_months=target_months,
        out_dir=args.out_dir,
    )
    
    print("=" * 80)
    print("Base Prediction Standardizer")
    print("=" * 80)
    print(f"\nInput: {metadata['input_file']}")
    print(f"Output: {metadata['output_file']}")
    print(f"Model: {metadata['model_name']}")
    print(f"Rows: {metadata['n_rows']}")
    print(f"Months: {metadata['n_months']}")
    print(f"Oracle baseline detected: {metadata['oracle_baseline_detected']}")
    print(f"Evaluation allowed: {metadata['evaluation_allowed']}")
    print("\n" + "=" * 80)
    print("Standardization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
