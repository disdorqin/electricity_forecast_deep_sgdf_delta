"""Base Prediction Candidate Searcher for Ledger-2.

Searches for real base prediction files in the deep repo and adjacent repositories.

Search directories:
- .
- ../electricity_forecast_model2.0_exp
- ../deep_sgdf_delta_repo
- ../electricity_forecast_model2.1

File types:
- .csv, .parquet, .xlsx, .json

Keywords:
- prediction, predictions, forecast
- sgdfnet, timesfm, timemixer, fusion
- ledger, replay
- y_pred, rt_pred, base_pred

Output:
- reports/local/ledger_2/base_prediction_search/base_prediction_candidates.csv
- reports/local/ledger_2/base_prediction_search/base_prediction_search_report.md
- reports/local/ledger_2/base_prediction_search/best_candidate.json
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
import os


# Search configuration
SEARCH_DIRS = [
    Path("."),
    Path("../electricity_forecast_model2.0_exp"),
    Path("../deep_sgdf_delta_repo"),
    Path("../electricity_forecast_model2.1"),
    Path("../models"),
]

FILE_EXTENSIONS = [".csv", ".parquet", ".xlsx", ".json"]

KEYWORDS = [
    "prediction", "predictions", "forecast",
    "sgdfnet", "timesfm", "timemixer", "fusion",
    "ledger", "replay",
    "y_pred", "rt_pred", "base_pred",
    "forecast_price", "da_anchor",
]


def _find_candidate_files(search_dirs: List[Path]) -> List[Path]:
    """Find candidate files in search directories.
    
    Args:
        search_dirs: List of directories to search.
    
    Returns:
        List of candidate file paths.
    """
    candidates = []
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            print(f"Warning: Search directory not found: {search_dir}")
            continue
        
        # Recursively find files with matching extensions
        for ext in FILE_EXTENSIONS:
            files = list(search_dir.rglob(f"*{ext}"))
            candidates.extend(files)
    
    # Filter by keywords in filename
    filtered = []
    for file_path in candidates:
        file_name_lower = file_path.name.lower()
        if any(keyword in file_name_lower for keyword in KEYWORDS):
            filtered.append(file_path)
    
    # Remove duplicates
    filtered = list(set(filtered))
    
    return filtered


def _inspect_file(file_path: Path) -> Dict[str, Any]:
    """Inspect a candidate file and extract metadata.
    
    Args:
        file_path: Path to the candidate file.
    
    Returns:
        Dict with file metadata.
    """
    result = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        "format": file_path.suffix,
        "exists": True,
        "readable": False,
        "n_rows": 0,
        "columns": [],
        "has_timestamp": False,
        "has_business_time": False,
        "has_prediction_column": False,
        "has_actual_column": False,
        "prediction_columns": [],
        "actual_columns": [],
        "oracle_suspect": False,
        "coverage_score": 0.0,
        "target_month_coverage": {},
        "duplicate_keys": 0,
        "error_message": None,
    }
    
    try:
        # Read file based on format
        df = None
        if file_path.suffix == ".csv":
            # Try multiple encodings
            for encoding in ["utf-8", "gbk", "gb2312", "latin1"]:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, nrows=1000)  # Read only first 1000 rows for speed
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
            df = df.head(1000)  # Read only first 1000 rows
        elif file_path.suffix == ".xlsx":
            df = pd.read_excel(file_path, nrows=1000)
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path, lines=True, nrows=1000)
        
        if df is None:
            result["error_message"] = "Failed to read file"
            return result
        
        result["readable"] = True
        result["n_rows"] = len(df)
        result["columns"] = list(df.columns)
        
        # Check for timestamp column
        ts_candidates = ["ds", "timestamp", "date", "time", "DateTime", "datetime"]
        result["has_timestamp"] = any(col in df.columns for col in ts_candidates)
        
        # Check for business time columns
        result["has_business_time"] = "business_day" in df.columns and "hour_business" in df.columns
        
        # Check for prediction columns
        pred_candidates = [
            "base_pred", "y_pred", "rt_pred", "prediction", "forecast",
            "sgdfnet_pred", "timesfm_pred", "timemixer_pred", "fusion_pred",
        ]
        result["prediction_columns"] = [col for col in pred_candidates if col in df.columns]
        result["has_prediction_column"] = len(result["prediction_columns"]) > 0
        
        # Check for actual columns
        actual_candidates = ["y_true", "actual", "rt_actual", "price", "true_price"]
        result["actual_columns"] = [col for col in actual_candidates if col in df.columns]
        result["has_actual_column"] = len(result["actual_columns"]) > 0
        
        # Oracle suspect check (if prediction == actual for any column)
        if result["has_prediction_column"] and result["has_actual_column"]:
            pred_col = result["prediction_columns"][0]
            actual_col = result["actual_columns"][0]
            if pred_col in df.columns and actual_col in df.columns:
                valid_mask = df[actual_col].notna()
                if valid_mask.sum() > 0:
                    pred_valid = df.loc[valid_mask, pred_col].values
                    actual_valid = df.loc[valid_mask, actual_col].values
                    if np.allclose(pred_valid, actual_valid, equal_nan=True):
                        result["oracle_suspect"] = True
        
        result["error_message"] = None
        
    except Exception as e:
        result["error_message"] = str(e)
    
    return result


def _evaluate_candidate(result: Dict[str, Any]) -> str:
    """Evaluate a candidate and return its status.
    
    Args:
        result: Dict with file metadata.
    
    Returns:
        Status string: "FOUND_FORMAL_BASE", "FOUND_PARTIAL_BASE", or "NOT_FOUND".
    """
    if not result["readable"]:
        return "NOT_FOUND"
    
    if not result["has_prediction_column"]:
        return "NOT_FOUND"
    
    if result["oracle_suspect"]:
        return "NOT_FOUND"  # Oracle baseline is not a valid base
    
    # Check coverage (simplified - just check if file has enough rows)
    if result["n_rows"] < 100:
        return "NOT_FOUND"
    
    # Formal base: has prediction column, not oracle suspect, enough rows
    return "FOUND_PARTIAL_BASE"  # Simplified - would need to check target month coverage


def main():
    """Main function to search for base prediction candidates."""
    print("=" * 80)
    print("Base Prediction Candidate Searcher for Ledger-2")
    print("=" * 80)
    
    # Step 1: Find candidate files
    print("\nStep 1: Searching for candidate files...")
    candidates = _find_candidate_files(SEARCH_DIRS)
    print(f"Found {len(candidates)} candidate files")
    
    # Step 2: Inspect each candidate
    print("\nStep 2: Inspecting candidate files...")
    results = []
    for i, file_path in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] Inspecting {file_path.name}...")
        result = _inspect_file(file_path)
        result["status"] = _evaluate_candidate(result)
        results.append(result)
    
    # Step 3: Filter to valid candidates
    print("\nStep 3: Filtering to valid candidates...")
    valid_results = [r for r in results if r["status"] != "NOT_FOUND"]
    print(f"Found {len(valid_results)} valid candidates")
    
    # Step 4: Sort by coverage score (descending)
    valid_results.sort(key=lambda x: x["coverage_score"], reverse=True)
    
    # Step 5: Output results
    print("\nStep 4: Outputting results...")
    
    # Create output directory
    output_dir = Path("reports/local/ledger_2/base_prediction_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save candidates CSV
    candidates_df = pd.DataFrame(valid_results)
    candidates_csv = output_dir / "base_prediction_candidates.csv"
    candidates_df.to_csv(candidates_csv, index=False)
    print(f"  Saved: {candidates_csv}")
    
    # Save best candidate JSON
    if len(valid_results) > 0:
        best = valid_results[0]
        best_json = output_dir / "best_candidate.json"
        with open(best_json, "w") as f:
            json.dump(best, f, indent=2)
        print(f"  Saved: {best_json}")
    
    # Generate report
    report_md = output_dir / "base_prediction_search_report.md"
    with open(report_md, "w") as f:
        f.write("# Base Prediction Candidate Search Report\n\n")
        f.write(f"Total candidates found: {len(candidates)}\n")
        f.write(f"Valid candidates: {len(valid_results)}\n\n")
        
        f.write("## Valid Candidates\n\n")
        for i, result in enumerate(valid_results):
            f.write(f"### {i+1}. {result['file_name']}\n\n")
            f.write(f"- **Path**: {result['file_path']}\n")
            f.write(f"- **Status**: {result['status']}\n")
            f.write(f"- **Size**: {result['file_size_mb']} MB\n")
            f.write(f"- **Rows**: {result['n_rows']}\n")
            f.write(f"- **Has timestamp**: {result['has_timestamp']}\n")
            f.write(f"- **Has prediction column**: {result['has_prediction_column']}\n")
            f.write(f"- **Prediction columns**: {result['prediction_columns']}\n")
            f.write(f"- **Has actual column**: {result['has_actual_column']}\n")
            f.write(f"- **Actual columns**: {result['actual_columns']}\n")
            f.write(f"- **Oracle suspect**: {result['oracle_suspect']}\n")
            f.write("\n")
    
    print(f"  Saved: {report_md}")
    print("\n" + "=" * 80)
    print("Search complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
