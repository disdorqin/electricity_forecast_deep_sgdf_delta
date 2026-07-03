#!/usr/bin/env python
"""Consolidate multiple SGDFNet prediction files into a single standard file.

Reads one or more input prediction files/directories, auto-detects columns,
deduplicates by (business_day, hour_business), and outputs a standard format.

Priority: formal protocol output > forecast output > validation fold
          later file > older file

Usage:
    python scripts/consolidate_sgdfnet_predictions.py \
        --inputs <file1.csv> <dir1> \
        --target-start 2026-01-01 \
        --target-end 2026-05-31 \
        --out reports/local/deep_final/sgdfnet_predictions/consolidated.csv
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.sgdfnet_prediction_loader import (
    SGDFNetPredictionLoader, save_coverage_report, CoverageReport,
)
from models.deep_sgdf_delta.business_time import add_business_time_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("consolidate_sgdfnet_predictions")

PREDICTION_ALIASES = ["sgdfnet_pred", "y_pred", "rt_hat", "prediction", "pred"]
TIMESTAMP_ALIASES = ["ds", "timestamp", "time"]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate SGDFNet prediction files")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input files or directories to scan")
    parser.add_argument("--target-start", type=str, default="2026-01-01")
    parser.add_argument("--target-end", type=str, default="2026-05-31")
    parser.add_argument("--out", type=str, required=True,
                        help="Output CSV path")
    return parser.parse_args()


def _collect_files(inputs: list[str]) -> list[Path]:
    files = []
    seen = set()
    for inp in inputs:
        p = Path(inp)
        if not p.exists():
            logger.warning("Input not found: %s", p)
            continue
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*.csv")):
                if str(f) not in seen:
                    seen.add(str(f))
                    files.append(f)
    return sorted(files)


def _detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    ts_col = next((c for c in TIMESTAMP_ALIASES if c in df.columns), None)
    pred_col = next((c for c in PREDICTION_ALIASES if c in df.columns), None)
    return ts_col, pred_col


def main():
    args = parse_args()
    files = _collect_files(args.inputs)
    logger.info("Found %d candidate files", len(files))

    all_rows = []
    source_order = ["protocol", "forecast", "fold", "other"]
    file_index = {str(f): i for i, f in enumerate(files)}

    for fpath in files:
        try:
            try:
                df = pd.read_csv(fpath, encoding="utf-8-sig")
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    df = pd.read_csv(fpath, encoding="gbk")
                except Exception:
                    continue
        except Exception:
            continue

        ts_col, pred_col = _detect_columns(df)
        if not ts_col or not pred_col:
            continue

        # Build record
        record = pd.DataFrame()
        record["ds"] = pd.to_datetime(df[ts_col])
        record["sgdfnet_pred"] = pd.to_numeric(df[pred_col], errors="coerce")
        record = record.dropna(subset=["sgdfnet_pred"])

        if "business_day" in df.columns and "hour_business" in df.columns:
            record["business_day"] = pd.to_datetime(df["business_day"])
            record["hour_business"] = df["hour_business"].astype(int)
        else:
            record = add_business_time_columns(record, timestamp_col="ds")
            record["business_day"] = pd.to_datetime(record["business_day"])
            record["hour_business"] = record["hour_business"].astype(int)

        record["source_file"] = str(fpath)
        # Determine source type
        fname = fpath.name.lower()
        if "protocol" in fname or "cutoff_recovery" in fname:
            record["source_type"] = "protocol"
        elif "forecast" in fname:
            record["source_type"] = "forecast"
        elif "fold" in fname:
            record["source_type"] = "fold"
        else:
            record["source_type"] = "other"

        all_rows.append(record)
        logger.info("Loaded %s: %d rows, pred_col=%s", fpath.name, len(record), pred_col)

    if not all_rows:
        logger.error("No prediction files could be loaded")
        print("NO_VALID_PREDICTIONS")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Deduplicate: keep highest priority per (business_day, hour_business)
    priority_map = {"protocol": 0, "forecast": 1, "fold": 2, "other": 3}
    combined["_priority"] = combined["source_type"].map(priority_map).fillna(4)
    combined = combined.sort_values("_priority").drop_duplicates(
        subset=["business_day", "hour_business"], keep="first"
    )
    combined = combined.drop(columns=["_priority"])
    combined = combined.sort_values("ds").reset_index(drop=True)

    # Filter to target range
    target_mask = (combined["ds"] >= args.target_start) & (combined["ds"] < args.target_end)
    target = combined[target_mask].copy()
    logger.info("Consolidated: %d rows (%d after target filter)",
                len(combined), len(target))

    # Calculate coverage
    target_hours = int(
        (pd.Timestamp(args.target_end) - pd.Timestamp(args.target_start)).total_seconds() / 3600
    )
    target_days = target["business_day"].nunique()
    coverage_pct = min(len(target) / target_hours * 100, 100.0) if target_hours > 0 else 0.0

    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["ds", "business_day", "hour_business", "sgdfnet_pred", "source_file", "source_type"]
    target[[c for c in out_cols if c in target.columns]].to_csv(out_path, index=False)
    logger.info("Output saved: %s (%d rows, coverage=%.1f%%)",
                out_path, len(target), coverage_pct)

    # Save coverage report
    report = CoverageReport(
        total_rows=target_hours,
        matched_rows=len(target),
        unmatched_rows=max(0, target_hours - len(target)),
        coverage_pct=round(coverage_pct, 1),
        n_unique_days=target_days,
        date_range=(str(target["ds"].min().date()), str(target["ds"].max().date())),
        source_file=str(out_path),
        fallback_required=coverage_pct < 100.0,
    )
    save_coverage_report(report, out_path.parent)

    # Print summary
    formal_candidate = coverage_pct >= 95.0
    print(f"\nConsolidated SGDFNet Predictions")
    print(f"  File: {out_path}")
    print(f"  Rows: {len(target)}")
    print(f"  Days: {target_days}")
    print(f"  Coverage: {coverage_pct:.1f}%")
    print(f"  formal_candidate: {formal_candidate}")

    if not formal_candidate:
        print("  formal_candidate=false: coverage < 95% — not ready for formal training")


if __name__ == "__main__":
    main()
