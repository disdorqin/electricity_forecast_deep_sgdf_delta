#!/usr/bin/env python
"""Generate SGDFNet realtime predictions for TrendKnightRT DeepFinal-3.

Calls the SGDFNet Protocol B cutoff walk-forward runner to produce
realtime predictions, then validates and saves them in the format
expected by ``sgdfnet_prediction_loader``.

Usage:
    python scripts/generate_sgdfnet_predictions_for_deep.py \
        --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \
        --data-path ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv \
        --start-date 2026-01-01 \
        --end-date 2026-05-31 \
        --out reports/local/deep_final/sgdfnet_predictions/sgdfnet_realtime_2026_01_05.csv

    # Dry-run (check config only, no training):
    python scripts/generate_sgdfnet_predictions_for_deep.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_sgdfnet_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SGDFNet realtime predictions"
    )
    parser.add_argument("--sgdfnet-root", type=str, default=None,
                        help="Path to SGDFNet project root")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to hourly data CSV")
    parser.add_argument("--start-date", type=str, default="2026-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2026-05-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config path (default: "
                             "SGDFNet/configs/cutoff_recovery_2026_diag_a_prune_actualside.yaml)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check config and exit without running")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        _dry_run(args)
        return

    # ── Resolve SGDFNet root ────────────────────────────────────────
    if args.sgdfnet_root:
        sgdfnet_root = Path(args.sgdfnet_root).resolve()
    else:
        sgdfnet_root = PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "SGDFNet"

    if not sgdfnet_root.exists():
        logger.error("SGDFNet root not found: %s", sgdfnet_root)
        print("NO_REAL_SGDFNET_AVAILABLE")
        return

    # ── Resolve config ──────────────────────────────────────────────
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = sgdfnet_root / "configs" / "cutoff_recovery_2026_diag_a_prune_actualside.yaml"

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        print("NO_REAL_SGDFNET_AVAILABLE")
        return

    # ── Resolve output path ─────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = PROJECT_ROOT / "reports" / "local" / "deep_final" / "sgdfnet_predictions" / "sgdfnet_realtime_2026_01_05.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Import and run SGDFNet protocol ─────────────────────────────
    sys.path.insert(0, str(sgdfnet_root / "src"))
    sys.path.insert(0, str(sgdfnet_root))

    try:
        from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment
    except ImportError as e:
        logger.error("Cannot import SGDFNet runner: %s", e)
        print("NO_REAL_SGDFNET_AVAILABLE")
        return

    logger.info("Running SGDFNet Protocol B with config: %s", config_path)
    logger.info("This may take a while (LightGBM training)...")
    t0 = time.time()

    # Change working directory to SGDFNet root so relative paths in config work
    original_cwd = Path.cwd()
    os.chdir(sgdfnet_root)

    try:
        run_dir = run_protocol_b_cutoff_experiment(str(config_path))
        elapsed = time.time() - t0
        logger.info("SGDFNet run completed in %.1f s. Output dir: %s", elapsed, run_dir)
    except Exception as e:
        logger.error("SGDFNet runner failed: %s", e)
        os.chdir(original_cwd)
        print("NO_REAL_SGDFNET_AVAILABLE")
        return
    finally:
        os.chdir(original_cwd)

    # ── Find prediction output file ─────────────────────────────────
    run_dir = Path(run_dir)
    candidates = list(run_dir.rglob("*prediction*.csv")) + list(run_dir.rglob("*realtime*.csv"))
    if not candidates:
        logger.error("No prediction CSVs found in output dir: %s", run_dir)
        print("NO_REAL_SGDFNET_AVAILABLE")
        return

    pred_file = candidates[0]
    logger.info("Using prediction file: %s", pred_file)

    # ── Load and format predictions ─────────────────────────────────
    df = pd.read_csv(pred_file)

    # Identify columns
    ts_col = next((c for c in ["ds", "timestamp", "time"] if c in df.columns), None)
    pred_col = next((c for c in ["sgdfnet_pred", "y_pred", "prediction", "pred"]
                     if c in df.columns), None)

    if not ts_col or not pred_col:
        logger.error("Cannot identify timestamp/prediction columns in %s. Columns: %s",
                     pred_file, list(df.columns))
        print("NO_REAL_SGDFNET_AVAILABLE")
        return

    # Format output
    out_df = pd.DataFrame()
    out_df["ds"] = pd.to_datetime(df[ts_col])
    out_df["sgdfnet_pred"] = pd.to_numeric(df[pred_col], errors="coerce")

    # Add business day alignment
    from models.deep_sgdf_delta.business_time import add_business_time_columns
    out_df = add_business_time_columns(out_df)

    # Drop NaN predictions
    before = len(out_df)
    out_df = out_df.dropna(subset=["sgdfnet_pred"])
    if len(out_df) < before:
        logger.warning("Dropped %d rows with NaN predictions", before - len(out_df))

    # Save output
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(out_df), out_path)

    # ── Run coverage audit ──────────────────────────────────────────
    from models.deep_sgdf_delta.sgdfnet_prediction_loader import (
        SGDFNetPredictionLoader, save_coverage_report,
    )

    loader = SGDFNetPredictionLoader(require_coverage=95.0)
    _, report = loader._process(out_df)

    save_coverage_report(report, out_path.parent)

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SGDFNet Prediction Generation Complete")
    print("=" * 60)
    print(f"  Source config: {config_path}")
    print(f"  Output: {out_path}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Coverage: {report.coverage_pct:.1f}%")
    print(f"  Unique days: {report.n_unique_days}")
    print(f"  Formal training allowed: {report.coverage_pct >= 95.0}")
    print("=" * 60)

    if report.coverage_pct >= 95.0:
        print(f"\nBest path: {out_path}")
    else:
        print(f"\nCoverage {report.coverage_pct:.1f}% < 95% — predictions insufficient for formal training.")
        print("NO_REAL_SGDFNET_AVAILABLE")


def _dry_run(args: argparse.Namespace) -> None:
    """Check configuration without running training."""
    if args.sgdfnet_root:
        root = Path(args.sgdfnet_root)
    else:
        root = PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "SGDFNet"

    if args.config:
        config = Path(args.config)
    else:
        config = root / "configs" / "cutoff_recovery_2026_diag_a_prune_actualside.yaml"

    print(f"SGDFNet root: {root} {'EXISTS' if root.exists() else 'NOT FOUND'}")
    print(f"Config: {config} {'EXISTS' if config.exists() else 'NOT FOUND'}")

    if root.exists():
        src = root / "src"
        sys.path.insert(0, str(src))
        try:
            from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment
            print("SGDFNet runner: IMPORTABLE")
        except ImportError as e:
            print(f"SGDFNet runner: IMPORT FAILED — {e}")

    print("\nDry-run complete. Use --dry-run to check before running.")


if __name__ == "__main__":
    main()
