#!/usr/bin/env python
"""Build unified teacher prediction pack from SGDFNet / RT916 / TimeMixer.

Usage:
    python scripts/build_teacher_prediction_pack.py \\
        --source-repo-root ../electricity_forecast_model2.0_exp \\
        --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \\
        --start-date 2026-01-01 --end-date 2026-03-31 \\
        --teachers sgdfnet,rt916,timemixer

    python scripts/build_teacher_prediction_pack.py --help
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.teacher_registry import TeacherRegistry, TEACHER_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_teacher_prediction_pack")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build unified teacher prediction pack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Teachers: sgdfnet, rt916, timemixer\n"
               "Each teacher is loaded independently. Missing teachers are marked unavailable.",
    )
    parser.add_argument("--source-repo-root", type=str, default=None,
                        help="Path to electricity_forecast_model2.0_exp root")
    parser.add_argument("--sgdfnet-root", type=str, default=None,
                        help="Path to SGDFNet project root")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to raw data file (for reference)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--teachers", type=str, default="sgdfnet,rt916,timemixer",
                        help="Comma-separated teacher names (default: sgdfnet,rt916,timemixer)")
    parser.add_argument("--out-dir", type=str, default="reports/local/phase3/teachers",
                        help="Output directory")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Quick smoke test with limited data")
    return parser.parse_args()


def main():
    args = parse_args()
    
    teachers = [t.strip() for t in args.teachers.split(",")]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building teacher prediction pack")
    logger.info("  Teachers: %s", teachers)
    logger.info("  Period: %s to %s", args.start_date, args.end_date)
    
    registry = TeacherRegistry()
    statuses = registry.load_all(
        teachers=teachers,
        source_repo_root=args.source_repo_root,
        sgdfnet_root=args.sgdfnet_root,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    # Write merged predictions (long format)
    merged = registry.get_merged_predictions()
    if merged is not None:
        merged.to_csv(out_dir / "teacher_predictions.csv", index=False, encoding="utf-8-sig")
        logger.info("Teacher predictions -> %s (%d rows)", out_dir / "teacher_predictions.csv", len(merged))
    else:
        pd.DataFrame().to_csv(out_dir / "teacher_predictions.csv", index=False, encoding="utf-8-sig")
        logger.warning("No teacher predictions available")
    
    # Write wide format
    wide = registry.get_wide_predictions()
    if wide is not None:
        wide.to_csv(out_dir / "teacher_predictions_wide.csv", index=False, encoding="utf-8-sig")
        logger.info("Wide predictions -> %s", out_dir / "teacher_predictions_wide.csv")
    
    # Write status report
    report = {
        "timestamp": datetime.now().isoformat(),
        "teachers": registry.summary(),
        "period": {"start": args.start_date, "end": args.end_date},
    }
    with open(out_dir / "teacher_status.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Teacher status -> %s", out_dir / "teacher_status.json")
    
    # Summary
    available = sum(1 for s in statuses.values() if s.availability == "available")
    logger.info("Teachers loaded: %d/%d available", available, len(teachers))


if __name__ == "__main__":
    main()
