#!/usr/bin/env python
"""Monitor SGDFNet Protocol B output directories for new predictions.

Scans configured paths for SGDFNet prediction files and reports coverage
status for target months (Feb 2026, Jan-May 2026).

Usage:
    python scripts/monitor_sgdfnet_protocol_b.py
    python scripts/monitor_sgdfnet_protocol_b.py --out-dir reports/local/deep_final/sgdfnet_monitor
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
    SGDFNetPredictionLoader,
    save_coverage_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitor_sgdfnet_protocol_b")

SEARCH_ROOTS = [
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "outputs",
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "outputs" / "RT916_SpikeMarketLab",
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "reports",
    PROJECT_ROOT / "reports" / "local" / "deep_final" / "sgdfnet_predictions",
]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor SGDFNet Protocol B outputs")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "reports" / "local" / "deep_final" / "sgdfnet_monitor"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scan for prediction files
    candidates = _scan_candidates()
    if not candidates:
        _write_not_ready(out_dir, "No prediction files found")
        print("NOT_READY: No prediction files found")
        return

    # Evaluate each candidate
    evaluated = []
    for path in candidates:
        result = _evaluate(path)
        if result:
            evaluated.append(result)

    if not evaluated:
        _write_not_ready(out_dir, "No readable prediction files")
        print("NOT_READY: No readable prediction files")
        return

    # Find best
    best = max(evaluated, key=lambda r: r["coverage_2026_02"])

    # Write reports
    _write_status_json(best, out_dir)
    _write_status_md(best, candidates, out_dir)

    # Determine readiness
    feb_ready = best["coverage_2026_02"] >= 95.0
    full_ready = best["coverage_2026_01_05"] >= 95.0

    print(f"\nBest: {best['candidate_file']}")
    print(f"  Feb coverage: {best['coverage_2026_02']:.1f}%")
    print(f"  Jan-May coverage: {best['coverage_2026_01_05']:.1f}%")
    print(f"  Rows: {best['n_rows']}, Days: {best['unique_business_days']}")
    print(f"  Pred column: {best['prediction_column']}")

    if feb_ready:
        print("\nREADY_2026_02: Full Feb coverage available")
    if full_ready:
        print("\nREADY_2026_01_05: Full Jan-May coverage available")
    if not feb_ready and not full_ready:
        print(f"\nNOT_READY: Feb={best['coverage_2026_02']:.1f}%, Jan-May={best['coverage_2026_01_05']:.1f}%")

    return best


def _scan_candidates() -> list[Path]:
    candidates = []
    seen = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for fpath in root.rglob("*prediction*.csv"):
            try:
                ap = str(fpath.resolve())
            except Exception:
                continue
            if ap not in seen:
                seen.add(ap)
                candidates.append(fpath)
        for fpath in root.rglob("predictions*.csv"):
            try:
                ap = str(fpath.resolve())
            except Exception:
                continue
            if ap not in seen:
                seen.add(ap)
                candidates.append(fpath)
        # Also look for files in Protocol B output dirs
        for fpath in root.rglob("*.csv"):
            if "sgdfnet" in fpath.name.lower() or "realtime" in fpath.name.lower():
                try:
                    ap = str(fpath.resolve())
                except Exception:
                    continue
                if ap not in seen:
                    seen.add(ap)
                    candidates.append(fpath)
    return sorted(set(candidates))


def _evaluate(path: Path) -> dict[str, Any] | None:
    """Evaluate a candidate prediction file."""
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", nrows=5)
        except (UnicodeDecodeError, pd.errors.ParserError):
            try:
                df = pd.read_csv(path, encoding="gbk", nrows=5)
            except Exception:
                return None

        # Check if it has a prediction-like column
        pred_cols = [c for c in df.columns if any(k in c.lower()
                     for k in ["sgdfnet", "pred", "y_pred", "rt_hat", "forecast"])]
        if not pred_cols:
            return None

        # Full read
        try:
            full = pd.read_csv(path, encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError):
            try:
                full = pd.read_csv(path, encoding="gbk")
            except Exception:
                return None

        # Identify timestamp and prediction columns
        ts_col = next((c for c in ["ds", "timestamp", "time"] if c in full.columns), None)
        if ts_col is None:
            return None

        full["_ts"] = pd.to_datetime(full[ts_col])

        # Find pred column (prefer sgdfnet_pred > y_pred > rt_hat > prediction)
        pred_col = None
        for cand in ["sgdfnet_pred", "y_pred", "rt_hat", "prediction"]:
            if cand in full.columns:
                pred_col = cand
                break
        if pred_col is None:
            pred_col = pred_cols[0]

        # Compute coverage
        feb_mask = (full["_ts"] >= "2026-02-01") & (full["_ts"] < "2026-03-01")
        feb_hours = 28 * 24  # 672
        feb_covered = int(feb_mask.sum())
        feb_coverage = min(feb_covered / feb_hours * 100, 100.0)

        full_mask = (full["_ts"] >= "2026-01-01") & (full["_ts"] < "2026-06-01")
        full_hours = int((pd.Timestamp("2026-06-01") - pd.Timestamp("2026-01-01")).total_seconds() / 3600)
        full_covered = int(full_mask.sum())
        full_coverage = min(full_covered / full_hours * 100, 100.0)

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "candidate_file": str(path),
            "n_rows": len(full),
            "n_cols": len(full.columns),
            "date_min": str(full["_ts"].min().date()),
            "date_max": str(full["_ts"].max().date()),
            "unique_business_days": full["business_day"].nunique() if "business_day" in full.columns else full["_ts"].dt.date.nunique(),
            "coverage_2026_02": round(feb_coverage, 1),
            "coverage_2026_01_05": round(full_coverage, 1),
            "prediction_column": pred_col,
            "is_formal_candidate": feb_covered >= 672 * 0.95,
        }
    except Exception as e:
        logger.debug("Cannot evaluate %s: %s", path, e)
        return None


def _write_status_json(best: dict, out_dir: Path) -> None:
    (out_dir / "monitor_status.json").write_text(
        json.dumps(best, indent=2, default=str), encoding="utf-8"
    )


def _write_status_md(best: dict, candidates: list[Path], out_dir: Path) -> None:
    lines = [
        "# SGDFNet Protocol B Monitor Status",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Files scanned: {len(candidates)}*",
        "",
        "## Best Candidate",
        "",
        f"- **File**: `{best['candidate_file']}`",
        f"- **Rows**: {best['n_rows']}",
        f"- **Date range**: {best['date_min']} ~ {best['date_max']}",
        f"- **Unique days**: {best['unique_business_days']}",
        f"- **Prediction column**: `{best['prediction_column']}`",
        f"- **Feb 2026 coverage**: {best['coverage_2026_02']:.1f}%",
        f"- **Jan-May 2026 coverage**: {best['coverage_2026_01_05']:.1f}%",
        "",
        "## Readiness",
        "",
    ]
    if best["coverage_2026_02"] >= 95.0:
        lines.append("### ✅ READY_2026_02")
    if best["coverage_2026_01_05"] >= 95.0:
        lines.append("### ✅ READY_2026_01_05")
    if best["coverage_2026_02"] < 95.0 and best["coverage_2026_01_05"] < 95.0:
        lines.append(f"### ❌ NOT_READY")
        lines.append(f"Coverage insufficient for formal training.")

    (out_dir / "monitor_status.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Monitor status written to %s", out_dir)


def _write_not_ready(out_dir: Path, reason: str) -> None:
    status = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "verdict": "NOT_READY",
        "reason": reason,
    }
    (out_dir / "monitor_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    lines = [
        "# SGDFNet Protocol B Monitor Status",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"",
        f"## ❌ NOT_READY",
        f"**Reason**: {reason}",
    ]
    (out_dir / "monitor_status.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
