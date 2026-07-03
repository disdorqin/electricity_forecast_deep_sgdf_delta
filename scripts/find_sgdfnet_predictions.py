#!/usr/bin/env python
"""Search for existing SGDFNet prediction files across neighbor projects.

Scans configured search paths for CSV/Parquet/XLSX files matching SGDFNet
prediction patterns, evaluates them via ``sgdfnet_prediction_loader``,
and produces a ranked candidate list and best-path recommendation.

Usage:
    python scripts/find_sgdfnet_predictions.py
    python scripts/find_sgdfnet_predictions.py --out-dir reports/local/deep_final/sgdfnet_search
"""
from __future__ import annotations

import csv
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("find_sgdfnet_predictions")

# ── Search configuration ───────────────────────────────────────────────

SEARCH_ROOTS = [
    PROJECT_ROOT,
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp",
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "outputs",
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "SGDFNet",
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "reports",
    PROJECT_ROOT.parent / "all_model_train_and_eval",
    PROJECT_ROOT.parent / "all_model_train_and_eval" / "outputs",
]

FILENAME_KEYWORDS = [
    "sgdfnet", "prediction", "predictions", "cutoff_recovery",
    "protocol_b", "realtime", "rt", "y_pred", "forecast",
]

FILE_EXTENSIONS = [".csv", ".parquet", ".xlsx"]

TARGET_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]

# ── Main ───────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Search for SGDFNet prediction files")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for search report")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: limit to first 200 files")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "reports" / "local" / "deep_final" / "sgdfnet_search"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Find candidate files ────────────────────────────────
    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            logger.info("Search root not found: %s", root)
            continue

        for ext in FILE_EXTENSIONS:
            pattern = f"**/*{ext}"
            matched = 0
            for fpath in sorted(root.glob(pattern)):
                fname_lower = fpath.name.lower()
                if not any(kw in fname_lower for kw in FILENAME_KEYWORDS):
                    continue
                try:
                    abs_path = str(fpath.resolve())
                except (FileNotFoundError, OSError):
                    continue
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)
                try:
                    size_bytes = fpath.stat().st_size
                except (FileNotFoundError, OSError):
                    continue
                candidates.append({
                    "path": abs_path,
                    "size_bytes": size_bytes,
                    "suffix": ext,
                })
                matched += 1
                if args.quick and len(candidates) >= 200:
                    break
            if args.quick and len(candidates) >= 200:
                break
        if args.quick and len(candidates) >= 200:
            break

    logger.info("Found %d candidate files", len(candidates))

    if not candidates:
        _write_empty_report(out_dir)
        print("NO_REAL_SGDFNET_AVAILABLE")
        return

    # ── Step 2: Evaluate each candidate ─────────────────────────────
    loader = SGDFNetPredictionLoader(require_coverage=0.0)  # don't fail yet
    evaluated: list[dict[str, Any]] = []

    for cand in candidates:
        result = _evaluate_candidate(cand, loader)
        if result:
            evaluated.append(result)

    if not evaluated:
        _write_empty_report(out_dir)
        print("NO_REAL_SGDFNET_AVAILABLE")
        return

    # ── Step 3: Rank and write results ──────────────────────────────
    evaluated.sort(key=lambda r: (
        -r.get("coverage_for_target", 0.0),
        -int(r.get("has_pred_col", False)),
        -int(r.get("has_business_time", False)),
        r.get("date_range_start", ""),
        -r.get("size_bytes", 0),
    ))

    _write_csv_report(evaluated, out_dir)
    _write_md_report(evaluated, out_dir)
    _write_best_path(evaluated, out_dir)

    best = evaluated[0]
    print(f"\nBest candidate: {best['path']}")
    print(f"  coverage_for_target: {best.get('coverage_for_target', 0):.1f}%")
    print(f"  has_pred_col: {best.get('has_pred_col', False)}")
    print(f"  date range: {best.get('date_range_start', 'N/A')} ~ {best.get('date_range_end', 'N/A')}")
    print(f"  n_rows: {best.get('n_rows', 0)}")

    if best.get("coverage_for_target", 0) >= 95.0:
        print(f"\nBest path: {best['path']}")
        (out_dir / "best_sgdfnet_prediction_path.txt").write_text(best["path"], encoding="utf-8")
        print(f"Saved to {out_dir / 'best_sgdfnet_prediction_path.txt'}")
    else:
        print("\nNo candidate with coverage >= 95%. Try generating predictions.")
        print("NO_REAL_SGDFNET_AVAILABLE")


def _evaluate_candidate(
    cand: dict[str, Any], loader: SGDFNetPredictionLoader,
) -> dict[str, Any] | None:
    """Try to load a candidate file and evaluate it."""
    path = cand["path"]
    suffix = cand["suffix"]

    try:
        if suffix == ".csv":
            try:
                raw = pd.read_csv(path, encoding="utf-8-sig", nrows=100)
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    raw = pd.read_csv(path, encoding="gbk", nrows=100)
                except Exception:
                    return None
        elif suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif suffix == ".xlsx":
            raw = pd.read_excel(path)
        else:
            return None

        if raw.empty:
            return None
    except Exception as e:
        logger.debug("Cannot read %s: %s", path, e)
        return None

    # Check if loader can identify columns
    has_ds = any(c in raw.columns for c in ["ds", "timestamp", "time", "时刻"])
    has_pred = any(c in raw.columns for c in
                   ["sgdfnet_pred", "pred", "prediction", "y_pred", "rt_pred"])
    has_business_time = ("business_day" in raw.columns
                         and "hour_business" in raw.columns)

    if not has_pred:
        return None

    # Try full load and coverage computation
    try:
        if suffix == ".csv":
            try:
                full = pd.read_csv(path, encoding="utf-8-sig")
            except (UnicodeDecodeError, pd.errors.ParserError):
                full = pd.read_csv(path, encoding="gbk")
        else:
            return None

        pred_df, report = loader._process(full)
        coverage_for_target = _compute_target_coverage(pred_df)
    except Exception as e:
        logger.debug("Cannot process %s: %s", path, e)
        coverage_for_target = 0.0

    return {
        "path": path,
        "size_bytes": cand["size_bytes"],
        "n_rows": len(raw),
        "n_cols": len(raw.columns),
        "columns": list(raw.columns[:15]),
        "has_ds_col": has_ds,
        "has_pred_col": has_pred,
        "has_business_time": has_business_time,
        "date_range_start": str(pred_df["ds"].min().date()) if not pred_df.empty else "N/A",
        "date_range_end": str(pred_df["ds"].max().date()) if not pred_df.empty else "N/A",
        "n_unique_days": report.n_unique_days,
        "coverage_pct": round(report.coverage_pct, 1),
        "coverage_for_target": round(coverage_for_target, 1),
    }


def _compute_target_coverage(pred_df: pd.DataFrame) -> float:
    """Compute coverage percentage across target months 2026-01~05."""
    if pred_df.empty or "business_day" not in pred_df.columns:
        return 0.0
    all_target_hours = 0
    covered_hours = 0
    for m in TARGET_MONTHS:
        start = pd.Timestamp(m)
        end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
        month_hours = int((end - start).total_seconds() / 3600)
        all_target_hours += month_hours
    covered_hours = len(pred_df)
    return (covered_hours / all_target_hours * 100) if all_target_hours > 0 else 0.0


def _write_csv_report(evaluated: list[dict], out_dir: Path) -> None:
    path = out_dir / "candidate_sgdfnet_predictions.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "size_bytes", "n_rows", "n_cols", "has_ds_col",
            "has_pred_col", "has_business_time", "date_range_start",
            "date_range_end", "n_unique_days", "coverage_pct",
            "coverage_for_target",
        ])
        writer.writeheader()
        for r in evaluated:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    logger.info("Written %d candidates to %s", len(evaluated), path)


def _write_md_report(evaluated: list[dict], out_dir: Path) -> None:
    lines = [
        "# SGDFNet Prediction Search Report",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Candidates found: {len(evaluated)}*",
        "",
        "## Top Candidates",
        "",
        "| Rank | File | Rows | Date Range | Pred Col | Bus Time | Coverage | Target Coverage |",
        "|------|------|------|------------|----------|----------|----------|-----------------|",
    ]
    for i, r in enumerate(evaluated[:20]):
        fname = Path(r["path"]).name
        lines.append(
            f"| {i+1} | `{fname}` | {r.get('n_rows', '?')} "
            f"| {r.get('date_range_start', '?')}~{r.get('date_range_end', '?')} "
            f"| {r.get('has_pred_col', False)} "
            f"| {r.get('has_business_time', False)} "
            f"| {r.get('coverage_pct', 0)}% "
            f"| {r.get('coverage_for_target', 0)}% |"
        )

    path = out_dir / "sgdfnet_search_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Written report to %s", path)


def _write_best_path(evaluated: list[dict], out_dir: Path) -> None:
    if not evaluated:
        return
    best = evaluated[0]
    if best.get("coverage_for_target", 0) >= 95.0:
        (out_dir / "best_sgdfnet_prediction_path.txt").write_text(
            best["path"], encoding="utf-8"
        )
        logger.info("Best path: %s", best["path"])


def _write_empty_report(out_dir: Path) -> None:
    (out_dir / "sgdfnet_search_report.md").write_text(
        "# SGDFNet Prediction Search Report\n\n**No candidates found.**\n",
        encoding="utf-8",
    )
    (out_dir / "best_sgdfnet_prediction_path.txt").write_text(
        "NO_REAL_SGDFNET_AVAILABLE", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
