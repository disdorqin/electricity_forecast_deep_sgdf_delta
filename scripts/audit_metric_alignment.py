#!/usr/bin/env python
"""Metric Alignment Audit — cross-module sMAPE consistency check.

Compares predictions from multiple modules to verify metric alignment:
  1. DeepFinal eval  — reports/local/deep_final/predictions_2026_02.csv
  2. DeltaSupply     — artifacts/delta_supply/exp_2026_02/predictions.csv
  3. Raw data        — ../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv

For each source, computes row_count, date_range, da_anchor stats, rt_actual
stats, and DA anchor sMAPE_floor50 using the CANONICAL formula from
models.deep_sgdf_delta.metrics.

Then finds the common intersection (merge on ds timestamp) and computes
sMAPE on common rows.

Verdict:
  PASS — common intersection sMAPE differs <= 0.1 pp
  WARN — differs <= 1.0 pp
  FAIL — differs > 1.0 pp

Output (to --out-dir, default reports/local/risk_modules/metric_alignment/):
  metric_alignment_summary.json
  metric_alignment_rows.csv
  metric_alignment_report.md

Usage:
    python scripts/audit_metric_alignment.py \\
        --out-dir reports/local/risk_modules/metric_alignment
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.deep_sgdf_delta.metrics import smape_floor50
from models.deep_sgdf_delta.realtime_column_mapping import rename_chinese_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Source paths ─────────────────────────────────────────────────────

DEFAULT_DEEP_FINAL_PATH = "reports/local/deep_final/predictions_2026_02.csv"
DEFAULT_DELTA_SUPPLY_PATH = "artifacts/delta_supply/exp_2026_02/predictions.csv"
DEFAULT_RAW_DATA_PATH = "../electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv"


# ── Loaders ──────────────────────────────────────────────────────────

def load_deep_final(path: Path) -> pd.DataFrame | None:
    """Load DeepFinal predictions CSV."""
    if not path.exists():
        logger.warning("DeepFinal predictions not found: %s", path)
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("DeepFinal loaded: %d rows from %s", len(df), path)
    return df


def load_delta_supply(path: Path) -> pd.DataFrame | None:
    """Load DeltaSupply predictions CSV."""
    if not path.exists():
        logger.warning("DeltaSupply predictions not found: %s", path)
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("DeltaSupply loaded: %d rows from %s", len(df), path)
    return df


def load_raw_data(path: Path) -> pd.DataFrame | None:
    """Load raw Shandong PMOS hourly CSV with Chinese column renaming."""
    if not path.exists():
        logger.warning("Raw data not found: %s", path)
        return None
    # Try GBK / gb18030 first (Chinese Windows CSV), then utf-8
    df = None
    for enc in ("gbk", "gb18030", "utf-8", "utf-8-sig"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("Raw data loaded with encoding=%s: %d rows from %s", enc, len(df), path)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        logger.error("Cannot decode raw data file: %s", path)
        return None

    # Rename Chinese columns to canonical English names
    df = rename_chinese_columns(df)
    return df


# ── Stats computation ────────────────────────────────────────────────

def compute_source_stats(df: pd.DataFrame, source_name: str, *, rt_col: str = "rt_actual") -> dict:
    """Compute per-source statistics.

    Expected columns after normalisation: ds, da_anchor, rt_actual (or rt_col).
    """
    stats: dict = {"source": source_name, "row_count": len(df)}

    # Date range
    if "ds" in df.columns:
        ds = pd.to_datetime(df["ds"], errors="coerce")
        stats["date_min"] = str(ds.min())
        stats["date_max"] = str(ds.max())
    else:
        stats["date_min"] = None
        stats["date_max"] = None

    # da_anchor stats
    if "da_anchor" in df.columns:
        da = df["da_anchor"].dropna().astype(float)
        if len(da) > 0:
            stats["da_anchor_mean"] = round(float(da.mean()), 4)
            stats["da_anchor_std"] = round(float(da.std()), 4)
            stats["da_anchor_min"] = round(float(da.min()), 4)
            stats["da_anchor_max"] = round(float(da.max()), 4)
        else:
            stats["da_anchor_mean"] = None
            stats["da_anchor_std"] = None
            stats["da_anchor_min"] = None
            stats["da_anchor_max"] = None
    else:
        stats["da_anchor_mean"] = None
        stats["da_anchor_std"] = None
        stats["da_anchor_min"] = None
        stats["da_anchor_max"] = None

    # rt_actual stats
    if rt_col in df.columns:
        rt = df[rt_col].dropna().astype(float)
        if len(rt) > 0:
            stats["rt_actual_mean"] = round(float(rt.mean()), 4)
            stats["rt_actual_std"] = round(float(rt.std()), 4)
            stats["rt_actual_min"] = round(float(rt.min()), 4)
            stats["rt_actual_max"] = round(float(rt.max()), 4)
        else:
            stats["rt_actual_mean"] = None
            stats["rt_actual_std"] = None
            stats["rt_actual_min"] = None
            stats["rt_actual_max"] = None
    else:
        stats["rt_actual_mean"] = None
        stats["rt_actual_std"] = None
        stats["rt_actual_min"] = None
        stats["rt_actual_max"] = None

    return stats


def compute_smape_for_source(
    df: pd.DataFrame,
    source_name: str,
    *,
    y_true_col: str = "rt_actual",
    y_pred_col: str = "rt_pred",
) -> float | None:
    """Compute sMAPE_floor50 for a source using the canonical formula.

    Returns None if required columns are missing or no valid rows.
    """
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        logger.warning(
            "%s: missing columns for sMAPE (%s, %s)", source_name, y_true_col, y_pred_col
        )
        return None
    valid = df.dropna(subset=[y_true_col, y_pred_col])
    if valid.empty:
        logger.warning("%s: no valid rows for sMAPE computation", source_name)
        return None
    yt = valid[y_true_col].to_numpy(dtype=float)
    yp = valid[y_pred_col].to_numpy(dtype=float)
    return round(smape_floor50(yt, yp), 4)


# ── Common intersection ──────────────────────────────────────────────

def find_common_intersection(
    dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Merge all sources on ds timestamp to find the common intersection.

    Returns the merged DataFrame or None if fewer than 2 sources have ds.
    """
    ds_dfs = {name: df for name, df in dfs.items() if "ds" in df.columns}
    if len(ds_dfs) < 2:
        logger.warning("Fewer than 2 sources have 'ds' column; cannot compute intersection")
        return None

    names = list(ds_dfs.keys())
    merged = ds_dfs[names[0]][["ds"]].copy()
    merged["ds"] = pd.to_datetime(merged["ds"], errors="coerce")
    merged = merged.dropna(subset=["ds"])

    for name in names[1:]:
        other = ds_dfs[name][["ds"]].copy()
        other["ds"] = pd.to_datetime(other["ds"], errors="coerce")
        other = other.dropna(subset=["ds"])
        merged = merged.merge(other, on="ds", how="inner")

    logger.info("Common intersection: %d rows across %d sources", len(merged), len(ds_dfs))
    return merged


def compute_common_smape(
    dfs: dict[str, pd.DataFrame],
    common_ds: pd.DataFrame,
    *,
    y_true_col: str = "rt_actual",
    y_pred_col: str = "rt_pred",
) -> dict:
    """Compute sMAPE on the common intersection for each source.

    For each source, merge with common_ds on ds, then compute sMAPE_floor50
    using the source's rt_actual vs rt_pred (or da_anchor as fallback for y_pred).
    """
    result: dict = {}
    common_timestamps = set(pd.to_datetime(common_ds["ds"], errors="coerce").dropna())

    for name, df in dfs.items():
        if "ds" not in df.columns:
            continue
        df_copy = df.copy()
        df_copy["ds"] = pd.to_datetime(df_copy["ds"], errors="coerce")
        df_common = df_copy[df_copy["ds"].isin(common_timestamps)]

        if df_common.empty:
            result[name] = {"common_rows": 0, "smape_common": None}
            continue

        # Determine y_true and y_pred columns
        yt_col = y_true_col if y_true_col in df_common.columns else None
        yp_col = y_pred_col if y_pred_col in df_common.columns else None

        if yt_col is None:
            result[name] = {"common_rows": len(df_common), "smape_common": None}
            continue

        # If no rt_pred, use da_anchor as the "prediction" (baseline)
        if yp_col is None:
            if "da_anchor" in df_common.columns:
                yp_col = "da_anchor"
            else:
                result[name] = {"common_rows": len(df_common), "smape_common": None}
                continue

        valid = df_common.dropna(subset=[yt_col, yp_col])
        if valid.empty:
            result[name] = {"common_rows": len(df_common), "smape_common": None}
            continue

        yt = valid[yt_col].to_numpy(dtype=float)
        yp = valid[yp_col].to_numpy(dtype=float)
        result[name] = {
            "common_rows": len(valid),
            "smape_common": round(smape_floor50(yt, yp), 4),
        }

    return result


# ── Verdict ──────────────────────────────────────────────────────────

def compute_verdict(common_smape_values: list[float]) -> tuple[str, float]:
    """Compute verdict based on sMAPE spread across sources on common rows.

    Returns (verdict, spread_pp).
    """
    if len(common_smape_values) < 2:
        return "INSUFFICIENT", 0.0

    spread = max(common_smape_values) - min(common_smape_values)
    # spread is already in percent scale (canonical formula uses 200x multiplier)
    spread_pp = round(spread, 4)

    if spread_pp <= 0.1:
        verdict = "PASS"
    elif spread_pp <= 1.0:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return verdict, spread_pp


# ── Report generation ────────────────────────────────────────────────

def build_comparison_rows(
    source_stats: list[dict],
    source_smapes: dict[str, float | None],
    common_smape: dict[str, dict],
) -> pd.DataFrame:
    """Build a comparison table DataFrame."""
    rows = []
    for stats in source_stats:
        name = stats["source"]
        rows.append({
            "source": name,
            "row_count": stats["row_count"],
            "date_min": stats.get("date_min"),
            "date_max": stats.get("date_max"),
            "da_anchor_mean": stats.get("da_anchor_mean"),
            "da_anchor_std": stats.get("da_anchor_std"),
            "rt_actual_mean": stats.get("rt_actual_mean"),
            "rt_actual_std": stats.get("rt_actual_std"),
            "smape_floor50_full": source_smapes.get(name),
            "common_rows": common_smape.get(name, {}).get("common_rows", 0),
            "smape_floor50_common": common_smape.get(name, {}).get("smape_common"),
        })
    return pd.DataFrame(rows)


def generate_report_md(
    source_stats: list[dict],
    source_smapes: dict[str, float | None],
    common_smape: dict[str, dict],
    verdict: str,
    spread_pp: float,
    comparison_df: pd.DataFrame,
) -> str:
    """Generate the markdown audit report."""
    lines = [
        "# Metric Alignment Audit Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Canonical formula:** `models.deep_sgdf_delta.metrics.smape_floor50` (multiplier=200, percent scale)",
        "",
        "## Per-Source Statistics",
        "",
        "| Source | Rows | Date Min | Date Max | DA mean | DA std | RT mean | RT std |",
        "|--------|------|----------|----------|---------|--------|---------|--------|",
    ]
    for stats in source_stats:
        lines.append(
            f"| {stats['source']} | {stats['row_count']} "
            f"| {stats.get('date_min', 'N/A')} | {stats.get('date_max', 'N/A')} "
            f"| {stats.get('da_anchor_mean', 'N/A')} | {stats.get('da_anchor_std', 'N/A')} "
            f"| {stats.get('rt_actual_mean', 'N/A')} | {stats.get('rt_actual_std', 'N/A')} |"
        )

    lines.extend([
        "",
        "## sMAPE_floor50 Comparison",
        "",
        "| Source | Full sMAPE | Common Rows | Common sMAPE |",
        "|--------|-----------|-------------|-------------|",
    ])
    for stats in source_stats:
        name = stats["source"]
        full_smape = source_smapes.get(name)
        cs = common_smape.get(name, {})
        common_rows = cs.get("common_rows", 0)
        common_val = cs.get("smape_common")
        lines.append(
            f"| {name} "
            f"| {f'{full_smape:.4f}' if full_smape is not None else 'N/A'} "
            f"| {common_rows} "
            f"| {f'{common_val:.4f}' if common_val is not None else 'N/A'} |"
        )

    common_vals = [
        cs["smape_common"]
        for cs in common_smape.values()
        if cs.get("smape_common") is not None
    ]
    if len(common_vals) >= 2:
        lines.extend([
            "",
            f"- Common sMAPE min: {min(common_vals):.4f}",
            f"- Common sMAPE max: {max(common_vals):.4f}",
            f"- Spread: {spread_pp:.4f} pp",
        ])

    lines.extend([
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"Spread threshold: PASS <= 0.1 pp, WARN <= 1.0 pp, FAIL > 1.0 pp",
        "",
    ])

    if verdict == "PASS":
        lines.append("All modules are metric-aligned within 0.1 pp on the common intersection.")
    elif verdict == "WARN":
        lines.append("Modules show mild metric divergence (<= 1.0 pp). Review feature alignment.")
    elif verdict == "FAIL":
        lines.append("Modules are NOT metric-aligned (> 1.0 pp divergence). Investigate root cause.")
    else:
        lines.append("Insufficient sources for cross-module comparison.")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Metric Alignment Audit")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/local/risk_modules/metric_alignment",
    )
    parser.add_argument("--deep-final-path", type=str, default=DEFAULT_DEEP_FINAL_PATH)
    parser.add_argument("--delta-supply-path", type=str, default=DEFAULT_DELTA_SUPPLY_PATH)
    parser.add_argument("--raw-data-path", type=str, default=DEFAULT_RAW_DATA_PATH)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Metric Alignment Audit")
    logger.info("=" * 60)

    # ── Load sources ─────────────────────────────────────────────────
    deep_final_df = load_deep_final(PROJECT_ROOT / args.deep_final_path)
    delta_supply_df = load_delta_supply(PROJECT_ROOT / args.delta_supply_path)
    raw_df = load_raw_data(PROJECT_ROOT / args.raw_data_path)

    sources: dict[str, pd.DataFrame] = {}
    if deep_final_df is not None:
        sources["DeepFinal"] = deep_final_df
    if delta_supply_df is not None:
        sources["DeltaSupply"] = delta_supply_df
    if raw_df is not None:
        sources["RawData"] = raw_df

    if len(sources) == 0:
        logger.error("No sources loaded. Cannot proceed.")
        sys.exit(1)

    logger.info("Loaded %d sources: %s", len(sources), list(sources.keys()))

    # ── Per-source stats ─────────────────────────────────────────────
    source_stats: list[dict] = []
    source_smapes: dict[str, float | None] = {}

    for name, df in sources.items():
        stats = compute_source_stats(df, name)
        source_stats.append(stats)
        logger.info(
            "  %s: rows=%d, da_mean=%s, rt_mean=%s",
            name, stats["row_count"], stats.get("da_anchor_mean"), stats.get("rt_actual_mean"),
        )

        # sMAPE on full source
        smape_val = compute_smape_for_source(df, name)
        source_smapes[name] = smape_val
        if smape_val is not None:
            logger.info("  %s: full sMAPE_floor50 = %.4f", name, smape_val)

    # ── Common intersection ──────────────────────────────────────────
    common_ds = find_common_intersection(sources)
    common_smape: dict[str, dict] = {}

    if common_ds is not None and not common_ds.empty:
        common_smape = compute_common_smape(sources, common_ds)
        for name, cs in common_smape.items():
            if cs.get("smape_common") is not None:
                logger.info(
                    "  %s: common_rows=%d, common_sMAPE=%.4f",
                    name, cs["common_rows"], cs["smape_common"],
                )
    else:
        logger.warning("No common intersection found across sources")

    # ── Verdict ──────────────────────────────────────────────────────
    common_vals = [
        cs["smape_common"]
        for cs in common_smape.values()
        if cs.get("smape_common") is not None
    ]
    verdict, spread_pp = compute_verdict(common_vals)
    logger.info("Verdict: %s (spread=%.4f pp)", verdict, spread_pp)

    # ── Build comparison table ───────────────────────────────────────
    comparison_df = build_comparison_rows(source_stats, source_smapes, common_smape)

    # ── Write outputs ────────────────────────────────────────────────

    # 1. JSON summary
    summary = {
        "audit_timestamp": datetime.now().isoformat(),
        "canonical_formula": "models.deep_sgdf_delta.metrics.smape_floor50",
        "multiplier": 200,
        "floor": 50.0,
        "sources_loaded": list(sources.keys()),
        "per_source_stats": source_stats,
        "per_source_smape_full": {k: v for k, v in source_smapes.items()},
        "common_intersection_rows": len(common_ds) if common_ds is not None else 0,
        "common_smape": common_smape,
        "verdict": verdict,
        "spread_pp": spread_pp,
    }
    json_path = out_dir / "metric_alignment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    # 2. CSV comparison table
    csv_path = out_dir / "metric_alignment_rows.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s", csv_path)

    # 3. Markdown report
    report_md = generate_report_md(source_stats, source_smapes, common_smape, verdict, spread_pp, comparison_df)
    md_path = out_dir / "metric_alignment_report.md"
    md_path.write_text(report_md, encoding="utf-8")
    logger.info("Wrote %s", md_path)

    logger.info("=" * 60)
    logger.info("Audit complete. Verdict: %s", verdict)
    logger.info("=" * 60)

    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
