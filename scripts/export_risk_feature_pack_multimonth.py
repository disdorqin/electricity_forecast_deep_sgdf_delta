#!/usr/bin/env python
"""Multi-month risk feature pack export.

Combines DeltaSupply + Spike + Negative backtest predictions across multiple
months into a single unified CSV with one row per (business_day, hour_business).

Reads monthly prediction CSVs from each module's backtest root directory,
merges them, and produces a unified risk feature pack compatible with
RISK_FEATURE_PACK_CONTRACT.md plus new multi-month fields.

Produces:
  <out-dir>/
    risk_feature_pack.csv       Unified risk features (all months combined)
    manifest.json               Column list, row count, version, alignment status
    monthly_manifest.csv        Per-month row counts and module status summary

Usage:
    python scripts/export_risk_feature_pack_multimonth.py \
      --delta-supply-root reports/local/risk_modules/delta_supply_risk_backtest_2026_01_05 \
      --spike-root reports/local/risk_modules/spike_risk_backtest_2026_01_05 \
      --negative-root reports/local/risk_modules/negative_risk_backtest_2026_01_05 \
      --metric-alignment-status PASS \
      --out-dir reports/local/risk_modules/risk_feature_pack_2026_01_05 \
      --mode online

Modes:
  online  -- NO rt_actual / y_true columns (safe for production)
  eval    -- includes y_true columns for backtesting
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_risk_feature_pack_multimonth")

# -- Constants ----------------------------------------------------------------

RISK_FEATURE_VERSION = "v1.1.0"
THRESHOLD_VERSION = "v1.0.0"

KEY_COLUMNS = ["business_day", "hour_business"]

# Canonical output columns (online mode).
ONLINE_COLUMNS = [
    "business_day",
    "hour_business",
    "ds",
    "target_month",
    # DeltaSupply deviation risk
    "deviation_up_prob",
    "deviation_down_prob",
    "deviation_large_abs_prob",
    "deviation_risk_score",
    # Spike risk
    "spike_prob",
    "extreme_spike_prob",
    "relative_spike_prob",
    "spike_risk_score",
    # Negative risk
    "negative_prob",
    "deep_negative_prob",
    "relative_down_prob",
    "negative_risk_score",
    # Module status
    "module_status_delta_supply",
    "module_status_spike",
    "module_status_negative",
    # Metadata
    "threshold_version",
    "risk_feature_version",
    "metric_alignment_status",
    "metric_alignment_warning_reason",
]

EVAL_EXTRA_COLUMNS = ["y_true"]

# Source column mappings (same as single-month export).
DELTA_SUPPLY_COL_MAP = {
    "upward_deviation_prob": "deviation_up_prob",
    "downward_deviation_prob": "deviation_down_prob",
    "large_abs_deviation_prob": "deviation_large_abs_prob",
    "deviation_risk_score": "deviation_risk_score",
}

SPIKE_RISK_COL_MAP = {
    "spike_prob": "spike_prob",
    "extreme_spike_prob": "extreme_spike_prob",
    "relative_spike_prob": "relative_spike_prob",
    "spike_risk_score": "spike_risk_score",
}

NEGATIVE_RISK_COL_MAP = {
    "negative_prob": "negative_prob",
    "deep_negative_prob": "deep_negative_prob",
    "relative_down_prob": "relative_down_prob",
    "negative_risk_score": "negative_risk_score",
}

# DeltaSupply risk columns (for NaN fill on NO-GO months).
DELTA_SUPPLY_RISK_COLS = [
    "deviation_up_prob",
    "deviation_down_prob",
    "deviation_large_abs_prob",
    "deviation_risk_score",
]

SPIKE_RISK_COLS = [
    "spike_prob",
    "extreme_spike_prob",
    "relative_spike_prob",
    "spike_risk_score",
]

NEGATIVE_RISK_COLS = [
    "negative_prob",
    "deep_negative_prob",
    "relative_down_prob",
    "negative_risk_score",
]


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-month risk feature pack export from backtest roots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  risk_feature_pack.csv   Unified risk features (one row per business_hour)
  manifest.json            Column list, row count, version, alignment status
  monthly_manifest.csv     Per-month row counts and module status summary

Modes:
  online  -- No rt_actual / y_true columns (production-safe)
  eval    -- Includes y_true columns for backtesting
""",
    )
    parser.add_argument(
        "--delta-supply-root", type=str, required=True,
        help="Root directory of DeltaSupply backtest (contains monthly prediction CSVs)",
    )
    parser.add_argument(
        "--spike-root", type=str, required=True,
        help="Root directory of SpikeRisk backtest (contains monthly prediction CSVs)",
    )
    parser.add_argument(
        "--negative-root", type=str, required=True,
        help="Root directory of NegativeRisk backtest (contains monthly prediction CSVs)",
    )
    parser.add_argument(
        "--metric-alignment-status", type=str, required=True,
        choices=["PASS", "WARN", "FAIL"],
        help=(
            "Metric alignment audit status. "
            "PASS: exact alignment. "
            "WARN: computational alignment passed but data completeness warning exists (allowed to export). "
            "FAIL: alignment failed (export forbidden)."
        ),
    )
    parser.add_argument(
        "--metric-alignment-warning-reason", type=str, default="",
        help="Warning reason when metric-alignment-status is WARN. Recorded in manifest.",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for the risk feature pack",
    )
    parser.add_argument(
        "--mode", type=str, default="online",
        choices=["online", "eval"],
        help="Output mode: online (no y_true) or eval (includes y_true)",
    )
    return parser.parse_args()


# -- Helpers ------------------------------------------------------------------

def _resolve_path(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _discover_monthly_csvs(root: Path) -> dict[str, Path]:
    """Discover monthly prediction CSVs in a backtest root directory.

    Looks for files matching patterns like:
      - predictions_2026_01.csv
      - predictions_2026_02.csv
      - monthly_2026_01_predictions.csv

    Returns a dict mapping target_month (YYYY-MM) -> Path.
    """
    month_map: dict[str, Path] = {}
    if not root.exists():
        logger.warning("Backtest root does not exist: %s", root)
        return month_map

    for csv_path in sorted(root.glob("*.csv")):
        # Try to extract YYYY_MM or YYYY-MM from filename.
        name = csv_path.stem
        # Pattern: predictions_YYYY_MM or *_YYYY_MM_*
        parts = name.replace("-", "_").split("_")
        for i, part in enumerate(parts):
            if len(part) == 4 and part.isdigit():
                # Check if next part is a 2-digit month.
                if i + 1 < len(parts) and len(parts[i + 1]) == 2 and parts[i + 1].isdigit():
                    month_key = f"{part}-{parts[i + 1]}"
                    month_map[month_key] = csv_path
                    break

    logger.info("Discovered %d monthly CSVs in %s", len(month_map), root)
    return month_map


def _load_champion_summary(root: Path) -> Optional[dict]:
    """Load champion_summary.json from a backtest root if it exists."""
    summary_path = root / "champion_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_verdict_summary(root: Path) -> Optional[dict]:
    """Load verdict summary from a backtest root.

    Reading order:
      1. champion_summary.json, if exists
      2. verdict.json, if exists
      3. fallback None

    Returns the loaded dict or None.
    """
    summary = _load_champion_summary(root)
    if summary is not None:
        return summary
    verdict_path = root / "verdict.json"
    if verdict_path.exists():
        with open(verdict_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _normalize_verdict_to_status(verdict: str | None) -> str:
    """Map an overall or per-month verdict string to a module status.

    Mapping rules:
      *_CHAMPION / *_STRONG / *_ACCEPTABLE / *_GO  -> GO
      *_LOW_VALUE                                   -> LOW_VALUE
      *_NO_GO / NOGO                                -> NO-GO
      INSUFFICIENT_*                                -> INSUFFICIENT
      None / unknown                                -> UNKNOWN
    """
    if verdict is None:
        return "UNKNOWN"
    v = verdict.upper().replace("-", "_").replace(" ", "_")
    # Check NO-GO first (before GO suffix, since NO_GO ends with _GO).
    if "NO_GO" in v or v in ("NOGO", "NO-GO"):
        return "NO-GO"
    # LOW_VALUE before GO (since LOW_VALUE doesn't end with _GO, but be safe).
    if "LOW_VALUE" in v or v == "LOW-VALUE":
        return "LOW_VALUE"
    # INSUFFICIENT
    if v.startswith("INSUFFICIENT"):
        return "INSUFFICIENT"
    # Check for GO-like verdicts
    for suffix in ("_CHAMPION", "_STRONG", "_ACCEPTABLE", "_GO"):
        if v.endswith(suffix) or v == suffix.lstrip("_"):
            return "GO"
    # Direct matches
    if v in ("GO", "ACCEPTABLE", "STRONG", "CHAMPION"):
        return "GO"
    return "UNKNOWN"


def _get_monthly_verdicts(champion_summary: Optional[dict]) -> dict[str, str]:
    """Extract per-month verdicts from champion summary.

    Returns dict mapping target_month -> verdict string.
    """
    if champion_summary is None:
        return {}
    return champion_summary.get("monthly_verdicts", {})


def _is_nogo_verdict(verdict: str) -> bool:
    """Check if a verdict string indicates NO-GO status."""
    if verdict is None:
        return False
    v = verdict.upper().replace(" ", "").replace("-", "").replace("_", "")
    return v in ("NOGO", "NOVALUE", "STRONGNOGO")


def _load_module_predictions(path: Path, module_name: str) -> pd.DataFrame:
    """Load a module's predictions CSV, validating it exists."""
    if not path.exists():
        logger.error("%s predictions file not found: %s", module_name, path)
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d rows from %s (%s)", len(df), module_name, path)
    return df


def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure business_day is datetime and hour_business is int."""
    df = df.copy()
    if "business_day" in df.columns:
        df["business_day"] = pd.to_datetime(df["business_day"]).dt.normalize()
    if "hour_business" in df.columns:
        df["hour_business"] = df["hour_business"].astype(int)
    return df


def _rename_available(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename columns that exist in df according to col_map."""
    rename = {src: dst for src, dst in col_map.items() if src in df.columns and src != dst}
    if rename:
        df = df.rename(columns=rename)
    return df


def _select_target_columns(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """Select only the target columns that exist; fill missing with NaN."""
    out = pd.DataFrame()
    for col in target_cols:
        if col in df.columns:
            out[col] = df[col].values
        else:
            logger.warning("Column '%s' not found -- filling with NaN", col)
            out[col] = np.nan
    return out


def _extract_target_month_from_predictions(df: pd.DataFrame, fallback_month: str) -> pd.Series:
    """Extract target_month from predictions DataFrame.

    Uses 'period' column if available (format YYYY-MM), otherwise uses fallback.
    """
    if "period" in df.columns:
        return df["period"].astype(str).str[:7]
    return pd.Series([fallback_month] * len(df), index=df.index)


# -- Core logic ---------------------------------------------------------------

def _load_and_prepare_module(
    root: Path,
    module_name: str,
    col_map: dict,
    risk_cols: list[str],
    monthly_verdicts: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load all monthly CSVs for a module, rename columns, tag with target_month.

    Returns:
      - DataFrame with columns: business_day, hour_business, ds, target_month,
        plus the renamed risk columns, plus y_true/rt_actual if present.
      - Dict mapping target_month -> module_status string.
    """
    csv_map = _discover_monthly_csvs(root)
    if not csv_map:
        logger.warning("No monthly CSVs found for %s in %s", module_name, root)
        return pd.DataFrame(), {}

    all_dfs = []
    month_status: dict[str, str] = {}

    for target_month, csv_path in sorted(csv_map.items()):
        df = _load_module_predictions(csv_path, f"{module_name}/{target_month}")
        df = _normalize_key_columns(df)
        df = _rename_available(df, col_map)

        # Add target_month column.
        df["target_month"] = _extract_target_month_from_predictions(df, target_month)

        # Determine module status for this month.
        verdict = monthly_verdicts.get(target_month, None)
        status = _normalize_verdict_to_status(verdict)
        month_status[target_month] = status

        # If NO-GO for this month, NaN out the risk columns.
        if status == "NO-GO":
            for col in risk_cols:
                if col in df.columns:
                    df[col] = np.nan
            logger.info(
                "%s month %s is NO-GO: risk columns set to NaN", module_name, target_month
            )

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(), month_status

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined, month_status


def build_risk_feature_pack_multimonth(
    delta_supply_root: Path,
    spike_root: Path,
    negative_root: Path,
    mode: str,
    metric_alignment_status: str,
    metric_alignment_warning_reason: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Merge multi-month predictions from all three modules into a unified pack.

    Parameters
    ----------
    delta_supply_root : Path to DeltaSupply backtest root directory
    spike_root : Path to SpikeRisk backtest root directory
    negative_root : Path to NegativeRisk backtest root directory
    mode : "online" or "eval"
    metric_alignment_status : "PASS", "WARN", or "FAIL"
    metric_alignment_warning_reason : reason string for WARN status

    Returns
    -------
    (pack_df, monthly_manifest_df, status_sources) tuple.
    status_sources maps module_name -> "monthly_verdicts" | "overall_verdict".
    """
    # Load verdict summaries (champion_summary.json -> verdict.json fallback).
    delta_summary = _load_verdict_summary(delta_supply_root)
    spike_summary = _load_verdict_summary(spike_root)
    negative_summary = _load_verdict_summary(negative_root)

    delta_verdicts = _get_monthly_verdicts(delta_summary)
    spike_verdicts = _get_monthly_verdicts(spike_summary)
    negative_verdicts = _get_monthly_verdicts(negative_summary)

    # Track status sources: "monthly_verdicts" if monthly verdicts exist,
    # otherwise "overall_verdict" if we used overall verdict as fallback.
    status_sources: dict[str, str] = {}
    for name, summary, verdicts in [
        ("delta_supply", delta_summary, delta_verdicts),
        ("spike", spike_summary, spike_verdicts),
        ("negative", negative_summary, negative_verdicts),
    ]:
        if verdicts:
            status_sources[name] = "monthly_verdicts"
        elif summary is not None and "overall_verdict" in summary:
            status_sources[name] = "overall_verdict"
        else:
            status_sources[name] = "none"

    # Load and prepare each module.
    delta_df, delta_status = _load_and_prepare_module(
        delta_supply_root, "DeltaSupply", DELTA_SUPPLY_COL_MAP,
        DELTA_SUPPLY_RISK_COLS, delta_verdicts,
    )
    spike_df, spike_status = _load_and_prepare_module(
        spike_root, "SpikeRisk", SPIKE_RISK_COL_MAP,
        SPIKE_RISK_COLS, spike_verdicts,
    )
    negative_df, negative_status = _load_and_prepare_module(
        negative_root, "NegativeRisk", NEGATIVE_RISK_COL_MAP,
        NEGATIVE_RISK_COLS, negative_verdicts,
    )

    # Determine the union of all target months.
    all_months = sorted(set(delta_status.keys()) | set(spike_status.keys()) | set(negative_status.keys()))

    # Extract relevant columns from each module.
    delta_target = ["business_day", "hour_business", "ds", "target_month"] + DELTA_SUPPLY_RISK_COLS
    spike_target = ["business_day", "hour_business", "ds", "target_month"] + SPIKE_RISK_COLS
    negative_target = ["business_day", "hour_business", "ds", "target_month"] + NEGATIVE_RISK_COLS

    delta_cols = _select_target_columns(delta_df, delta_target) if not delta_df.empty else pd.DataFrame()
    spike_cols = _select_target_columns(spike_df, spike_target) if not spike_df.empty else pd.DataFrame()
    negative_cols = _select_target_columns(negative_df, negative_target) if not negative_df.empty else pd.DataFrame()

    # Merge on key columns + target_month.
    merge_keys = KEY_COLUMNS + ["target_month"]

    if not delta_cols.empty:
        merged = delta_cols
    else:
        # If delta is empty, start from whichever module has data.
        merged = spike_cols if not spike_cols.empty else negative_cols

    if not merged.empty:
        if not spike_cols.empty:
            merged = merged.merge(
                spike_cols, on=merge_keys, how="outer", suffixes=("", "_spike"),
            )
        if not negative_cols.empty:
            merged = merged.merge(
                negative_cols, on=merge_keys, how="outer", suffixes=("", "_neg"),
            )

        # Reconcile ds: prefer non-null.
        if "ds" in merged.columns and "ds_spike" in merged.columns:
            merged["ds"] = merged["ds"].combine_first(merged["ds_spike"])
        if "ds" in merged.columns and "ds_neg" in merged.columns:
            merged["ds"] = merged["ds"].combine_first(merged["ds_neg"])

        # Drop duplicate ds columns from merge.
        drop_cols = [c for c in merged.columns if c.startswith("ds_")]
        merged = merged.drop(columns=drop_cols, errors="ignore")

        # If ds is still missing, construct from business_day + hour_business.
        if "ds" not in merged.columns or merged["ds"].isna().all():
            merged["ds"] = merged["business_day"] + pd.to_timedelta(
                merged["hour_business"].astype(int), unit="h"
            )
    else:
        logger.error("No data from any module. Cannot produce pack.")
        sys.exit(1)

    # Add module status columns.
    def _map_status(row: pd.Series, status_dict: dict[str, str]) -> str:
        month = row.get("target_month", "")
        return status_dict.get(str(month), "UNKNOWN")

    merged["module_status_delta_supply"] = merged.apply(
        lambda r: _map_status(r, delta_status), axis=1
    )
    merged["module_status_spike"] = merged.apply(
        lambda r: _map_status(r, spike_status), axis=1
    )
    merged["module_status_negative"] = merged.apply(
        lambda r: _map_status(r, negative_status), axis=1
    )

    # Add metadata columns.
    merged["threshold_version"] = THRESHOLD_VERSION
    merged["risk_feature_version"] = RISK_FEATURE_VERSION
    merged["metric_alignment_status"] = metric_alignment_status
    merged["metric_alignment_warning_reason"] = (
        metric_alignment_warning_reason if metric_alignment_status == "WARN" else ""
    )

    # Eval mode: include y_true if available from any source.
    if mode == "eval":
        y_true_candidates = ["y_true", "rt_actual"]
        for src_df, name in [(delta_df, "delta"), (spike_df, "spike"), (negative_df, "negative")]:
            if src_df.empty:
                continue
            for col in y_true_candidates:
                if col in src_df.columns:
                    y_series = src_df[merge_keys + [col]].copy()
                    y_series = y_series.rename(columns={col: "y_true"})
                    merged = merged.merge(y_series, on=merge_keys, how="left")
                    break
            if "y_true" in merged.columns:
                break
        if "y_true" not in merged.columns:
            logger.warning("Eval mode requested but no y_true/rt_actual found in sources")
            merged["y_true"] = np.nan

    # Select final columns in canonical order.
    if mode == "eval":
        final_cols = ONLINE_COLUMNS + EVAL_EXTRA_COLUMNS
    else:
        final_cols = ONLINE_COLUMNS

    available = [c for c in final_cols if c in merged.columns]
    pack = merged[available].copy()

    # Deduplicate by key -- keep first occurrence.
    dedup_keys = KEY_COLUMNS + ["target_month"]
    pack = pack.drop_duplicates(subset=dedup_keys, keep="first")

    # Sort by time.
    pack = pack.sort_values(["target_month"] + KEY_COLUMNS).reset_index(drop=True)

    # Build monthly manifest.
    monthly_rows = []
    for month in all_months:
        month_pack = pack[pack["target_month"] == month]
        monthly_rows.append({
            "target_month": month,
            "n_rows": len(month_pack),
            "module_status_delta_supply": delta_status.get(month, "MISSING"),
            "module_status_spike": spike_status.get(month, "MISSING"),
            "module_status_negative": negative_status.get(month, "MISSING"),
        })
    monthly_manifest = pd.DataFrame(monthly_rows)

    return pack, monthly_manifest, status_sources


def write_manifest(
    out_dir: Path,
    pack_df: pd.DataFrame,
    mode: str,
    metric_alignment_status: str,
    monthly_manifest_df: pd.DataFrame,
    status_sources: dict[str, str] | None = None,
    metric_alignment_warning_reason: str = "",
) -> None:
    """Write manifest.json and monthly_manifest.csv."""
    if status_sources is None:
        status_sources = {}

    # Compute per-module NO-GO month lists.
    nogo_delta = []
    nogo_spike = []
    nogo_negative = []
    if not monthly_manifest_df.empty:
        for _, row in monthly_manifest_df.iterrows():
            if row.get("module_status_delta_supply") == "NO-GO":
                nogo_delta.append(row["target_month"])
            if row.get("module_status_spike") == "NO-GO":
                nogo_spike.append(row["target_month"])
            if row.get("module_status_negative") == "NO-GO":
                nogo_negative.append(row["target_month"])

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "risk_feature_version": RISK_FEATURE_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "mode": mode,
        "metric_alignment_status": metric_alignment_status,
        "metric_alignment_warning_reason": metric_alignment_warning_reason,
        "n_rows": len(pack_df),
        "n_months": int(pack_df["target_month"].nunique()) if "target_month" in pack_df.columns else 0,
        "columns": list(pack_df.columns),
        "column_types": {col: str(pack_df[col].dtype) for col in pack_df.columns},
        "key_columns": KEY_COLUMNS + ["target_month"],
        "unique_keys": int(pack_df.drop_duplicates(subset=KEY_COLUMNS + ["target_month"]).shape[0]),
        "missing_values": {col: int(pack_df[col].isna().sum()) for col in pack_df.columns},
        "date_range": {
            "start": str(pack_df["business_day"].min().date()) if "business_day" in pack_df.columns and len(pack_df) > 0 else None,
            "end": str(pack_df["business_day"].max().date()) if "business_day" in pack_df.columns and len(pack_df) > 0 else None,
        },
        "target_months": sorted(pack_df["target_month"].unique().tolist()) if "target_month" in pack_df.columns else [],
        "module_nogo_months": {
            "delta_supply": nogo_delta,
            "spike": nogo_spike,
            "negative": nogo_negative,
        },
        "status_sources": {
            "delta_supply": status_sources.get("delta_supply", "unknown"),
            "spike": status_sources.get("spike", "unknown"),
            "negative": status_sources.get("negative", "unknown"),
        },
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Manifest -> %s", manifest_path)

    # Write monthly manifest CSV.
    monthly_path = out_dir / "monthly_manifest.csv"
    monthly_manifest_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    logger.info("Monthly manifest -> %s", monthly_path)


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Gate: refuse to produce formal pack if alignment FAIL.
    if args.metric_alignment_status == "FAIL":
        logger.error(
            "Metric alignment status is FAIL. "
            "Refusing to produce formal risk feature pack. "
            "Fix alignment issues first and re-run with --metric-alignment-status PASS or WARN."
        )
        sys.exit(1)

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_root = _resolve_path(args.delta_supply_root)
    spike_root = _resolve_path(args.spike_root)
    negative_root = _resolve_path(args.negative_root)

    # Build unified multi-month pack.
    pack_df, monthly_manifest_df, status_sources = build_risk_feature_pack_multimonth(
        delta_supply_root=delta_root,
        spike_root=spike_root,
        negative_root=negative_root,
        mode=args.mode,
        metric_alignment_status=args.metric_alignment_status,
        metric_alignment_warning_reason=args.metric_alignment_warning_reason,
    )

    # Validate: not all module_status can be UNKNOWN.
    status_cols = ["module_status_delta_supply", "module_status_spike", "module_status_negative"]
    all_unknown = True
    for col in status_cols:
        if col in pack_df.columns:
            non_unknown = pack_df[col][pack_df[col] != "UNKNOWN"]
            if len(non_unknown) > 0:
                all_unknown = False
                break
    if all_unknown:
        logger.error(
            "All module_status columns are UNKNOWN. "
            "Cannot produce a valid risk feature pack without any known module status. "
            "Check that verdict.json or champion_summary.json exist in backtest roots."
        )
        sys.exit(1)

    logger.info(
        "Risk feature pack (multimonth): %d rows, %d columns, %d months (mode=%s)",
        len(pack_df), len(pack_df.columns),
        pack_df["target_month"].nunique() if "target_month" in pack_df.columns else 0,
        args.mode,
    )

    # Validate uniqueness by (business_day, hour_business).
    n_unique = pack_df.drop_duplicates(subset=KEY_COLUMNS).shape[0]
    if n_unique != len(pack_df):
        logger.error(
            "Duplicate keys found: %d rows but only %d unique (business_day, hour_business). "
            "Note: multi-month packs allow the same key across different target_months, "
            "but within the same target_month keys must be unique.",
            len(pack_df), n_unique,
        )
        # Check per-month uniqueness.
        for month, group in pack_df.groupby("target_month"):
            month_unique = group.drop_duplicates(subset=KEY_COLUMNS).shape[0]
            if month_unique != len(group):
                logger.error(
                    "Month %s: %d rows but %d unique keys",
                    month, len(group), month_unique,
                )
        sys.exit(1)
    logger.info("Uniqueness check: PASSED (%d unique keys)", n_unique)

    # Write outputs.
    pack_path = out_dir / "risk_feature_pack.csv"
    pack_df.to_csv(pack_path, index=False, encoding="utf-8-sig")
    logger.info("Risk feature pack -> %s", pack_path)

    write_manifest(
        out_dir, pack_df, args.mode, args.metric_alignment_status,
        monthly_manifest_df, status_sources, args.metric_alignment_warning_reason,
    )

    logger.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
