#!/usr/bin/env python
"""Export a unified risk feature pack from DeltaSupply + Spike + Negative predictions.

Combines the three risk module predictions into a single CSV with one row per
(business_day, hour_business), suitable for downstream fusion and decision modules.

Produces:
  <out-dir>/
    risk_feature_pack.csv       Unified risk features
    manifest.json               Column list, row count, version, alignment status

Usage:
    python scripts/export_risk_feature_pack.py \
        --delta-supply-predictions artifacts/delta_supply/exp_2026_02/predictions.csv \
        --spike-predictions artifacts/spike_risk/exp_2026_02/predictions.csv \
        --negative-predictions artifacts/negative_risk/exp_2026_02/predictions.csv \
        --metric-alignment-status PASS \
        --out-dir reports/local/risk_modules/risk_feature_pack_2026_02 \
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
logger = logging.getLogger("export_risk_feature_pack")

# -- Constants ----------------------------------------------------------------

RISK_FEATURE_VERSION = "v1.0.0"

# Columns that form the join key for all three modules.
KEY_COLUMNS = ["business_day", "hour_business"]

# Online-only output columns (no y_true / rt_actual).
ONLINE_COLUMNS = [
    "business_day",
    "hour_business",
    "ds",
    # DeltaSupply deviation risk
    "deviation_up_prob",
    "deviation_down_prob",
    "deviation_large_abs_prob",
    "deviation_risk_score",
    # Spike risk
    "spike_prob",
    "extreme_spike_prob",
    "spike_risk_score",
    # Negative risk
    "negative_prob",
    "deep_negative_prob",
    "negative_risk_score",
    # Metadata
    "risk_feature_version",
    "metric_alignment_status",
]

# Extra columns added in eval mode.
EVAL_EXTRA_COLUMNS = [
    "y_true",
]

# Expected source columns from each module's predictions CSV.
# The script maps from source column names to the unified pack names.

DELTA_SUPPLY_COL_MAP = {
    # source -> target
    "upward_deviation_prob": "deviation_up_prob",
    "downward_deviation_prob": "deviation_down_prob",
    "large_abs_deviation_prob": "deviation_large_abs_prob",
    "deviation_risk_score": "deviation_risk_score",
}

SPIKE_RISK_COL_MAP = {
    "spike_prob": "spike_prob",
    "extreme_spike_prob": "extreme_spike_prob",
    "spike_risk_score": "spike_risk_score",
}

NEGATIVE_RISK_COL_MAP = {
    "negative_prob": "negative_prob",
    "deep_negative_prob": "deep_negative_prob",
    "negative_risk_score": "negative_risk_score",
}


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unified risk feature pack from DeltaSupply + Spike + Negative",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  risk_feature_pack.csv   Unified risk features (one row per business_hour)
  manifest.json            Column list, row count, version, alignment status

Modes:
  online  -- No rt_actual / y_true columns (production-safe)
  eval    -- Includes y_true columns for backtesting
""",
    )
    parser.add_argument(
        "--delta-supply-predictions", type=str, required=True,
        help="Path to DeltaSupply predictions.csv",
    )
    parser.add_argument(
        "--spike-predictions", type=str, required=True,
        help="Path to SpikeRisk predictions.csv",
    )
    parser.add_argument(
        "--negative-predictions", type=str, required=True,
        help="Path to NegativeRisk predictions.csv",
    )
    parser.add_argument(
        "--metric-alignment-status", type=str, required=True,
        choices=["PASS", "FAIL"],
        help="Metric alignment audit status. FAIL refuses to produce a formal pack.",
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


# -- Core logic ---------------------------------------------------------------

def build_risk_feature_pack(
    delta_supply_df: pd.DataFrame,
    spike_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    mode: str,
    metric_alignment_status: str,
) -> pd.DataFrame:
    """Merge the three module predictions into a unified risk feature pack.

    Parameters
    ----------
    delta_supply_df : DataFrame from DeltaSupply predictions.csv
    spike_df : DataFrame from SpikeRisk predictions.csv
    negative_df : DataFrame from NegativeRisk predictions.csv
    mode : "online" or "eval"
    metric_alignment_status : "PASS" or "FAIL"

    Returns
    -------
    DataFrame with one row per (business_day, hour_business).
    """
    # Normalize keys
    delta_supply_df = _normalize_key_columns(delta_supply_df)
    spike_df = _normalize_key_columns(spike_df)
    negative_df = _normalize_key_columns(negative_df)

    # Rename source columns to unified names
    delta_supply_df = _rename_available(delta_supply_df, DELTA_SUPPLY_COL_MAP)
    spike_df = _rename_available(spike_df, SPIKE_RISK_COL_MAP)
    negative_df = _rename_available(negative_df, NEGATIVE_RISK_COL_MAP)

    # Determine which columns to extract from each module
    delta_target = ["business_day", "hour_business", "ds",
                    "deviation_up_prob", "deviation_down_prob",
                    "deviation_large_abs_prob", "deviation_risk_score"]
    spike_target = ["business_day", "hour_business", "ds",
                    "spike_prob", "extreme_spike_prob", "spike_risk_score"]
    negative_target = ["business_day", "hour_business", "ds",
                       "negative_prob", "deep_negative_prob", "negative_risk_score"]

    # Extract relevant columns
    delta_cols = _select_target_columns(delta_supply_df, delta_target)
    spike_cols = _select_target_columns(spike_df, spike_target)
    negative_cols = _select_target_columns(negative_df, negative_target)

    # Merge on key columns
    merged = delta_cols
    merged = merged.merge(
        spike_cols, on=KEY_COLUMNS, how="outer", suffixes=("", "_spike"),
    )
    merged = merged.merge(
        negative_cols, on=KEY_COLUMNS, how="outer", suffixes=("", "_neg"),
    )

    # Reconcile ds: prefer non-null
    if "ds" in merged.columns and "ds_spike" in merged.columns:
        merged["ds"] = merged["ds"].combine_first(merged["ds_spike"])
    if "ds" in merged.columns and "ds_neg" in merged.columns:
        merged["ds"] = merged["ds"].combine_first(merged["ds_neg"])

    # Drop duplicate ds columns from merge
    drop_cols = [c for c in merged.columns if c.startswith("ds_")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    # If ds is still missing, construct from business_day + hour_business
    if "ds" not in merged.columns or merged["ds"].isna().all():
        merged["ds"] = merged["business_day"] + pd.to_timedelta(
            merged["hour_business"].astype(int), unit="h"
        )

    # Add metadata columns
    merged["risk_feature_version"] = RISK_FEATURE_VERSION
    merged["metric_alignment_status"] = metric_alignment_status

    # Eval mode: include y_true if available from any source
    if mode == "eval":
        y_true_candidates = ["y_true", "rt_actual"]
        for src_df, name in [(delta_supply_df, "delta"), (spike_df, "spike"),
                             (negative_df, "negative")]:
            for col in y_true_candidates:
                if col in src_df.columns:
                    y_series = src_df[KEY_COLUMNS + [col]].copy()
                    y_series = y_series.rename(columns={col: "y_true"})
                    merged = merged.merge(y_series, on=KEY_COLUMNS, how="left")
                    break
            if "y_true" in merged.columns:
                break
        if "y_true" not in merged.columns:
            logger.warning("Eval mode requested but no y_true/rt_actual found in sources")
            merged["y_true"] = np.nan

    # Select final columns in canonical order
    if mode == "eval":
        final_cols = ONLINE_COLUMNS + EVAL_EXTRA_COLUMNS
    else:
        final_cols = ONLINE_COLUMNS

    # Only keep columns that exist
    available = [c for c in final_cols if c in merged.columns]
    pack = merged[available].copy()

    # Deduplicate by key -- keep first occurrence
    pack = pack.drop_duplicates(subset=KEY_COLUMNS, keep="first")

    # Sort by time
    pack = pack.sort_values(KEY_COLUMNS).reset_index(drop=True)

    return pack


def write_manifest(
    out_dir: Path,
    pack_df: pd.DataFrame,
    mode: str,
    metric_alignment_status: str,
) -> None:
    """Write manifest.json with column list, row count, version, alignment status."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "risk_feature_version": RISK_FEATURE_VERSION,
        "mode": mode,
        "metric_alignment_status": metric_alignment_status,
        "n_rows": len(pack_df),
        "columns": list(pack_df.columns),
        "column_types": {col: str(pack_df[col].dtype) for col in pack_df.columns},
        "key_columns": KEY_COLUMNS,
        "unique_keys": int(pack_df.drop_duplicates(subset=KEY_COLUMNS).shape[0]),
        "missing_values": {col: int(pack_df[col].isna().sum()) for col in pack_df.columns},
        "date_range": {
            "start": str(pack_df["business_day"].min().date()) if "business_day" in pack_df.columns else None,
            "end": str(pack_df["business_day"].max().date()) if "business_day" in pack_df.columns else None,
        },
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Manifest -> %s", manifest_path)


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Gate: refuse to produce formal pack if alignment FAIL
    if args.metric_alignment_status == "FAIL":
        logger.error(
            "Metric alignment status is FAIL. "
            "Refusing to produce formal risk feature pack. "
            "Fix alignment issues first and re-run with --metric-alignment-status PASS."
        )
        sys.exit(1)

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source predictions
    delta_supply_df = _load_module_predictions(
        _resolve_path(args.delta_supply_predictions), "DeltaSupply"
    )
    spike_df = _load_module_predictions(
        _resolve_path(args.spike_predictions), "SpikeRisk"
    )
    negative_df = _load_module_predictions(
        _resolve_path(args.negative_predictions), "NegativeRisk"
    )

    # Build unified pack
    pack_df = build_risk_feature_pack(
        delta_supply_df, spike_df, negative_df,
        mode=args.mode,
        metric_alignment_status=args.metric_alignment_status,
    )
    logger.info(
        "Risk feature pack: %d rows, %d columns (mode=%s)",
        len(pack_df), len(pack_df.columns), args.mode,
    )

    # Validate uniqueness
    n_unique = pack_df.drop_duplicates(subset=KEY_COLUMNS).shape[0]
    if n_unique != len(pack_df):
        logger.error(
            "Duplicate keys found: %d rows but only %d unique (business_day, hour_business)",
            len(pack_df), n_unique,
        )
        sys.exit(1)
    logger.info("Uniqueness check: PASSED (%d unique keys)", n_unique)

    # Write outputs
    pack_path = out_dir / "risk_feature_pack.csv"
    pack_df.to_csv(pack_path, index=False, encoding="utf-8-sig")
    logger.info("Risk feature pack -> %s", pack_path)

    write_manifest(out_dir, pack_df, args.mode, args.metric_alignment_status)

    logger.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
