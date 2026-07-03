#!/usr/bin/env python
"""P0: Reproduce SGDFNet baseline metrics for DeepSGDFDelta comparison.

Full-featured baseline reproduction script that:
  1. Locates SGDFNet via the sgdfnet_bridge module
  2. Runs SGDFNet's Protocol B cutoff walk-forward experiment
  3. Processes predictions into the standard output format
  4. Computes comprehensive metrics (sMAPE_floor50, MAE, bucket breakdowns)
  5. Generates all report artefacts for go/no-go comparison

Output files (written to ``--out-dir``):
  predictions.csv       Standardised prediction rows
  metrics_summary.json  Aggregate scalar metrics
  monthly_metrics.csv   Per-month sMAPE_floor50
  segment_metrics.csv   Per-period-segment sMAPE_floor50
  bucket_metrics.csv    Per-bucket (normal / high_price / negative) metrics
  go_nogo.md            Human-readable go/no-go summary

Usage examples::

    # Auto-discover SGDFNet sibling directory, default config
    python scripts/p0_reproduce_sgdfnet_baseline.py

    # Explicit SGDFNet location and custom config
    python scripts/p0_reproduce_sgdfnet_baseline.py \\
        --sgdfnet-root /path/to/SGDFNet \\
        --config /path/to/config.yaml

    # Restrict evaluation window
    python scripts/p0_reproduce_sgdfnet_baseline.py \\
        --start-date 2026-01-01 --end-date 2026-05-11

    python scripts/p0_reproduce_sgdfnet_baseline.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────
# Make the project root importable so ``models.deep_sgdf_delta`` resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT_STR = str(PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

# ── Bridge import ──────────────────────────────────────────────────────
# sgdfnet_bridge resolves the SGDFNet source tree (via --sgdfnet-root,
# $SGDFNET_ROOT, or sibling-directory heuristic) and re-exports all
# the symbols we need.
from models.deep_sgdf_delta.sgdfnet_bridge import (  # noqa: E402
    find_sgdfnet_root,
    run_protocol_b_cutoff_experiment,
    load_protocol_b_cutoff_config,
    build_metrics_frame,
    build_segment_metrics,
    capped_smape,
)
from models.deep_sgdf_delta.metrics import smape_floor50  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("p0_baseline")

# ── Constants ──────────────────────────────────────────────────────────
DEFAULT_OUT_DIR = "reports/local/phase2/baseline_sgdfnet"
DEFAULT_SGDFNET_CONFIG_REL = (
    "configs/cutoff_recovery_2026_diag_a_prune_actualside.yaml"
)
SPIKE_THRESHOLD_DEFAULT = 500.0
HIGH_PRICE_QUANTILE = 0.90  # top 10 % when no spike label column exists
BASELINE_PASS_THRESHOLD = 15.0
SOFT_PASS_THRESHOLD = 15.8


# ======================================================================
#  CLI
# ======================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce SGDFNet baseline for DeepSGDFDelta comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Resolution order for SGDFNet root:
              1. --sgdfnet-root <path>
              2. $SGDFNET_ROOT environment variable
              3. ../electricity_forecast_model2.0_exp/SGDFNet  (sibling)
        """),
    )
    parser.add_argument(
        "--sgdfnet-root",
        type=str,
        default=None,
        help="Path to SGDFNet project root (directory containing src/).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override raw data path (Excel/CSV).  When omitted the path "
             "from the SGDFNet config YAML is used.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Evaluation window start (inclusive), e.g. 2026-01-01.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Evaluation window end (inclusive), e.g. 2026-05-11.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="SGDFNet config YAML path.  Defaults to the well-known "
             "cutoff_recovery config inside the SGDFNet tree.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for reports (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args(argv)


# ======================================================================
#  Config resolution
# ======================================================================

def resolve_sgdfnet_config(
    sgdfnet_root: Path,
    explicit_config: Optional[str],
) -> Path:
    """Return the absolute path to the SGDFNet config YAML.

    Raises ``FileNotFoundError`` when the resolved path does not exist.
    """
    if explicit_config is not None:
        cfg = Path(explicit_config).resolve()
    else:
        cfg = (sgdfnet_root / DEFAULT_SGDFNET_CONFIG_REL).resolve()

    if not cfg.is_file():
        raise FileNotFoundError(
            f"SGDFNet config not found: {cfg}\n"
            f"Pass --config <path> or place the YAML at the default location."
        )
    return cfg


# ======================================================================
#  Prediction format standardisation
# ======================================================================

_STANDARD_PRED_COLS = [
    "business_day",
    "hour_business",
    "period",
    "ds",
    "y_true",
    "y_pred",
    "da_anchor",
    "delta_true",
    "delta_pred",
    "segment_id",
]

# SGDFNet's raw output uses various column names across versions.
# Map every known variant to our canonical names.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "business_day": ["business_day", "target_day", "biz_day"],
    "hour_business": ["hour_business", "target_hour", "hour", "h"],
    "period": ["period", "segment", "segment_id"],
    "ds": ["ds", "timestamp", "datetime", "time"],
    "y_true": ["y_true", "rt_actual", "rt", "actual"],
    "y_pred": ["y_pred", "rt_pred", "rt_hat", "predicted", "forecast"],
    "da_anchor": ["da_anchor", "da", "da_price", "day_ahead"],
    "delta_true": ["delta_true", "delta_target", "delta_actual"],
    "delta_pred": ["delta_pred", "delta_hat", "delta_forecast"],
    "segment_id": ["segment_id", "segment", "period"],
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical column names."""
    rename_map: dict[str, str] = {}
    canonical_present = set(df.columns)
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in canonical_present:
            continue  # already canonical
        for alias in aliases:
            if alias in canonical_present and alias != canonical:
                rename_map[alias] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _ensure_period_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``period`` from ``hour_business`` when missing."""
    if "period" in df.columns:
        return df
    if "hour_business" in df.columns:
        hours = pd.to_numeric(df["hour_business"], errors="coerce")

        def _h2p(h: float) -> str:
            if pd.isna(h):
                return "unknown"
            h = int(h)
            if 1 <= h <= 8:
                return "1_8"
            if 9 <= h <= 16:
                return "9_16"
            if 17 <= h <= 24:
                return "17_24"
            return "unknown"

        df["period"] = hours.apply(_h2p)
    return df


def _ensure_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    """Populate ``segment_id`` — same as ``period`` when absent."""
    if "segment_id" not in df.columns:
        if "period" in df.columns:
            df["segment_id"] = df["period"]
        else:
            df["segment_id"] = "all"
    return df


def _ensure_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive delta_true / delta_pred from rt and da when missing."""
    if "delta_true" not in df.columns:
        if "y_true" in df.columns and "da_anchor" in df.columns:
            df["delta_true"] = (
                pd.to_numeric(df["y_true"], errors="coerce")
                - pd.to_numeric(df["da_anchor"], errors="coerce")
            )
        else:
            df["delta_true"] = np.nan
    if "delta_pred" not in df.columns:
        if "y_pred" in df.columns and "da_anchor" in df.columns:
            df["delta_pred"] = (
                pd.to_numeric(df["y_pred"], errors="coerce")
                - pd.to_numeric(df["da_anchor"], errors="coerce")
            )
        else:
            df["delta_pred"] = np.nan
    return df


def standardise_predictions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform SGDFNet raw output into our standard prediction schema.

    Returns a DataFrame with at least the columns in ``_STANDARD_PRED_COLS``.
    Extra columns from the source are preserved (not dropped).
    """
    df = raw_df.copy()
    df = _rename_columns(df)
    df = _ensure_period_column(df)
    df = _ensure_segment_id(df)
    df = _ensure_delta_columns(df)

    # Coerce types
    for col in ("y_true", "y_pred", "da_anchor", "delta_true", "delta_pred"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hour_business" in df.columns:
        df["hour_business"] = pd.to_numeric(
            df["hour_business"], errors="coerce"
        ).astype("Int64")
    for col in ("business_day", "ds"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ======================================================================
#  Metrics computation
# ======================================================================

def _bucket_labels(
    y_true: np.ndarray,
    df: pd.DataFrame,
    spike_threshold: float = SPIKE_THRESHOLD_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (normal_mask, high_price_mask, negative_mask).

    Uses explicit spike / negative label columns when available,
    otherwise falls back to heuristic rules.
    """
    n = len(y_true)

    # ── Spike / high-price ────────────────────────────────────────
    spike_col = None
    for cand in ("is_spike", "spike_label", "spike_flag", "high_price_bucket_flag"):
        if cand in df.columns:
            spike_col = cand
            break

    if spike_col is not None:
        high_price_mask = df[spike_col].fillna(0).astype(bool).to_numpy()
    else:
        # Fallback: check if absolute threshold captures enough rows
        abs_mask = np.abs(y_true) > spike_threshold
        if abs_mask.sum() >= max(5, int(0.005 * n)):
            high_price_mask = abs_mask
        else:
            # Use top-10 % quantile
            threshold_q = np.nanquantile(np.abs(y_true), HIGH_PRICE_QUANTILE)
            high_price_mask = np.abs(y_true) >= threshold_q
            logger.info(
                f"No spike label column found; using price quantile "
                f"(top {1 - HIGH_PRICE_QUANTILE:.0%}), threshold={threshold_q:.2f}"
            )

    # ── Negative ──────────────────────────────────────────────────
    neg_col = None
    for cand in ("is_negative", "negative_label", "negative_flag", "negative_bucket_flag"):
        if cand in df.columns:
            neg_col = cand
            break

    if neg_col is not None:
        negative_mask = df[neg_col].fillna(0).astype(bool).to_numpy()
    else:
        negative_mask = y_true < 0
        if negative_mask.sum() == 0:
            logger.info("No negative label column and no y_true < 0 rows found.")

    # ── Normal = neither high-price nor negative ──────────────────
    normal_mask = ~high_price_mask & ~negative_mask

    return normal_mask, high_price_mask, negative_mask


def compute_baseline_metrics(df: pd.DataFrame) -> dict:
    """Compute the full set of baseline metrics from a standardised DataFrame.

    Returns a dict suitable for JSON serialisation.
    """
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    n_total = len(df)
    n_valid = len(valid)
    n_missing = n_total - n_valid

    if n_valid == 0:
        logger.warning("No valid prediction rows — returning NaN metrics.")
        return {
            "overall_sMAPE_floor50": float("nan"),
            "MAE": float("nan"),
            "rows_total": n_total,
            "rows_missing": n_missing,
            "coverage_rate": 0.0,
        }

    yt = valid["y_true"].to_numpy(dtype=float)
    yp = valid["y_pred"].to_numpy(dtype=float)
    hours = (
        valid["hour_business"].to_numpy(dtype=float)
        if "hour_business" in valid.columns
        else np.full(n_valid, np.nan)
    )

    metrics: dict = {}

    # ── Overall ───────────────────────────────────────────────────
    metrics["overall_sMAPE_floor50"] = smape_floor50(yt, yp)
    metrics["MAE"] = float(np.mean(np.abs(yp - yt)))
    metrics["rows_total"] = n_total
    metrics["rows_missing"] = n_missing
    metrics["coverage_rate"] = n_valid / n_total if n_total > 0 else 0.0

    # ── Period segments ───────────────────────────────────────────
    for label, lo, hi in [("1_8", 1, 8), ("9_16", 9, 16), ("17_24", 17, 24)]:
        mask = (hours >= lo) & (hours <= hi)
        if mask.sum() > 0:
            metrics[f"{label}_sMAPE_floor50"] = smape_floor50(yt[mask], yp[mask])
        else:
            metrics[f"{label}_sMAPE_floor50"] = float("nan")

    # ── Bucket breakdown ──────────────────────────────────────────
    normal_mask, hp_mask, neg_mask = _bucket_labels(yt, valid)

    if normal_mask.sum() > 0:
        metrics["normal_bucket_sMAPE_floor50"] = smape_floor50(
            yt[normal_mask], yp[normal_mask]
        )
    else:
        metrics["normal_bucket_sMAPE_floor50"] = float("nan")

    if hp_mask.sum() > 0:
        metrics["high_price_bucket_sMAPE_floor50"] = smape_floor50(
            yt[hp_mask], yp[hp_mask]
        )
    else:
        metrics["high_price_bucket_sMAPE_floor50"] = float("nan")

    if neg_mask.sum() > 0:
        metrics["negative_bucket_sMAPE_floor50"] = smape_floor50(
            yt[neg_mask], yp[neg_mask]
        )
    else:
        metrics["negative_bucket_sMAPE_floor50"] = float("nan")

    return metrics


def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-month sMAPE_floor50 and MAE."""
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    if valid.empty:
        return pd.DataFrame()

    # Derive month column
    if "business_day" in valid.columns:
        valid["_month"] = pd.to_datetime(valid["business_day"]).dt.to_period("M").astype(str)
    elif "ds" in valid.columns:
        valid["_month"] = pd.to_datetime(valid["ds"]).dt.to_period("M").astype(str)
    else:
        valid["_month"] = "unknown"

    rows = []
    for month, grp in valid.groupby("_month"):
        yt = grp["y_true"].to_numpy(dtype=float)
        yp = grp["y_pred"].to_numpy(dtype=float)
        rows.append({
            "month": month,
            "sMAPE_floor50": smape_floor50(yt, yp),
            "MAE": float(np.mean(np.abs(yp - yt))),
            "count": len(grp),
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def compute_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-period-segment sMAPE_floor50 and MAE."""
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    if valid.empty or "period" not in valid.columns:
        return pd.DataFrame()

    rows = []
    for seg, grp in valid.groupby("period"):
        yt = grp["y_true"].to_numpy(dtype=float)
        yp = grp["y_pred"].to_numpy(dtype=float)
        rows.append({
            "segment": seg,
            "sMAPE_floor50": smape_floor50(yt, yp),
            "MAE": float(np.mean(np.abs(yp - yt))),
            "count": len(grp),
        })
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def compute_bucket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bucket (normal / high_price / negative) metrics as a DataFrame."""
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    if valid.empty:
        return pd.DataFrame()

    yt = valid["y_true"].to_numpy(dtype=float)
    yp = valid["y_pred"].to_numpy(dtype=float)

    normal_mask, hp_mask, neg_mask = _bucket_labels(yt, valid)

    rows = []
    for label, mask in [
        ("normal", normal_mask),
        ("high_price", hp_mask),
        ("negative", neg_mask),
    ]:
        if mask.sum() > 0:
            rows.append({
                "bucket": label,
                "sMAPE_floor50": smape_floor50(yt[mask], yp[mask]),
                "MAE": float(np.mean(np.abs(yp[mask] - yt[mask]))),
                "count": int(mask.sum()),
                "share": float(mask.sum()) / len(yt),
            })
        else:
            rows.append({
                "bucket": label,
                "sMAPE_floor50": float("nan"),
                "MAE": float("nan"),
                "count": 0,
                "share": 0.0,
            })
    return pd.DataFrame(rows)


# ======================================================================
#  Date filtering
# ======================================================================

def apply_date_filter(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """Filter rows to [start_date, end_date] using the best available date column."""
    if start_date is None and end_date is None:
        return df

    # Pick the date column to filter on
    date_col: Optional[str] = None
    for cand in ("business_day", "ds", "timestamp"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        logger.warning(
            "No date column found (business_day / ds / timestamp); "
            "skipping date filter."
        )
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)

    if start_date is not None:
        mask &= dates >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= dates <= pd.Timestamp(end_date)

    filtered = df.loc[mask].copy()
    logger.info(
        f"Date filter [{start_date} .. {end_date}] on '{date_col}': "
        f"{len(df)} -> {len(filtered)} rows"
    )
    return filtered


# ======================================================================
#  Report generation
# ======================================================================

def generate_go_nogo_report(
    metrics: dict,
    monthly_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    out_dir: Path,
    *,
    sgdfnet_config_path: Path,
    data_path: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    """Write ``go_nogo.md`` into *out_dir*."""
    overall = metrics.get("overall_sMAPE_floor50", float("nan"))

    # Determine verdict
    if not np.isnan(overall) and overall < BASELINE_PASS_THRESHOLD:
        verdict = "PASS"
        verdict_detail = (
            f"Overall sMAPE_floor50 = {overall:.4f} < {BASELINE_PASS_THRESHOLD}"
        )
    elif not np.isnan(overall) and overall <= SOFT_PASS_THRESHOLD:
        verdict = "SOFT_PASS"
        verdict_detail = (
            f"Overall sMAPE_floor50 = {overall:.4f} <= {SOFT_PASS_THRESHOLD}; "
            f"awaiting spike / negative module fusion"
        )
    else:
        verdict = "BASELINE"
        verdict_detail = (
            f"Overall sMAPE_floor50 = {overall:.4f}  "
            f"(this IS the SGDFNet baseline for comparison)"
        )

    lines: list[str] = [
        "# SGDFNet Baseline Reproduction — Go/No-Go Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Verdict:** {verdict}",
        "",
        f"{verdict_detail}",
        "",
        "## Configuration",
        "",
        f"- SGDFNet config: `{sgdfnet_config_path}`",
        f"- Data path: `{data_path or '(from config)'}`",
        f"- Evaluation window: {start_date or '(all)'} .. {end_date or '(all)'}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall sMAPE_floor50 | {_fmt(overall)} |",
        f"| MAE | {_fmt(metrics.get('MAE'))} |",
        f"| 1_8 sMAPE_floor50 | {_fmt(metrics.get('1_8_sMAPE_floor50'))} |",
        f"| 9_16 sMAPE_floor50 | {_fmt(metrics.get('9_16_sMAPE_floor50'))} |",
        f"| 17_24 sMAPE_floor50 | {_fmt(metrics.get('17_24_sMAPE_floor50'))} |",
        f"| Normal bucket sMAPE_floor50 | {_fmt(metrics.get('normal_bucket_sMAPE_floor50'))} |",
        f"| High-price bucket sMAPE_floor50 | {_fmt(metrics.get('high_price_bucket_sMAPE_floor50'))} |",
        f"| Negative bucket sMAPE_floor50 | {_fmt(metrics.get('negative_bucket_sMAPE_floor50'))} |",
        f"| Rows total | {metrics.get('rows_total', 0)} |",
        f"| Rows missing | {metrics.get('rows_missing', 0)} |",
        f"| Coverage rate | {_fmt(metrics.get('coverage_rate'))} |",
        "",
    ]

    # Monthly breakdown
    if not monthly_df.empty:
        lines += [
            "## Monthly Breakdown",
            "",
            "| Month | sMAPE_floor50 | MAE | Count |",
            "|-------|---------------|-----|-------|",
        ]
        for _, row in monthly_df.iterrows():
            lines.append(
                f"| {row['month']} | {_fmt(row['sMAPE_floor50'])} "
                f"| {_fmt(row['MAE'])} | {row['count']} |"
            )
        lines.append("")

    # Segment breakdown
    if not segment_df.empty:
        lines += [
            "## Segment Breakdown",
            "",
            "| Segment | sMAPE_floor50 | MAE | Count |",
            "|---------|---------------|-----|-------|",
        ]
        for _, row in segment_df.iterrows():
            lines.append(
                f"| {row['segment']} | {_fmt(row['sMAPE_floor50'])} "
                f"| {_fmt(row['MAE'])} | {row['count']} |"
            )
        lines.append("")

    # Bucket breakdown
    if not bucket_df.empty:
        lines += [
            "## Bucket Breakdown",
            "",
            "| Bucket | sMAPE_floor50 | MAE | Count | Share |",
            "|--------|---------------|-----|-------|-------|",
        ]
        for _, row in bucket_df.iterrows():
            lines.append(
                f"| {row['bucket']} | {_fmt(row['sMAPE_floor50'])} "
                f"| {_fmt(row['MAE'])} | {row['count']} "
                f"| {row['share']:.1%} |"
            )
        lines.append("")

    lines += [
        "## Thresholds",
        "",
        f"- PASS: overall sMAPE_floor50 < {BASELINE_PASS_THRESHOLD}",
        f"- SOFT PASS: overall sMAPE_floor50 <= {SOFT_PASS_THRESHOLD}",
        "- BASELINE: this run establishes the SGDFNet reference number",
        "",
    ]

    (out_dir / "go_nogo.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"go_nogo.md written to {out_dir}")


def _fmt(val: object) -> str:
    """Format a metric value for Markdown tables."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


# ======================================================================
#  Main
# ======================================================================

def main() -> None:
    args = parse_args()

    # ── 1. Resolve SGDFNet location ───────────────────────────────
    logger.info("Resolving SGDFNet root ...")
    sgdfnet_root = find_sgdfnet_root(args.sgdfnet_root)
    logger.info(f"SGDFNet root: {sgdfnet_root}")

    # ── 2. Resolve config ─────────────────────────────────────────
    sgdfnet_config = resolve_sgdfnet_config(sgdfnet_root, args.config)
    logger.info(f"SGDFNet config: {sgdfnet_config}")

    # ── 3. Prepare output directory ───────────────────────────────
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # ── 4. Run SGDFNet Protocol B cutoff experiment ───────────────
    logger.info("Running SGDFNet Protocol B cutoff walk-forward experiment ...")
    try:
        # SGDFNet config uses relative data paths (data/shandong_pmos_hourly.xlsx)
        # relative to the parent project root (electricity_forecast_model2.0_exp/)
        import os
        orig_cwd = os.getcwd()
        project_root = sgdfnet_root.parent  # electricity_forecast_model2.0_exp/
        try:
            os.chdir(str(project_root))
            logger.info(f"Changed CWD to project root: {project_root}")
            run_dir = run_protocol_b_cutoff_experiment(str(sgdfnet_config))
        finally:
            os.chdir(orig_cwd)
        run_dir = Path(run_dir)
        logger.info(f"SGDFNet experiment completed.  Output dir: {run_dir}")
    except Exception:
        logger.error("SGDFNet experiment failed:")
        traceback.print_exc()
        sys.exit(1)

    # ── 5. Load raw predictions ───────────────────────────────────
    pred_path = run_dir / "predictions.csv"
    if not pred_path.is_file():
        # Try alternative names the experiment may produce
        for alt in ("pred.csv", "forecast.csv", "results.csv"):
            alt_path = run_dir / alt
            if alt_path.is_file():
                pred_path = alt_path
                break
        else:
            logger.error(
                f"No predictions CSV found in {run_dir}.  "
                f"Looked for: predictions.csv, pred.csv, forecast.csv, results.csv"
            )
            sys.exit(1)

    raw_df = pd.read_csv(pred_path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(raw_df)} raw prediction rows from {pred_path}")

    # ── 6. Standardise to our prediction format ───────────────────
    std_df = standardise_predictions(raw_df)

    # ── 7. Optional: override data path in config ─────────────────
    # (The experiment already ran with the config's data path; this
    #  parameter is informational and used in the report header.)
    data_path_display = args.data_path

    # ── 8. Apply date filter ──────────────────────────────────────
    std_df = apply_date_filter(std_df, args.start_date, args.end_date)
    if std_df.empty:
        logger.error("No rows remaining after date filtering.  Aborting.")
        sys.exit(1)
    logger.info(f"Working with {len(std_df)} prediction rows after filtering.")

    # ── 9. Compute all metrics ────────────────────────────────────
    logger.info("Computing baseline metrics ...")
    metrics = compute_baseline_metrics(std_df)

    monthly_df = compute_monthly_metrics(std_df)
    if not monthly_df.empty:
        metrics["monthly_sMAPE_floor50"] = float(monthly_df["sMAPE_floor50"].mean())
    else:
        metrics["monthly_sMAPE_floor50"] = float("nan")

    segment_df = compute_segment_metrics(std_df)
    bucket_df = compute_bucket_metrics(std_df)

    # ── 10. Write output files ────────────────────────────────────
    logger.info("Writing output files ...")

    # predictions.csv — standard columns first, extras after
    pred_cols = [c for c in _STANDARD_PRED_COLS if c in std_df.columns]
    extra_cols = [c for c in std_df.columns if c not in _STANDARD_PRED_COLS]
    std_df[pred_cols + extra_cols].to_csv(
        out_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )

    # metrics_summary.json
    # Convert any numpy types to native Python for JSON serialisation
    serialisable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            serialisable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serialisable[k] = float(v)
        elif isinstance(v, float) and np.isnan(v):
            serialisable[k] = None  # JSON has no NaN
        else:
            serialisable[k] = v
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)

    # monthly_metrics.csv
    if not monthly_df.empty:
        monthly_df.to_csv(
            out_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig"
        )

    # segment_metrics.csv
    if not segment_df.empty:
        segment_df.to_csv(
            out_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig"
        )

    # bucket_metrics.csv
    if not bucket_df.empty:
        bucket_df.to_csv(
            out_dir / "bucket_metrics.csv", index=False, encoding="utf-8-sig"
        )

    # go_nogo.md
    generate_go_nogo_report(
        metrics,
        monthly_df,
        segment_df,
        bucket_df,
        out_dir,
        sgdfnet_config_path=sgdfnet_config,
        data_path=data_path_display,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # ── 11. Summary to console ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SGDFNet Baseline Metrics Summary")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            if np.isnan(v):
                logger.info(f"  {k}: N/A")
            else:
                logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    logger.info(f"All reports written to: {out_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
