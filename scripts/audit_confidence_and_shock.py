#!/usr/bin/env python
"""Audit calibration of confidence and shock_sensitivity from TrendKnight-X v3.

Checks whether:
  1. High confidence predictions actually have lower error (calibration).
  2. High shock_sensitivity predictions align with volatile / spike hours.

Outputs (in --out-dir):
  confidence_calibration.csv    Per-bucket error statistics
  shock_sensitivity_audit.csv   Top-10% shock coverage analysis
  confidence_shock_summary.json Readiness flags and correlation stats

Usage:
    python scripts/audit_confidence_and_shock.py \\
        --predictions reports/local/phase4/v3_predictions.csv \\
        --ground-truth data/raw/combined.xlsx \\
        --start-date 2025-01-01 --end-date 2025-03-31 \\
        --out-dir reports/local/phase4

    python scripts/audit_confidence_and_shock.py --help
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

# Import smape_floor50 with a fallback so --help works without the full
# models package being importable (e.g. missing torch / SGDFNet).
try:
    from models.deep_sgdf_delta.metrics import smape_floor50  # type: ignore[import]
except Exception:  # pragma: no cover – fallback for minimal environments
    def smape_floor50(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        floor: float = 50.0,
        eps: float = 1e-6,
    ) -> float:
        """sMAPE with floor-50 capping (standalone fallback)."""
        yt = np.where(y_true < floor, floor, y_true)
        yp = np.where(y_pred < floor, floor, y_pred)
        denom = np.abs(yt) + np.abs(yp) + eps
        return float(np.mean(200.0 * np.abs(yp - yt) / denom))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_confidence_and_shock")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit confidence and shock_sensitivity calibration for TrendKnight-X v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs:
  confidence_calibration.csv    Per-bucket calibration statistics
  shock_sensitivity_audit.csv   Top-10% shock coverage analysis
  confidence_shock_summary.json Readiness flags and summary statistics

Examples:
  python scripts/audit_confidence_and_shock.py \\
      --predictions reports/local/phase4/v3_predictions.csv \\
      --ground-truth data/raw/combined.xlsx

  python scripts/audit_confidence_and_shock.py \\
      --predictions pred.csv --ground-truth data.xlsx \\
      --start-date 2025-01-01 --end-date 2025-03-31 \\
      --out-dir reports/local/phase4
""",
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to v3 prediction CSV with columns: "
             "business_day, hour, rt_pred, confidence, shock_sensitivity",
    )
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to raw data xlsx containing rt_actual and da_anchor",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Audit start date (YYYY-MM-DD). Default: earliest available",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Audit end date (YYYY-MM-DD). Default: latest available",
    )
    parser.add_argument(
        "--out-dir", type=str, default="reports/local/phase4",
        help="Output directory (default: reports/local/phase4)",
    )
    parser.add_argument(
        "--volatility-threshold", type=float, default=100.0,
        help="Threshold for |rt_actual - da_anchor| to flag high-volatility hours (default: 100)",
    )
    parser.add_argument(
        "--spike-threshold", type=float, default=500.0,
        help="Threshold for |rt_actual| to flag price-spike hours (default: 500)",
    )
    parser.add_argument(
        "--error-top-quantile", type=float, default=0.9,
        help="Quantile for defining high-error hours (default: 0.9 = top 10%%)",
    )
    return parser.parse_args()


# ── Data loading helpers ─────────────────────────────────────────────

def _resolve_timestamp_col(df: pd.DataFrame) -> str:
    """Return the name of the timestamp column, handling Chinese variants."""
    for candidate in ("时刻", "timestamp", "time", "时间", "ds", "datetime"):
        if candidate in df.columns:
            return candidate
    # Try case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in ("timestamp", "时刻", "time", "datetime", "ds"):
        if candidate in lower_map:
            return lower_map[candidate]
    raise KeyError(
        f"Cannot find timestamp column. Available columns: {list(df.columns)}"
    )


def _resolve_price_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from *candidates*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load raw data xlsx and normalise key columns.

    Returns DataFrame with at least: business_day, hour, rt_actual, da_anchor
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {p}")

    logger.info("Loading ground truth from %s", p)
    if p.suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(p)
    else:
        raw = pd.read_csv(p, encoding="utf-8-sig")

    logger.info("  raw shape: %s, columns: %s", raw.shape, list(raw.columns))

    # ── Timestamp → business_day + hour ───────────────────────────
    ts_col = _resolve_timestamp_col(raw)
    logger.info("  timestamp column: %s", ts_col)
    raw[ts_col] = pd.to_datetime(raw[ts_col], errors="coerce")
    raw = raw.dropna(subset=[ts_col])

    # Derive business_day (date part) and hour
    raw["business_day"] = raw[ts_col].dt.normalize()

    # Hour column: prefer existing, else derive from timestamp
    if "hour" in raw.columns:
        raw["hour"] = pd.to_numeric(raw["hour"], errors="coerce").astype("Int64")
    elif "target_hour" in raw.columns:
        raw["hour"] = pd.to_numeric(raw["target_hour"], errors="coerce").astype("Int64")
    else:
        raw["hour"] = (raw[ts_col].dt.hour + 1).astype("Int64")  # 1-24 convention

    # ── Price columns ─────────────────────────────────────────────
    rt_col = _resolve_price_col(raw, ["rt_actual", "rt_price", "实时电价", "实际电价", "y_true"])
    da_col = _resolve_price_col(raw, ["da_anchor", "da_price", "日前电价", "预测电价"])

    if rt_col is not None and rt_col != "rt_actual":
        raw["rt_actual"] = pd.to_numeric(raw[rt_col], errors="coerce")
    elif rt_col is not None:
        raw["rt_actual"] = pd.to_numeric(raw["rt_actual"], errors="coerce")

    if da_col is not None and da_col != "da_anchor":
        raw["da_anchor"] = pd.to_numeric(raw[da_col], errors="coerce")
    elif da_col is not None:
        raw["da_anchor"] = pd.to_numeric(raw["da_anchor"], errors="coerce")

    if "rt_actual" not in raw.columns:
        raise KeyError(
            f"Cannot find rt_actual column. Available: {list(raw.columns)}"
        )

    # Keep only what we need
    keep_cols = ["business_day", "hour", "rt_actual"]
    if "da_anchor" in raw.columns:
        keep_cols.append("da_anchor")
    gt = raw[keep_cols].dropna(subset=["business_day", "hour"]).copy()
    gt["business_day"] = pd.to_datetime(gt["business_day"])
    logger.info("  ground truth rows: %d, day range: %s – %s",
                len(gt), gt["business_day"].min(), gt["business_day"].max())
    return gt


def load_predictions(path: str) -> pd.DataFrame:
    """Load v3 prediction CSV and validate required columns."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found: {p}")

    df = pd.read_csv(p, encoding="utf-8-sig")
    logger.info("Loaded %d predictions from %s", len(df), p)

    required = {"business_day", "hour", "rt_pred", "confidence", "shock_sensitivity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Predictions CSV is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["business_day"] = pd.to_datetime(df["business_day"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    for col in ("rt_pred", "confidence", "shock_sensitivity"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["business_day", "hour"])
    return df


# ── Confidence calibration ───────────────────────────────────────────

CONFIDENCE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONFIDENCE_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


def compute_confidence_calibration(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """Bin predictions by confidence and compute per-bucket error stats.

    Parameters
    ----------
    merged : DataFrame with columns
        rt_actual, rt_pred, confidence (already merged with ground truth).

    Returns
    -------
    DataFrame with one row per confidence bucket.
    """
    valid = merged.dropna(subset=["rt_actual", "rt_pred", "confidence"]).copy()
    if valid.empty:
        logger.warning("No valid rows for confidence calibration")
        return pd.DataFrame()

    valid["conf_bucket"] = pd.cut(
        valid["confidence"],
        bins=CONFIDENCE_BINS,
        labels=CONFIDENCE_LABELS,
        include_lowest=True,
        right=True,
    )

    rows = []
    for label in CONFIDENCE_LABELS:
        bucket = valid[valid["conf_bucket"] == label]
        n = len(bucket)
        if n == 0:
            rows.append({
                "confidence_bucket": label,
                "count": 0,
                "mean_mae": np.nan,
                "mean_smape_floor50": np.nan,
                "mean_abs_error": np.nan,
            })
            continue

        yt = bucket["rt_actual"].to_numpy(dtype=float)
        yp = bucket["rt_pred"].to_numpy(dtype=float)
        abs_err = np.abs(yt - yp)

        rows.append({
            "confidence_bucket": label,
            "count": n,
            "mean_mae": float(np.mean(abs_err)),
            "mean_smape_floor50": smape_floor50(yt, yp),
            "mean_abs_error": float(np.mean(abs_err)),
        })

    return pd.DataFrame(rows)


def _confidence_is_calibrated(cal_df: pd.DataFrame) -> tuple[bool, str]:
    """Check whether higher confidence is associated with lower error.

    Returns (is_calibrated, detail_string).
    """
    non_empty = cal_df[cal_df["count"] > 0].copy()
    if len(non_empty) < 2:
        return False, "fewer than 2 non-empty buckets — cannot assess calibration"

    # Spearman rank correlation between bucket index and mean_mae
    x = np.arange(len(non_empty), dtype=float)
    y = non_empty["mean_mae"].to_numpy(dtype=float)
    if np.any(np.isnan(y)):
        return False, "NaN in bucket MAE — incomplete calibration data"

    corr = np.corrcoef(x, y)[0, 1]

    # Calibrated if correlation is clearly negative (higher bucket → lower error)
    if corr < -0.3:
        return True, f"negative correlation (r={corr:+.3f}): high confidence → low error"
    elif corr > 0.3:
        return False, f"positive correlation (r={corr:+.3f}): high confidence → HIGH error (inverted)"
    else:
        return False, f"weak correlation (r={corr:+.3f}): confidence not meaningfully related to error"


# ── Shock sensitivity audit ──────────────────────────────────────────

def compute_shock_sensitivity_audit(
    merged: pd.DataFrame,
    *,
    volatility_threshold: float = 100.0,
    spike_threshold: float = 500.0,
    error_quantile: float = 0.9,
) -> pd.DataFrame:
    """Analyse whether top shock_sensitivity hours cover volatile / spike / high-error hours.

    Returns a summary DataFrame with one row per audit dimension.
    """
    valid = merged.dropna(subset=["rt_actual", "rt_pred", "shock_sensitivity"]).copy()
    if valid.empty:
        logger.warning("No valid rows for shock sensitivity audit")
        return pd.DataFrame()

    # Derived signals
    valid["abs_error"] = np.abs(valid["rt_actual"] - valid["rt_pred"])

    has_da = "da_anchor" in valid.columns and valid["da_anchor"].notna().any()
    if has_da:
        valid["volatility"] = np.abs(valid["rt_actual"] - valid["da_anchor"])
    else:
        valid["volatility"] = np.nan

    valid["is_spike"] = np.abs(valid["rt_actual"]) > spike_threshold

    # Top 10% by shock_sensitivity
    shock_cutoff = valid["shock_sensitivity"].quantile(error_quantile)
    top_shock = valid[valid["shock_sensitivity"] >= shock_cutoff]
    n_top = len(top_shock)

    if n_top == 0:
        logger.warning("No rows in top shock_sensitivity bucket")
        return pd.DataFrame()

    # --- Coverage metrics ---
    rows = []

    # 1. High-volatility coverage
    if has_da:
        n_high_vol = int((valid["volatility"] > volatility_threshold).sum())
        n_top_high_vol = int((top_shock["volatility"] > volatility_threshold).sum())
        vol_coverage = n_top_high_vol / max(n_high_vol, 1)
        rows.append({
            "dimension": "high_volatility",
            "threshold": f"|rt-da|>{volatility_threshold}",
            "total_qualifying": n_high_vol,
            "top_shock_qualifying": n_top_high_vol,
            "coverage_pct": round(vol_coverage * 100, 2),
            "top_shock_size": n_top,
        })

    # 2. High-error coverage
    err_cutoff = valid["abs_error"].quantile(error_quantile)
    n_high_err = int((valid["abs_error"] > err_cutoff).sum())
    n_top_high_err = int((top_shock["abs_error"] > err_cutoff).sum())
    err_coverage = n_top_high_err / max(n_high_err, 1)
    rows.append({
        "dimension": "high_error",
        "threshold": f"|error|>q{error_quantile}({err_cutoff:.1f})",
        "total_qualifying": n_high_err,
        "top_shock_qualifying": n_top_high_err,
        "coverage_pct": round(err_coverage * 100, 2),
        "top_shock_size": n_top,
    })

    # 3. Price spike coverage
    n_spikes = int(valid["is_spike"].sum())
    n_top_spikes = int(top_shock["is_spike"].sum())
    spike_coverage = n_top_spikes / max(n_spikes, 1)
    rows.append({
        "dimension": "price_spike",
        "threshold": f"|rt_actual|>{spike_threshold}",
        "total_qualifying": n_spikes,
        "top_shock_qualifying": n_top_spikes,
        "coverage_pct": round(spike_coverage * 100, 2),
        "top_shock_size": n_top,
    })

    # 4. Mean shock vs non-shock comparison
    mean_shock_top = float(top_shock["shock_sensitivity"].mean())
    mean_shock_rest = float(valid.loc[valid["shock_sensitivity"] < shock_cutoff, "shock_sensitivity"].mean())
    mean_err_top = float(top_shock["abs_error"].mean())
    mean_err_rest = float(valid.loc[valid["abs_error"] < err_cutoff, "abs_error"].mean()) if n_high_err > 0 else np.nan
    rows.append({
        "dimension": "mean_comparison",
        "threshold": "top10% vs rest",
        "total_qualifying": np.nan,
        "top_shock_qualifying": np.nan,
        "coverage_pct": np.nan,
        "top_shock_size": n_top,
    })

    audit_df = pd.DataFrame(rows)
    # Attach extra comparison columns
    audit_df.loc[audit_df["dimension"] == "mean_comparison", "mean_shock_top10"] = mean_shock_top
    audit_df.loc[audit_df["dimension"] == "mean_comparison", "mean_shock_rest"] = mean_shock_rest
    audit_df.loc[audit_df["dimension"] == "mean_comparison", "mean_error_top10"] = mean_err_top
    audit_df.loc[audit_df["dimension"] == "mean_comparison", "mean_error_rest"] = mean_err_rest

    return audit_df


def _shock_signal_is_ready(audit_df: pd.DataFrame) -> tuple[bool, str]:
    """Decide whether shock_sensitivity is a useful signal.

    Heuristic: at least 2 of 3 coverage dimensions must exceed 30%.
    """
    if audit_df.empty:
        return False, "no audit data"

    coverage_rows = audit_df[audit_df["dimension"].isin(
        ["high_volatility", "high_error", "price_spike"]
    )]
    if coverage_rows.empty:
        return False, "no coverage dimensions computed"

    coverages = coverage_rows["coverage_pct"].dropna()
    n_good = int((coverages > 30).sum())

    detail_parts = [f"{r['dimension']}={r['coverage_pct']:.1f}%"
                    for _, r in coverage_rows.iterrows()]
    detail = ", ".join(detail_parts)

    if n_good >= 2:
        return True, f"{n_good}/3 dimensions >30% coverage ({detail})"
    else:
        return False, f"only {n_good}/3 dimensions >30% coverage ({detail})"


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────
    pred_df = load_predictions(args.predictions)
    gt_df = load_ground_truth(args.ground_truth)

    # ── Date filtering ────────────────────────────────────────────
    if args.start_date:
        start = pd.Timestamp(args.start_date)
        pred_df = pred_df[pred_df["business_day"] >= start]
        gt_df = gt_df[gt_df["business_day"] >= start]
        logger.info("Filtered to >= %s", start.date())
    if args.end_date:
        end = pd.Timestamp(args.end_date)
        pred_df = pred_df[pred_df["business_day"] <= end]
        gt_df = gt_df[gt_df["business_day"] <= end]
        logger.info("Filtered to <= %s", end.date())

    # ── Merge predictions with ground truth ───────────────────────
    merge_keys = ["business_day", "hour"]
    merged = pred_df.merge(gt_df, on=merge_keys, how="inner")
    logger.info("Merged rows: %d (pred=%d, gt=%d)", len(merged), len(pred_df), len(gt_df))

    if merged.empty:
        logger.error("No overlapping rows between predictions and ground truth. "
                      "Check business_day / hour alignment.")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Confidence calibration ─────────────────────────────────
    logger.info("Computing confidence calibration ...")
    cal_df = compute_confidence_calibration(merged)
    cal_path = out_dir / "confidence_calibration.csv"
    cal_df.to_csv(cal_path, index=False, encoding="utf-8")
    logger.info("  → %s", cal_path)

    conf_ready, conf_detail = _confidence_is_calibrated(cal_df)

    # ── 2. Shock sensitivity audit ────────────────────────────────
    logger.info("Computing shock sensitivity audit ...")
    audit_df = compute_shock_sensitivity_audit(
        merged,
        volatility_threshold=args.volatility_threshold,
        spike_threshold=args.spike_threshold,
        error_quantile=args.error_top_quantile,
    )
    audit_path = out_dir / "shock_sensitivity_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8")
    logger.info("  → %s", audit_path)

    shock_ready, shock_detail = _shock_signal_is_ready(audit_df)

    # ── 3. Summary JSON ───────────────────────────────────────────
    summary: dict = {
        "audit_timestamp": datetime.now().isoformat(timespec="seconds"),
        "predictions_file": str(Path(args.predictions).resolve()),
        "ground_truth_file": str(Path(args.ground_truth).resolve()),
        "date_range": {
            "start": str(merged["business_day"].min()),
            "end": str(merged["business_day"].max()),
        },
        "merged_rows": len(merged),
        "confidence_calibration": {
            "ready": conf_ready,
            "detail": conf_detail,
            "flag": "ok" if conf_ready else "confidence_not_ready",
            "buckets": cal_df.to_dict(orient="records") if not cal_df.empty else [],
        },
        "shock_sensitivity": {
            "ready": shock_ready,
            "detail": shock_detail,
            "flag": "ok" if shock_ready else "shock_signal_not_ready",
            "audit": audit_df.to_dict(orient="records") if not audit_df.empty else [],
        },
        "flags": [],
    }
    if not conf_ready:
        summary["flags"].append("confidence_not_ready")
    if not shock_ready:
        summary["flags"].append("shock_signal_not_ready")

    summary_path = out_dir / "confidence_shock_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("  → %s", summary_path)

    # ── Console summary ───────────────────────────────────────────
    print()
    print("=" * 64)
    print("  Confidence & Shock Sensitivity Audit")
    print("=" * 64)
    print(f"  Merged rows:  {len(merged)}")
    print(f"  Date range:   {merged['business_day'].min()} – {merged['business_day'].max()}")
    print()
    print(f"  Confidence calibration:  {'OK' if conf_ready else 'NOT READY'}")
    print(f"    {conf_detail}")
    print()
    print(f"  Shock sensitivity:       {'OK' if shock_ready else 'NOT READY'}")
    print(f"    {shock_detail}")
    print()
    if summary["flags"]:
        print(f"  FLAGS: {', '.join(summary['flags'])}")
    else:
        print("  FLAGS: (none — both signals look calibrated)")
    print()
    print("  Artifacts:")
    for fpath in [cal_path, audit_path, summary_path]:
        print(f"    {fpath}")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
