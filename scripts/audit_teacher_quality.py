#!/usr/bin/env python
"""Audit teacher model quality for TrendKnight-X (Phase 4).

Evaluates SGDFNet, RT916, and TimeMixer teachers by computing coverage,
accuracy (sMAPE_floor50, MAE), period breakdowns, residual correlation,
and high-volatility / negative-bucket behaviour.

Outputs:
  teacher_quality_report.json       -- structured audit results
  docs/TEACHER_QUALITY_AUDIT.md     -- narrative report answering 6 key questions

Usage:
    python scripts/audit_teacher_quality.py \\
        --source-repo-root ../electricity_forecast_model2.0_exp \\
        --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \\
        --data-path data/shandong_pmos_hourly.xlsx \\
        --start-date 2026-03-01 --end-date 2026-05-01

    python scripts/audit_teacher_quality.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SGDFNet sibling path resolution (same pattern as run_trendknight_x_ablation.py)
_SIBLING_SGDFNET_SRC = (
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "SGDFNet" / "src"
)
if _SIBLING_SGDFNET_SRC.exists() and str(_SIBLING_SGDFNET_SRC) not in sys.path:
    sys.path.insert(0, str(_SIBLING_SGDFNET_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_teacher_quality")

# ── Constants ────────────────────────────────────────────────────────
TEACHER_NAMES = ["sgdfnet", "rt916", "timemixer"]
SPIKE_THRESHOLD = 500.0
HOURS_PER_DAY = 24


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit teacher model quality for TrendKnight-X (Phase 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Teachers audited:\n"
            "  1. SGDFNet   -- deep hybrid delta model\n"
            "  2. RT916     -- spike / sudden-change model\n"
            "  3. TimeMixer -- multiscale decomposition model\n"
            "\n"
            "Outputs:\n"
            "  teacher_quality_report.json   -- structured metrics\n"
            "  docs/TEACHER_QUALITY_AUDIT.md -- narrative report\n"
        ),
    )
    parser.add_argument("--source-repo-root", type=str, default=None,
                        help="Path to source repo root (for teacher models)")
    parser.add_argument("--sgdfnet-root", type=str, default=None,
                        help="Path to SGDFNet project root")
    parser.add_argument("--data-path", type=str,
                        default="data/shandong_pmos_hourly.xlsx",
                        help="Path to raw data file")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Audit period start (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="Audit period end (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=str, default="reports/local/phase4",
                        help="Output directory (default: reports/local/phase4)")
    return parser.parse_args()


# ── SGDFNet availability (lightweight, same pattern as ablation) ─────

_SGDFNET_AVAILABLE = False
_SGDFNET_IMPORT_ERROR: str | None = None


def _try_import_sgdfnet(sgdfnet_root: str | None = None) -> bool:
    global _SGDFNET_AVAILABLE, _SGDFNET_IMPORT_ERROR
    if sgdfnet_root:
        src = Path(sgdfnet_root) / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    try:
        import sgdfnet.data_contract  # noqa: F401
        import sgdfnet.protocol_b_cutoff  # noqa: F401
        import sgdfnet.metrics  # noqa: F401
        _SGDFNET_AVAILABLE = True
        return True
    except ImportError as exc:
        _SGDFNET_IMPORT_ERROR = str(exc)
        _SGDFNET_AVAILABLE = False
        return False


# ── Data loading (same pattern as run_trendknight_x_ablation.py) ─────

def load_raw_data(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if not p.is_absolute():
        sibling_project = PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp"
        for base in [PROJECT_ROOT, sibling_project]:
            candidate = base / p
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {data_path} (resolved to {p})")
    logger.info("Loading data from %s", p)
    if _SGDFNET_AVAILABLE:
        try:
            from models.deep_sgdf_delta.sgdfnet_bridge import load_dataset
            return load_dataset(str(p))
        except Exception:
            pass  # fall through to manual loading
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if ext == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig")
    return pd.read_csv(p)


# ── Ground truth (same timestamp normalisation as ablation) ──────────

def build_ground_truth(
    raw_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    df = raw_df.copy()
    ts_col: str | None = None
    for c in ("时刻", "timestamp", "ds", "time"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("Cannot find timestamp column in raw data")
    df[ts_col] = pd.to_datetime(df[ts_col])
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
    ts_col = "timestamp"

    if "business_day" not in df.columns:
        if _SGDFNET_AVAILABLE:
            try:
                from models.deep_sgdf_delta.sgdfnet_bridge import add_business_time_columns
                df = add_business_time_columns(df)
            except Exception:
                pass
        if "business_day" not in df.columns:
            df["business_day"] = df[ts_col].dt.normalize()
            df["target_hour"] = df[ts_col].dt.hour
            mask_h0 = df["target_hour"] == 0
            if mask_h0.any():
                df.loc[mask_h0, "target_hour"] = 24
                df.loc[mask_h0, "business_day"] = (
                    df.loc[mask_h0, "business_day"] - pd.Timedelta(days=1)
                )

    df["business_day"] = pd.to_datetime(df["business_day"]).dt.normalize()

    rt_col = da_col = None
    for c in ("rt_actual", "实时电价", "rt_price"):
        if c in df.columns:
            rt_col = c
            break
    for c in ("da_anchor", "日前电价", "da_price"):
        if c in df.columns:
            da_col = c
            break
    if rt_col is None or da_col is None:
        if _SGDFNET_AVAILABLE:
            try:
                from models.deep_sgdf_delta.sgdfnet_bridge import RT_COL, DA_COL
                rt_col, da_col = RT_COL, DA_COL
            except Exception:
                pass
        if rt_col is None or da_col is None:
            raise ValueError("Cannot find RT / DA columns in raw data")

    df["rt_actual"] = pd.to_numeric(df[rt_col], errors="coerce")
    df["da_anchor"] = pd.to_numeric(df[da_col], errors="coerce")
    df["delta_target"] = df["rt_actual"] - df["da_anchor"]

    if "target_hour" not in df.columns and "hour" in df.columns:
        df["target_hour"] = df["hour"]
    df["hour"] = df["target_hour"].astype(int)

    if "segment_id" not in df.columns:
        df["segment_id"] = pd.cut(
            df["hour"], bins=[0, 8, 16, 24],
            labels=[0, 1, 2], include_lowest=True,
        ).astype(int)

    mask = (df["business_day"] >= start_date) & (df["business_day"] <= end_date)
    gt = df.loc[
        mask,
        ["business_day", "hour", "rt_actual", "delta_target", "da_anchor", "segment_id"],
    ].copy()
    gt = gt.dropna(subset=["rt_actual"])
    gt = gt.sort_values(["business_day", "hour"]).reset_index(drop=True)
    logger.info(
        "Ground truth: %d rows, %d days (%s to %s)",
        len(gt), gt["business_day"].nunique(),
        start_date.date(), end_date.date(),
    )
    return gt


# ── Metric helpers ───────────────────────────────────────────────────

def _import_smape_floor50():
    """Import smape_floor50 from the project metrics module."""
    try:
        from models.deep_sgdf_delta.metrics import smape_floor50
        return smape_floor50
    except ImportError:
        # Inline fallback (identical implementation)
        def smape_floor50(y_true, y_pred, floor=50.0, eps=1e-6):
            yt = np.where(y_true < floor, floor, y_true)
            yp = np.where(y_pred < floor, floor, y_pred)
            denom = np.abs(yt) + np.abs(yp) + eps
            return float(np.mean(200.0 * np.abs(yp - yt) / denom))
        return smape_floor50


def _import_period_mask():
    """Import compute_period_mask from the project metrics module."""
    try:
        from models.deep_sgdf_delta.metrics import compute_period_mask
        return compute_period_mask
    except ImportError:
        def compute_period_mask(hours, period):
            if period == "1_8":
                return (hours >= 1) & (hours <= 8)
            if period == "9_16":
                return (hours >= 9) & (hours <= 16)
            if period == "17_24":
                return (hours >= 17) & (hours <= 24)
            raise ValueError(f"Unknown period: {period}")
        return compute_period_mask


def _import_classify_helpers():
    """Import classify_spike and classify_negative."""
    try:
        from models.deep_sgdf_delta.metrics import classify_spike, classify_negative
        return classify_spike, classify_negative
    except ImportError:
        def classify_spike(y_true, threshold=500.0):
            return np.abs(y_true) > threshold
        def classify_negative(y_true, floor=0.0):
            return y_true < floor
        return classify_spike, classify_negative


# ── Teacher quality evaluation ───────────────────────────────────────

def _expected_rows(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Number of hourly rows expected for the period (24 h/day)."""
    n_days = (end_date - start_date).days + 1
    return n_days * HOURS_PER_DAY


def _align_teacher_to_gt(
    teacher_pred: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge teacher predictions with ground truth on (business_day, hour)."""
    tp = teacher_pred.copy()
    tp["business_day"] = pd.to_datetime(tp["business_day"]).dt.normalize()

    # Resolve hour column
    hour_col = None
    for c in ("hour_business", "hour", "target_hour"):
        if c in tp.columns:
            hour_col = c
            break
    if hour_col is None:
        return pd.DataFrame()
    tp["hour"] = tp[hour_col].astype(int)

    # Resolve prediction column
    pred_col = None
    for c in ("teacher_pred", "rt_pred", "y_pred"):
        if c in tp.columns:
            pred_col = c
            break
    if pred_col is None:
        return pd.DataFrame()

    gt = gt_df.copy()
    gt["business_day"] = pd.to_datetime(gt["business_day"]).dt.normalize()
    gt["hour"] = gt["hour"].astype(int)

    merged = tp.merge(gt, on=["business_day", "hour"], how="inner")
    merged["_pred"] = merged[pred_col].astype(float)
    return merged


def audit_single_teacher(
    name: str,
    teacher_pred: Optional[pd.DataFrame],
    gt_df: pd.DataFrame,
    sgdfnet_pred: Optional[pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """Compute quality metrics for a single teacher."""
    smape_fn = _import_smape_floor50()
    period_mask_fn = _import_period_mask()
    classify_spike_fn, classify_negative_fn = _import_classify_helpers()

    expected = _expected_rows(start_date, end_date)
    result: dict[str, Any] = {
        "teacher": name,
        "availability": "unavailable",
        "coverage_rate": 0.0,
        "rows_total": 0,
        "period_coverage_1_8": 0.0,
        "period_coverage_9_16": 0.0,
        "period_coverage_17_24": 0.0,
        "teacher_sMAPE_floor50": float("nan"),
        "teacher_MAE": float("nan"),
        "teacher_vs_sgdfnet_delta": float("nan"),
        "teacher_residual_correlation": float("nan"),
        "teacher_high_volatility_gain": float("nan"),
        "teacher_negative_bucket_behavior": float("nan"),
    }

    if teacher_pred is None or teacher_pred.empty:
        result["availability"] = "unavailable"
        return result

    # Align with ground truth
    merged = _align_teacher_to_gt(teacher_pred, gt_df)
    if merged.empty:
        result["availability"] = "missing_checkpoint"
        result["error"] = "No overlapping rows with ground truth"
        return result

    result["availability"] = "available"
    result["rows_total"] = len(merged)
    result["coverage_rate"] = round(len(merged) / max(expected, 1), 4)

    yp = merged["_pred"].values.astype(float)
    yt = merged["rt_actual"].values.astype(float)
    hours = merged["hour"].values.astype(int)

    # ── Period coverage ──────────────────────────────────────────────
    for period, expected_frac in [("1_8", 8 / 24), ("9_16", 8 / 24), ("17_24", 8 / 24)]:
        mask = period_mask_fn(hours, period)
        n_period = int(mask.sum())
        n_expected_period = max(int(expected * expected_frac), 1)
        result[f"period_coverage_{period}"] = round(n_period / n_expected_period, 4)

    # ── sMAPE_floor50 ────────────────────────────────────────────────
    result["teacher_sMAPE_floor50"] = round(smape_fn(yt, yp), 4)

    # ── Per-period sMAPE (Phase 5 Task C: RT916 local performance) ───
    for period in ["1_8", "9_16", "17_24"]:
        mask = period_mask_fn(hours, period)
        if mask.sum() > 0:
            result[f"teacher_sMAPE_{period}"] = round(smape_fn(yt[mask], yp[mask]), 4)
        else:
            result[f"teacher_sMAPE_{period}"] = float("nan")

    # ── MAE ──────────────────────────────────────────────────────────
    result["teacher_MAE"] = round(float(np.mean(np.abs(yp - yt))), 4)

    # ── Delta vs SGDFNet baseline ────────────────────────────────────
    if sgdfnet_pred is not None and not sgdfnet_pred.empty:
        sgdf_aligned = _align_teacher_to_gt(sgdfnet_pred, gt_df)
        if not sgdf_aligned.empty:
            # Merge teacher and sgdfnet on same keys
            common = merged.merge(
                sgdf_aligned[["business_day", "hour", "_pred"]],
                on=["business_day", "hour"],
                how="inner",
                suffixes=("", "_sgdf"),
            )
            if not common.empty:
                yp_t = common["_pred"].values.astype(float)
                yp_s = common["_pred_sgdf"].values.astype(float)
                yt_c = common["rt_actual"].values.astype(float)
                smape_t = smape_fn(yt_c, yp_t)
                smape_s = smape_fn(yt_c, yp_s)
                result["teacher_vs_sgdfnet_delta"] = round(smape_t - smape_s, 4)

    # ── Residual correlation ─────────────────────────────────────────
    # Correlation between teacher residual (yp - yt) and |yt| (volatility proxy)
    residuals = yp - yt
    abs_yt = np.abs(yt)
    if len(residuals) > 2 and np.std(residuals) > 1e-9 and np.std(abs_yt) > 1e-9:
        corr = float(np.corrcoef(residuals, abs_yt)[0, 1])
        if not np.isnan(corr):
            result["teacher_residual_correlation"] = round(corr, 4)

    # ── High-volatility gain ─────────────────────────────────────────
    # Compare teacher sMAPE vs SGDFNet sMAPE on spike hours only
    spike_mask = classify_spike_fn(yt, SPIKE_THRESHOLD)
    if spike_mask.sum() > 0 and sgdfnet_pred is not None and not sgdfnet_pred.empty:
        sgdf_aligned = _align_teacher_to_gt(sgdfnet_pred, gt_df)
        if not sgdf_aligned.empty:
            common = merged.merge(
                sgdf_aligned[["business_day", "hour", "_pred"]],
                on=["business_day", "hour"],
                how="inner",
                suffixes=("", "_sgdf"),
            )
            if not common.empty:
                # Recompute spike mask on common's rt_actual (not merged's)
                spike_common_mask = classify_spike_fn(
                    common["rt_actual"].values.astype(float), SPIKE_THRESHOLD
                )
                spike_common = common[spike_common_mask]
                if len(spike_common) > 0:
                    yp_t_sp = spike_common["_pred"].values.astype(float)
                    yp_s_sp = spike_common["_pred_sgdf"].values.astype(float)
                    yt_sp = spike_common["rt_actual"].values.astype(float)
                    smape_t_sp = smape_fn(yt_sp, yp_t_sp)
                    smape_s_sp = smape_fn(yt_sp, yp_s_sp)
                    result["teacher_high_volatility_gain"] = round(
                        smape_s_sp - smape_t_sp, 4  # positive = teacher better
                    )

    # ── Negative bucket behaviour ────────────────────────────────────
    neg_mask = classify_negative_fn(yt)
    if neg_mask.sum() > 0:
        neg_yp = yp[neg_mask]
        neg_yt = yt[neg_mask]
        # sMAPE on negative bucket (lower is better)
        result["teacher_negative_bucket_behavior"] = round(
            smape_fn(neg_yt, neg_yp), 4
        )

    return result


# ── Load teachers via registry ───────────────────────────────────────

def load_teachers(
    source_repo_root: str | None,
    sgdfnet_root: str | None,
    start_date: str,
    end_date: str,
) -> tuple[Any, dict[str, Any]]:
    """Load all teachers via TeacherRegistry, returning (registry, statuses)."""
    from models.deep_sgdf_delta.teacher_registry import TeacherRegistry

    registry = TeacherRegistry()
    statuses = registry.load_all(
        teachers=TEACHER_NAMES,
        source_repo_root=source_repo_root,
        sgdfnet_root=sgdfnet_root,
        start_date=start_date,
        end_date=end_date,
    )
    return registry, statuses


# ── JSON report writer ───────────────────────────────────────────────

def write_json_report(
    out_dir: Path,
    audit_results: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> None:
    report = {
        "audit_timestamp": datetime.now().isoformat(),
        "period": {"start": start_date, "end": end_date},
        "teachers": audit_results,
    }
    path = out_dir / "teacher_quality_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("teacher_quality_report.json -> %s", path)


# ── Markdown narrative report ────────────────────────────────────────

def _fmt(val: Any) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if val != val or val == float("inf") or val == float("-inf"):
            return "N/A"
        return f"{val:.4f}"
    return str(val)


def _verdict(condition: bool, yes_text: str, no_text: str) -> str:
    return yes_text if condition else no_text


def write_audit_markdown(
    out_dir: Path,
    audit_results: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> None:
    """Write docs/TEACHER_QUALITY_AUDIT.md answering the 6 key questions."""

    def _get(name: str, key: str) -> Any:
        for r in audit_results:
            if r["teacher"] == name:
                return r.get(key)
        return None

    # Extract key numbers
    sgdf_avail = _get("sgdfnet", "availability")
    sgdf_coverage = _get("sgdfnet", "coverage_rate")
    sgdf_smape = _get("sgdfnet", "teacher_sMAPE_floor50")
    sgdf_rows = _get("sgdfnet", "rows_total")

    rt_avail = _get("rt916", "availability")
    rt_coverage = _get("rt916", "coverage_rate")
    rt_smape = _get("rt916", "teacher_sMAPE_floor50")
    rt_rows = _get("rt916", "rows_total")
    rt_916_cov = _get("rt916", "period_coverage_9_16")
    rt_hv_gain = _get("rt916", "teacher_high_volatility_gain")

    tm_avail = _get("timemixer", "availability")
    tm_coverage = _get("timemixer", "coverage_rate")
    tm_smape = _get("timemixer", "teacher_sMAPE_floor50")
    tm_rows = _get("timemixer", "rows_total")
    tm_916_cov = _get("timemixer", "period_coverage_9_16")

    # ── Q1: SGDFNet teacher 是否完整可用？ ──────────────────────────
    sgdf_complete = sgdf_avail == "available" and (sgdf_coverage or 0) >= 0.9
    q1_verdict = _verdict(
        sgdf_complete,
        "YES -- SGDFNet is fully available with sufficient coverage.",
        "NO -- SGDFNet is NOT fully available or has insufficient coverage.",
    )

    # ── Q2: RT916 teacher 是否有足够覆盖？ ──────────────────────────
    rt_sufficient = rt_avail == "available" and (rt_coverage or 0) >= 0.5
    q2_verdict = _verdict(
        rt_sufficient,
        "YES -- RT916 has sufficient coverage (>=50%).",
        "NO -- RT916 coverage is insufficient (<50%) or unavailable.",
    )

    # ── Q3: TimeMixer teacher 是否有足够覆盖？ ──────────────────────
    tm_sufficient = tm_avail == "available" and (tm_coverage or 0) >= 0.5
    q3_verdict = _verdict(
        tm_sufficient,
        "YES -- TimeMixer has sufficient coverage (>=50%).",
        "NO -- TimeMixer coverage is insufficient (<50%) or unavailable.",
    )

    # ── Q4: RT916 是否只在 high-volatility / 9_16 有价值？ ──────────
    rt_only_hv = False
    if rt_avail == "available" and rt_hv_gain is not None:
        hv_positive = rt_hv_gain > 0
        # Check if 9_16 coverage is much higher than other periods
        rt_18_cov = _get("rt916", "period_coverage_1_8") or 0
        rt_1724_cov = _get("rt916", "period_coverage_17_24") or 0
        rt_916_val = rt_916_cov or 0
        concentrated = rt_916_val > (rt_18_cov + rt_1724_cov)
        rt_only_hv = hv_positive and concentrated
    q4_verdict = _verdict(
        rt_only_hv,
        "YES -- RT916 shows value primarily in high-volatility / 9_16 hours.",
        "NO / PARTIAL -- RT916 value is not limited to high-volatility / 9_16 hours.",
    )

    # ── Q5: TimeMixer 是否值得进入 teacher_moe？ ────────────────────
    tm_worth_moe = False
    if tm_avail == "available":
        tm_delta = _get("timemixer", "teacher_vs_sgdfnet_delta")
        if tm_delta is not None and tm_delta != tm_delta:  # NaN check
            tm_delta = None
        # Worth entering MoE if coverage is reasonable and it's not dramatically worse
        if tm_sufficient and tm_delta is not None and tm_delta < 5.0:
            tm_worth_moe = True
        elif tm_sufficient and tm_delta is None:
            tm_worth_moe = True  # no comparison data but coverage is good
    q5_verdict = _verdict(
        tm_worth_moe,
        "YES -- TimeMixer is worth including in teacher_moe.",
        "NO -- TimeMixer does not meet the threshold for teacher_moe inclusion.",
    )

    # ── Q6: teacher 不可用时是否必须自动降级？ ──────────────────────
    any_unavailable = any(
        r.get("availability") != "available" for r in audit_results
    )
    q6_verdict = _verdict(
        any_unavailable,
        "YES -- at least one teacher is unavailable; v3_teacher_residual and "
        "v3_teacher_moe MUST auto-degrade (fallback to student-only mode).",
        "NO -- all teachers are available; no auto-degradation needed.",
    )

    # ── Build markdown ───────────────────────────────────────────────
    md: list[str] = [
        "# Teacher Quality Audit Report",
        "",
        f"**Audit Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Period:** {start_date} to {end_date}",
        "",
        "## Summary Table",
        "",
        "| Teacher | Availability | Coverage | Rows | sMAPE_floor50 | MAE | vs SGDFNet delta |",
        "|---------|-------------|----------|------|---------------|-----|------------------|",
    ]

    for r in audit_results:
        md.append(
            f"| {r['teacher']} | {r['availability']} | {_fmt(r.get('coverage_rate'))} | "
            f"{r.get('rows_total', 0)} | {_fmt(r.get('teacher_sMAPE_floor50'))} | "
            f"{_fmt(r.get('teacher_MAE'))} | {_fmt(r.get('teacher_vs_sgdfnet_delta'))} |"
        )

    md += [
        "",
        "## Period Coverage Breakdown",
        "",
        "| Teacher | 1_8 | 9_16 | 17_24 |",
        "|---------|-----|------|-------|",
    ]
    for r in audit_results:
        md.append(
            f"| {r['teacher']} | "
            f"{_fmt(r.get('period_coverage_1_8'))} | "
            f"{_fmt(r.get('period_coverage_9_16'))} | "
            f"{_fmt(r.get('period_coverage_17_24'))} |"
        )

    md += [
        "",
        "## Advanced Metrics",
        "",
        "| Teacher | Residual Corr | High-Vol Gain | Neg Bucket sMAPE |",
        "|---------|--------------|---------------|------------------|",
    ]
    for r in audit_results:
        md.append(
            f"| {r['teacher']} | "
            f"{_fmt(r.get('teacher_residual_correlation'))} | "
            f"{_fmt(r.get('teacher_high_volatility_gain'))} | "
            f"{_fmt(r.get('teacher_negative_bucket_behavior'))} |"
        )

    md += [
        "",
        "---",
        "",
        "## Key Questions",
        "",
        "### Q1: SGDFNet teacher 是否完整可用？",
        "",
        f"- Availability: `{sgdf_avail}`",
        f"- Coverage rate: {_fmt(sgdf_coverage)}",
        f"- Rows: {sgdf_rows}",
        f"- Overall sMAPE_floor50: {_fmt(sgdf_smape)}",
        f"- **Verdict:** {q1_verdict}",
        "",
        "### Q2: RT916 teacher 是否有足够覆盖？",
        "",
        f"- Availability: `{rt_avail}`",
        f"- Coverage rate: {_fmt(rt_coverage)}",
        f"- Rows: {rt_rows}",
        f"- 9_16 period coverage: {_fmt(rt_916_cov)}",
        f"- **Verdict:** {q2_verdict}",
        "",
        "### Q3: TimeMixer teacher 是否有足够覆盖？",
        "",
        f"- Availability: `{tm_avail}`",
        f"- Coverage rate: {_fmt(tm_coverage)}",
        f"- Rows: {tm_rows}",
        f"- 9_16 period coverage: {_fmt(tm_916_cov)}",
        f"- **Verdict:** {q3_verdict}",
        "",
        "### Q4: RT916 是否只在 high-volatility / 9_16 有价值？",
        "",
        f"- High-volatility gain (vs SGDFNet): {_fmt(rt_hv_gain)}",
        f"- 1_8 coverage: {_fmt(_get('rt916', 'period_coverage_1_8'))}",
        f"- 9_16 coverage: {_fmt(rt_916_cov)}",
        f"- 17_24 coverage: {_fmt(_get('rt916', 'period_coverage_17_24'))}",
        f"- **Per-period sMAPE (RT916 local performance):**",
        f"  - 1_8 sMAPE: {_fmt(_get('rt916', 'teacher_sMAPE_1_8'))}",
        f"  - 9_16 sMAPE: {_fmt(_get('rt916', 'teacher_sMAPE_9_16'))}",
        f"  - 17_24 sMAPE: {_fmt(_get('rt916', 'teacher_sMAPE_17_24'))}",
        f"- **Per-period sMAPE (SGDFNet for comparison):**",
        f"  - 1_8 sMAPE: {_fmt(_get('sgdfnet', 'teacher_sMAPE_1_8'))}",
        f"  - 9_16 sMAPE: {_fmt(_get('sgdfnet', 'teacher_sMAPE_9_16'))}",
        f"  - 17_24 sMAPE: {_fmt(_get('sgdfnet', 'teacher_sMAPE_17_24'))}",
        f"- **Verdict:** {q4_verdict}",
        "",
        "### Q5: TimeMixer 是否值得进入 teacher_moe？",
        "",
        f"- Availability: `{tm_avail}`",
        f"- Coverage rate: {_fmt(tm_coverage)}",
        f"- vs SGDFNet delta: {_fmt(_get('timemixer', 'teacher_vs_sgdfnet_delta'))}",
        f"- Residual correlation: {_fmt(_get('timemixer', 'teacher_residual_correlation'))}",
        f"- **Verdict:** {q5_verdict}",
        "",
        "### Q6: 如果 teacher 不可用，v3_teacher_residual / v3_teacher_moe 必须自动降级",
        "",
        f"- Any teacher unavailable: {any_unavailable}",
        f"- **Verdict:** {q6_verdict}",
        "",
        "## Recommendations",
        "",
    ]

    # Build recommendations
    if sgdf_complete:
        md.append("- SGDFNet: **USE** as primary teacher for residual distillation.")
    else:
        md.append("- SGDFNet: **UNAVAILABLE** -- all teacher-dependent variants must degrade.")

    if rt_sufficient:
        if rt_only_hv:
            md.append("- RT916: **USE selectively (LOCAL scope)** -- restricted to high-volatility / 9_16 / 17_24 hours only. Blocked from normal low-volatility 1_8 hours.")
        else:
            md.append("- RT916: **USE with scope restriction** -- Phase 5 local scope limits RT916 to high-volatility hours. See Q4 per-period sMAPE for details.")
    else:
        md.append("- RT916: **SKIP** -- insufficient coverage for reliable teacher distillation.")

    if tm_worth_moe:
        md.append("- TimeMixer: **INCLUDE in teacher_moe** -- coverage and accuracy are acceptable.")
    else:
        md.append("- TimeMixer: **EXCLUDE from teacher_moe** -- does not meet quality threshold.")

    if any_unavailable:
        md.append("")
        md.append("**IMPORTANT:** At least one teacher is unavailable. The training pipeline")
        md.append("MUST auto-degrade v3_teacher_residual and v3_teacher_moe to student-only")
        md.append("mode when teacher predictions are missing. Verify the fallback logic in")
        md.append("`train_v3.py` handles this gracefully.")

    md.append("")

    # Write to docs/ subdirectory under out_dir
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "TEACHER_QUALITY_AUDIT.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("TEACHER_QUALITY_AUDIT.md -> %s", md_path)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Teacher Quality Audit (Phase 4)")
    logger.info("=" * 60)
    logger.info("  Period       : %s to %s", start_date.date(), end_date.date())
    logger.info("  Output       : %s", out_dir)
    logger.info("  Source root  : %s", args.source_repo_root or "(auto-detect)")
    logger.info("  SGDFNet root : %s", args.sgdfnet_root or "(auto-detect)")
    logger.info("  Data path    : %s", args.data_path)

    # ── SGDFNet import (optional) ────────────────────────────────────
    sgdfnet_ok = _try_import_sgdfnet(args.sgdfnet_root)
    if sgdfnet_ok:
        logger.info("SGDFNet: available")
    else:
        logger.warning("SGDFNet: NOT available (%s) -- proceeding without it",
                        _SGDFNET_IMPORT_ERROR)

    # ── Load raw data & ground truth ─────────────────────────────────
    try:
        raw_df = load_raw_data(args.data_path)
        logger.info("Raw data: %d rows", len(raw_df))
    except FileNotFoundError as exc:
        logger.error("Data file not found: %s", exc)
        logger.info("Proceeding with audit based on teacher registry status only.")
        raw_df = None

    gt_df = None
    if raw_df is not None:
        try:
            gt_df = build_ground_truth(raw_df, start_date, end_date)
        except Exception as exc:
            logger.error("Failed to build ground truth: %s", exc)
            logger.debug(traceback.format_exc())

    # ── Load teachers via registry ───────────────────────────────────
    logger.info("-" * 60)
    logger.info("Loading teachers via TeacherRegistry ...")
    logger.info("-" * 60)

    try:
        registry, statuses = load_teachers(
            source_repo_root=args.source_repo_root,
            sgdfnet_root=args.sgdfnet_root,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except Exception as exc:
        logger.error("TeacherRegistry failed: %s", exc)
        logger.debug(traceback.format_exc())
        # Create empty registry so audit can still report unavailability
        from models.deep_sgdf_delta.teacher_registry import TeacherRegistry
        registry = TeacherRegistry()
        statuses = registry.summary()

    # Log statuses
    for name, status_info in (
        statuses.items() if isinstance(statuses, dict) and all(isinstance(v, dict) for v in statuses.values())
        else [(n, {"availability": s.availability, "n_predictions": s.n_predictions, "error": s.error})
              for n, s in (statuses.items() if hasattr(statuses, 'items') else [])]
    ):
        avail = status_info.get("availability", "unknown") if isinstance(status_info, dict) else str(status_info)
        logger.info("  Teacher '%s': %s", name, avail)

    # ── Audit each teacher ───────────────────────────────────────────
    audit_results: list[dict[str, Any]] = []
    sgdfnet_pred = registry.predictions.get("sgdfnet")

    for name in TEACHER_NAMES:
        logger.info("-" * 60)
        logger.info("Auditing teacher: %s", name)
        logger.info("-" * 60)

        teacher_pred = registry.predictions.get(name)
        status = registry.teachers.get(name)

        try:
            result = audit_single_teacher(
                name=name,
                teacher_pred=teacher_pred,
                gt_df=gt_df if gt_df is not None else pd.DataFrame(),
                sgdfnet_pred=sgdfnet_pred if name != "sgdfnet" else None,
                start_date=start_date,
                end_date=end_date,
            )
            # Override availability from registry if we couldn't compute locally
            if status is not None and hasattr(status, "availability"):
                if result["availability"] == "unavailable" and status.availability != "unavailable":
                    result["availability"] = status.availability
                    result["registry_note"] = "Registry reports availability but no predictions could be aligned"
            if status is not None and hasattr(status, "error") and status.error:
                result["registry_error"] = status.error
            if status is not None and hasattr(status, "n_predictions"):
                result["registry_n_predictions"] = status.n_predictions
        except Exception as exc:
            logger.error("Audit failed for %s: %s", name, exc)
            logger.debug(traceback.format_exc())
            result = {
                "teacher": name,
                "availability": "unavailable",
                "coverage_rate": 0.0,
                "rows_total": 0,
                "period_coverage_1_8": 0.0,
                "period_coverage_9_16": 0.0,
                "period_coverage_17_24": 0.0,
                "teacher_sMAPE_floor50": float("nan"),
                "teacher_MAE": float("nan"),
                "teacher_vs_sgdfnet_delta": float("nan"),
                "teacher_residual_correlation": float("nan"),
                "teacher_high_volatility_gain": float("nan"),
                "teacher_negative_bucket_behavior": float("nan"),
                "error": str(exc),
            }

        audit_results.append(result)
        logger.info("  availability=%s  coverage=%.4f  sMAPE=%s",
                     result["availability"], result["coverage_rate"],
                     _fmt(result.get("teacher_sMAPE_floor50")))

    # ── Write outputs ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Writing audit outputs")
    logger.info("=" * 60)

    write_json_report(out_dir, audit_results, args.start_date, args.end_date)
    write_audit_markdown(out_dir, audit_results, args.start_date, args.end_date)

    # ── Summary ──────────────────────────────────────────────────────
    n_available = sum(1 for r in audit_results if r["availability"] == "available")
    logger.info("=" * 60)
    logger.info("Audit complete: %d/%d teachers available", n_available, len(TEACHER_NAMES))
    logger.info("Outputs saved to %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
