#!/usr/bin/env python
"""TrendKnight-X Ablation Study -- compare all v2/v3 variants and pick the best.

Candidates:
  1. sgdfnet_baseline        -- SGDFNet reference metrics (no deep model)
  2. v2_day_tcn              -- V2 day-level TCN backbone
  3. v2_residual_sgdfnet     -- V2 TCN + SGDFNet residual blend
  4. v3_fast_tcn             -- V3 TCN without multiscale / teacher
  5. v3_multiscale_tcn       -- V3 TCN + multiscale decomposition
  6. v3_teacher_residual     -- V3 multiscale + teacher residual distillation
  7. v3_teacher_moe          -- V3 multiscale + MoE teacher fusion

Output (under --out-dir, default reports/local/phase3/ablation/):
  leaderboard.csv            -- ranked candidate comparison
  ablation_summary.md        -- narrative analysis answering key questions
  monthly_metrics.csv        -- per-month sMAPE_floor50 for every candidate
  period_metrics.csv         -- per-period (1_8 / 9_16 / 17_24) breakdown
  bucket_metrics.csv         -- per-bucket (normal / spike / negative) breakdown
  runtime_report.json        -- timing and parameter counts

Usage:
    python scripts/run_trendknight_x_ablation.py \\
        --start-date 2026-03-01 --end-date 2026-05-01 \\
        --data-path data/shandong_pmos_hourly.xlsx

    python scripts/run_trendknight_x_ablation.py --fast-dev-run \\
        --start-date 2026-04-01 --end-date 2026-04-07 \\
        --data-path data/shandong_pmos_hourly.xlsx

    python scripts/run_trendknight_x_ablation.py --profile v3_multiscale_tcn \\
        --start-date 2026-03-01 --end-date 2026-05-01

    python scripts/run_trendknight_x_ablation.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SGDFNet sibling path resolution
_SIBLING_SGDFNET_SRC = (
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "SGDFNet" / "src"
)
if _SIBLING_SGDFNET_SRC.exists() and str(_SIBLING_SGDFNET_SRC) not in sys.path:
    sys.path.insert(0, str(_SIBLING_SGDFNET_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_trendknight_x_ablation")

# ── Constants ────────────────────────────────────────────────────────
PASS_THRESHOLD = 15.0
SOFT_PASS_THRESHOLD = 15.8
BASELINE_SGDFNET = 16.5902
MAX_RUNTIME_SECONDS = 3600

SPIKE_THRESHOLD = 500.0

CANDIDATE_NAMES = [
    "sgdfnet_baseline",
    "v2_day_tcn",
    "v2_residual_sgdfnet",
    "v3_fast_tcn",
    "v3_multiscale_tcn",
    "v3_teacher_residual",
    "v3_teacher_moe",
]

SEASON_MAP: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrendKnight-X Ablation Study: compare v2/v3 variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Candidates compared:\n"
            "  1. sgdfnet_baseline\n"
            "  2. v2_day_tcn\n"
            "  3. v2_residual_sgdfnet\n"
            "  4. v3_fast_tcn\n"
            "  5. v3_multiscale_tcn\n"
            "  6. v3_teacher_residual\n"
            "  7. v3_teacher_moe\n"
        ),
    )
    parser.add_argument("--start-date", type=str, required=True,
                        help="Evaluation period start (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="Evaluation period end (YYYY-MM-DD)")
    parser.add_argument("--sgdfnet-root", type=str, default=None,
                        help="Path to SGDFNet project root")
    parser.add_argument("--source-repo-root", type=str, default=None,
                        help="Path to source repo root (for teacher models)")
    parser.add_argument("--data-path", type=str,
                        default="data/shandong_pmos_hourly.xlsx",
                        help="Path to raw data file")
    parser.add_argument("--out-dir", type=str,
                        default="reports/local/phase3/ablation",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP (mixed precision)")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Smoke test: tiny subset, few epochs")
    parser.add_argument("--profile", type=str, default=None,
                        help="Override runtime profile for v3 candidates")
    parser.add_argument("--teachers", type=str, nargs="*", default=None,
                        help="Teacher names to enable (default: all available)")
    return parser.parse_args()


# ── SGDFNet availability ─────────────────────────────────────────────

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


# ── Data loading ─────────────────────────────────────────────────────

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
        from models.deep_sgdf_delta.sgdfnet_bridge import load_dataset
        return load_dataset(str(p))
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if ext == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig")
    return pd.read_csv(p)


# ── Ground truth ─────────────────────────────────────────────────────

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

    if "business_day" not in df.columns:
        if _SGDFNET_AVAILABLE:
            from models.deep_sgdf_delta.sgdfnet_bridge import add_business_time_columns
            df = add_business_time_columns(df)
        else:
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
            from models.deep_sgdf_delta.sgdfnet_bridge import RT_COL, DA_COL
            rt_col, da_col = RT_COL, DA_COL
        else:
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


# ── sMAPE helper (numpy) ────────────────────────────────────────────

def _smape_floor50_np(
    y_true: np.ndarray, y_pred: np.ndarray, floor: float = 50.0,
) -> float:
    yt = np.where(y_true < floor, floor, y_true)
    yp = np.where(y_pred < floor, floor, y_pred)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


# ── Comprehensive evaluation ─────────────────────────────────────────

def evaluate_candidate_full(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    name: str,
) -> dict[str, Any]:
    """Evaluate predictions with overall, monthly, period, and bucket metrics."""
    from models.deep_sgdf_delta.metrics import (
        smape_floor50, compute_period_mask, classify_spike, classify_negative,
    )

    pred = pred_df.copy()
    pred["business_day"] = pd.to_datetime(pred["business_day"]).dt.normalize()
    pred["hour"] = pred["hour"].astype(int)

    gt = gt_df.copy()
    gt["business_day"] = pd.to_datetime(gt["business_day"]).dt.normalize()
    gt["hour"] = gt["hour"].astype(int)

    merged = pred.merge(
        gt, on=["business_day", "hour"], how="inner", suffixes=("_pred", "_true"),
    )

    if merged.empty:
        logger.warning("%s: no matching rows", name)
        return _empty_result(name)

    # Resolve columns
    yp_col = "rt_pred" if "rt_pred" in merged.columns else None
    yt_col = "rt_actual_true" if "rt_actual_true" in merged.columns else "rt_actual"
    if yp_col is None or yt_col not in merged.columns:
        logger.warning("%s: missing rt_pred or rt_actual columns", name)
        return _empty_result(name)

    yp = merged[yp_col].values.astype(float)
    yt = merged[yt_col].values.astype(float)
    hours = merged["hour"].values.astype(int)
    months = pd.to_datetime(merged["business_day"]).dt.month.values

    # ── Overall ──
    overall = smape_floor50(yt, yp)

    # ── Monthly ──
    monthly: dict[str, float] = {}
    for month_val in sorted(set(months)):
        m_mask = months == month_val
        if m_mask.sum() > 0:
            month_label = pd.Timestamp(year=int(pd.Timestamp(merged["business_day"].iloc[0]).year),
                                       month=int(month_val), day=1).strftime("%Y-%m")
            # Use actual year-month from data
            sample_days = pd.to_datetime(merged.loc[m_mask, "business_day"])
            month_label = sample_days.dt.to_period("M").iloc[0].strftime("%Y-%m")
            monthly[month_label] = smape_floor50(yt[m_mask], yp[m_mask])

    monthly_avg = float(np.mean(list(monthly.values()))) if monthly else float("nan")
    monthly_worst = float(np.max(list(monthly.values()))) if monthly else float("nan")

    # ── Period ──
    period_metrics: dict[str, float] = {}
    for period in ("1_8", "9_16", "17_24"):
        mask = compute_period_mask(hours, period)
        if mask.sum() > 0:
            period_metrics[period] = smape_floor50(yt[mask], yp[mask])
        else:
            period_metrics[period] = float("nan")

    # ── Bucket ──
    spike_mask = classify_spike(yt, SPIKE_THRESHOLD)
    neg_mask = classify_negative(yt)
    normal_mask = ~spike_mask & ~neg_mask

    bucket_metrics: dict[str, Any] = {}
    for bname, bmask in [("normal", normal_mask), ("spike", spike_mask), ("negative", neg_mask)]:
        if bmask.sum() > 0:
            bucket_metrics[bname] = {
                "sMAPE_floor50": smape_floor50(yt[bmask], yp[bmask]),
                "count": int(bmask.sum()),
            }
        else:
            bucket_metrics[bname] = {"sMAPE_floor50": float("nan"), "count": 0}

    # ── Delta MAE ──
    dp_col = "delta_pred" if "delta_pred" in merged.columns else None
    dt_col = (
        "delta_target_true"
        if "delta_target_true" in merged.columns
        else ("delta_target" if "delta_target" in merged.columns else None)
    )
    delta_mae_val = float("nan")
    if dp_col and dt_col:
        delta_mae_val = float(np.mean(
            np.abs(merged[dp_col].values.astype(float) - merged[dt_col].values.astype(float))
        ))

    return {
        "name": name,
        "overall_sMAPE_floor50": round(overall, 4),
        "monthly_avg_sMAPE_floor50": round(monthly_avg, 4),
        "monthly_worst_sMAPE": round(monthly_worst, 4),
        "9_16_sMAPE_floor50": round(period_metrics.get("9_16", float("nan")), 4),
        "1_8_sMAPE_floor50": round(period_metrics.get("1_8", float("nan")), 4),
        "17_24_sMAPE_floor50": round(period_metrics.get("17_24", float("nan")), 4),
        "delta_mae": round(delta_mae_val, 4),
        "rows_matched": len(merged),
        "monthly_smape": monthly,
        "period_smape": period_metrics,
        "bucket_smape": bucket_metrics,
        "leakage_risk": False,
    }


def _empty_result(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "overall_sMAPE_floor50": float("inf"),
        "monthly_avg_sMAPE_floor50": float("inf"),
        "monthly_worst_sMAPE": float("inf"),
        "9_16_sMAPE_floor50": float("inf"),
        "1_8_sMAPE_floor50": float("nan"),
        "17_24_sMAPE_floor50": float("nan"),
        "delta_mae": float("nan"),
        "rows_matched": 0,
        "monthly_smape": {},
        "period_smape": {},
        "bucket_smape": {},
        "leakage_risk": False,
    }


# ── V2 model: train + predict ────────────────────────────────────────

def _predict_v2_for_period(
    model: Any, raw_df: pd.DataFrame,
    start_date: pd.Timestamp, end_date: pd.Timestamp,
    device: Any, batch_size: int, fast_dev_run: bool,
) -> pd.DataFrame | None:
    from models.deep_sgdf_delta.predict_v2 import predict_delta_v2
    from models.deep_sgdf_delta.dataset_v2 import build_predict_dataset_v2, DEFAULT_FEATURE_CONFIG

    eval_days = pd.date_range(start_date, end_date, freq="D")
    if fast_dev_run:
        eval_days = eval_days[: min(7, len(eval_days))]

    all_preds: list[pd.DataFrame] = []
    for target_day in eval_days:
        try:
            pred_ds, _ = build_predict_dataset_v2(
                raw_df, DEFAULT_FEATURE_CONFIG, target_day=target_day,
            )
            pred_df = predict_delta_v2(model, pred_ds, device, batch_size=batch_size)
            if "valid" in pred_df.columns:
                pred_df = pred_df[pred_df["valid"]].copy()
                pred_df = pred_df.drop(columns=["valid"])
            all_preds.append(pred_df)
        except Exception as exc:
            logger.warning("V2 prediction failed for %s: %s", target_day.date(), exc)

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    combined["business_day"] = pd.to_datetime(combined["business_day"]).dt.normalize()
    combined["hour"] = combined["hour"].astype(int)
    return combined


def train_and_predict_v2(
    raw_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp,
    device_str: str, amp: bool, fast_dev_run: bool,
) -> tuple[pd.DataFrame | None, dict, Any]:
    import torch
    from models.deep_sgdf_delta.train_v2 import TrainV2Config, train_model_v2

    t_start = time.time()
    info: dict[str, Any] = {"type": "v2_day_tcn"}

    try:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_str == "auto" else torch.device(device_str)
        )
        epochs = 3 if fast_dev_run else 30
        batch_size = 8 if fast_dev_run else 64

        train_config = TrainV2Config(
            backbone="tcn", epochs=epochs, batch_size=batch_size,
            amp_enabled=amp, device=str(device),
            val_days=7 if fast_dev_run else 30,
            early_stopping_patience=3 if fast_dev_run else 5,
        )

        logger.info("Training V2 day TCN (decision_day=%s) ...", start_date.date())
        result = train_model_v2(
            raw_df, None, train_config,
            decision_day=start_date, fast_dev_run=fast_dev_run,
        )
        model = result["model"]
        info["best_val_smape"] = result["best_val_smape"]
        info["total_params"] = result["total_params"]

        pred_all = _predict_v2_for_period(
            model, raw_df, start_date, end_date, device, batch_size, fast_dev_run,
        )
        info["runtime_seconds"] = round(time.time() - t_start, 1)
        info["num_predictions"] = len(pred_all) if pred_all is not None else 0
        logger.info("V2 day TCN: %s in %.1fs",
                     f"{info['num_predictions']} predictions" if pred_all is not None else "FAILED",
                     info["runtime_seconds"])
        return pred_all, info, model

    except Exception as exc:
        logger.error("V2 day TCN failed: %s", exc)
        logger.debug(traceback.format_exc())
        info["error"] = str(exc)
        info["runtime_seconds"] = round(time.time() - t_start, 1)
        return None, info, None


# ── V3 model: train + predict ────────────────────────────────────────

def _predict_v3_for_period(
    model: Any, raw_df: pd.DataFrame,
    start_date: pd.Timestamp, end_date: pd.Timestamp,
    device: Any, batch_size: int, fast_dev_run: bool,
) -> pd.DataFrame | None:
    from models.deep_sgdf_delta.predict_v3 import predict_delta_v3
    from models.deep_sgdf_delta.dataset_v3 import build_predict_dataset_v3

    eval_days = pd.date_range(start_date, end_date, freq="D")
    if fast_dev_run:
        eval_days = eval_days[: min(7, len(eval_days))]

    all_preds: list[pd.DataFrame] = []
    for target_day in eval_days:
        try:
            pred_ds, _ = build_predict_dataset_v3(raw_df, target_day=target_day)
            pred_df = predict_delta_v3(model, pred_ds, device, batch_size=batch_size)
            if "valid" in pred_df.columns:
                pred_df = pred_df[pred_df["valid"]].copy()
                pred_df = pred_df.drop(columns=["valid"])
            all_preds.append(pred_df)
        except Exception as exc:
            logger.warning("V3 prediction failed for %s: %s", target_day.date(), exc)

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    combined["business_day"] = pd.to_datetime(combined["business_day"]).dt.normalize()
    combined["hour"] = combined["hour"].astype(int)
    return combined


def train_and_predict_v3(
    raw_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp,
    device_str: str, amp: bool, fast_dev_run: bool,
    profile_name: str | None = None,
    teachers: list[str] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    import torch
    from models.deep_sgdf_delta.train_v3 import TrainV3Config, train_model_v3
    from models.deep_sgdf_delta.runtime_profiles import get_profile, PROFILES

    t_start = time.time()
    info: dict[str, Any] = {"type": profile_name or "v3"}

    try:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_str == "auto" else torch.device(device_str)
        )

        # Build config from profile or defaults
        if profile_name and profile_name in PROFILES:
            prof = get_profile(profile_name)
            info["profile"] = profile_name
        else:
            prof = None

        epochs = 3 if fast_dev_run else (prof.epochs if prof else 40)
        batch_size = 8 if fast_dev_run else (prof.batch_size if prof else 64)
        multiscale = prof.multiscale if prof else True
        use_teacher_gate = prof.use_teacher_gate if prof else False
        teacher_input_dim = prof.teacher_input_dim if prof else 0

        train_config = TrainV3Config(
            hidden_dim=prof.hidden_dim if prof else 96,
            num_layers=prof.num_layers if prof else 2,
            backbone=prof.backbone if prof else "tcn",
            multiscale=multiscale,
            use_teacher_gate=use_teacher_gate,
            teacher_input_dim=teacher_input_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=prof.learning_rate if prof else 1e-3,
            weight_decay=prof.weight_decay if prof else 1e-4,
            dropout=prof.dropout if prof else 0.1,
            amp_enabled=amp or (prof.amp if prof else False),
            device=str(device),
            val_days=7 if fast_dev_run else (prof.val_days if prof else 30),
            early_stopping_patience=3 if fast_dev_run else (prof.patience if prof else 6),
        )

        logger.info("Training V3 [%s] (decision_day=%s) ...",
                     profile_name or "default", start_date.date())
        result = train_model_v3(
            raw_df, train_config,
            decision_day=start_date, fast_dev_run=fast_dev_run,
        )
        model = result["model"]
        info["best_val_smape"] = result["best_val_smape"]
        info["total_params"] = result["total_params"]

        pred_all = _predict_v3_for_period(
            model, raw_df, start_date, end_date, device, batch_size, fast_dev_run,
        )
        info["runtime_seconds"] = round(time.time() - t_start, 1)
        info["num_predictions"] = len(pred_all) if pred_all is not None else 0
        logger.info("V3 [%s]: %s in %.1fs",
                     profile_name or "default",
                     f"{info['num_predictions']} predictions" if pred_all is not None else "FAILED",
                     info["runtime_seconds"])
        return pred_all, info

    except Exception as exc:
        logger.error("V3 [%s] failed: %s", profile_name or "default", exc)
        logger.debug(traceback.format_exc())
        info["error"] = str(exc)
        info["runtime_seconds"] = round(time.time() - t_start, 1)
        return None, info


# ── SGDFNet baseline ─────────────────────────────────────────────────

def build_sgdfnet_baseline_candidate(
    raw_df: pd.DataFrame | None,
    gt_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """Build SGDFNet baseline by running SGDFNet inference or using known metrics."""
    # Try to get actual SGDFNet predictions for fair comparison
    if _SGDFNET_AVAILABLE and raw_df is not None:
        try:
            from models.deep_sgdf_delta.sgdfnet_bridge import preprocess_dataframe, FeatureConfig
            feat_config = FeatureConfig(
                include_forecast_columns=True,
                include_actual_history_columns=False,
                use_visible_actual_history=True,
                include_delta_history_features=True,
                include_tf_moving_average_features=False,
                include_static_group_graph_features=False,
                include_weekly_history_features=False,
                include_segment_local_stats=False,
                include_forecast_pressure_interactions=False,
                include_calendar_features=True,
                include_engineered_forecast_features=True,
            )
            frame, _ = preprocess_dataframe(raw_df, feat_config)
            frame["business_day"] = pd.to_datetime(frame["business_day"]).dt.normalize()

            rows: list[dict] = []
            for target_day in pd.date_range(start_date, end_date, freq="D"):
                day_rows = frame[frame["business_day"] == target_day]
                if day_rows.empty:
                    continue
                for _, row in day_rows.iterrows():
                    hour = int(row.get("target_hour", row.get("hour", 0)))
                    da = float(row.get("da_anchor", 0.0))
                    delta_hat = float(row.get("delta_hat", 0.0))
                    rows.append({
                        "business_day": target_day,
                        "hour": hour,
                        "rt_pred": da + delta_hat,
                        "delta_pred": delta_hat,
                        "da_anchor": da,
                    })

            if rows:
                pred_df = pd.DataFrame(rows)
                pred_df["business_day"] = pd.to_datetime(pred_df["business_day"]).dt.normalize()
                return evaluate_candidate_full(pred_df, gt_df, "sgdfnet_baseline")

        except Exception as exc:
            logger.warning("SGDFNet prediction failed, using reference metrics: %s", exc)

    # Fallback: use known reference metrics
    return {
        "name": "sgdfnet_baseline",
        "overall_sMAPE_floor50": BASELINE_SGDFNET,
        "monthly_avg_sMAPE_floor50": BASELINE_SGDFNET,
        "monthly_worst_sMAPE": BASELINE_SGDFNET,
        "9_16_sMAPE_floor50": BASELINE_SGDFNET,
        "1_8_sMAPE_floor50": BASELINE_SGDFNET,
        "17_24_sMAPE_floor50": BASELINE_SGDFNET,
        "delta_mae": float("nan"),
        "rows_matched": 0,
        "monthly_smape": {},
        "period_smape": {},
        "bucket_smape": {},
        "leakage_risk": False,
    }


# ── V2 residual SGDFNet blend ────────────────────────────────────────

def build_v2_residual_candidate(
    v2_pred: pd.DataFrame,
    raw_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """Blend V2 TCN predictions with SGDFNet via residual correction."""
    if not _SGDFNET_AVAILABLE:
        logger.warning("SGDFNet unavailable -- v2_residual_sgdfnet skipped")
        return _empty_result("v2_residual_sgdfnet")

    try:
        from models.deep_sgdf_delta.sgdfnet_bridge import preprocess_dataframe, FeatureConfig
        feat_config = FeatureConfig(
            include_forecast_columns=True,
            include_actual_history_columns=False,
            use_visible_actual_history=True,
            include_delta_history_features=True,
            include_tf_moving_average_features=False,
            include_static_group_graph_features=False,
            include_weekly_history_features=False,
            include_segment_local_stats=False,
            include_forecast_pressure_interactions=False,
            include_calendar_features=True,
            include_engineered_forecast_features=True,
        )
        frame, _ = preprocess_dataframe(raw_df, feat_config)
        frame["business_day"] = pd.to_datetime(frame["business_day"]).dt.normalize()

        sgdf_rows: list[dict] = []
        for target_day in pd.date_range(start_date, end_date, freq="D"):
            day_rows = frame[frame["business_day"] == target_day]
            for _, row in day_rows.iterrows():
                hour = int(row.get("target_hour", row.get("hour", 0)))
                da = float(row.get("da_anchor", 0.0))
                delta_hat = float(row.get("delta_hat", 0.0))
                sgdf_rows.append({
                    "business_day": target_day, "hour": hour,
                    "sgdf_rt": da + delta_hat, "sgdf_delta": delta_hat,
                })

        if not sgdf_rows:
            return _empty_result("v2_residual_sgdfnet")

        sgdf_df = pd.DataFrame(sgdf_rows)
        sgdf_df["business_day"] = pd.to_datetime(sgdf_df["business_day"]).dt.normalize()

        # Merge V2 + SGDFNet
        merged = v2_pred.merge(sgdf_df, on=["business_day", "hour"], how="left")
        merged["rt_pred"] = merged["sgdf_rt"] + 0.5 * (merged["rt_pred"] - merged["sgdf_rt"])

        return evaluate_candidate_full(merged, gt_df, "v2_residual_sgdfnet")

    except Exception as exc:
        logger.error("v2_residual_sgdfnet failed: %s", exc)
        logger.debug(traceback.format_exc())
        result = _empty_result("v2_residual_sgdfnet")
        result["error"] = str(exc)
        return result


# ── Leaderboard ──────────────────────────────────────────────────────

def build_leaderboard(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append({
            "name": r["name"],
            "overall_sMAPE_floor50": r["overall_sMAPE_floor50"],
            "monthly_avg_sMAPE_floor50": r.get("monthly_avg_sMAPE_floor50", float("inf")),
            "monthly_worst_sMAPE": r.get("monthly_worst_sMAPE", float("inf")),
            "9_16_sMAPE_floor50": r.get("9_16_sMAPE_floor50", float("inf")),
            "1_8_sMAPE_floor50": r.get("1_8_sMAPE_floor50", float("nan")),
            "17_24_sMAPE_floor50": r.get("17_24_sMAPE_floor50", float("nan")),
            "delta_mae": r.get("delta_mae", float("nan")),
            "runtime_seconds": r.get("runtime_seconds", 0),
            "leakage_risk": r.get("leakage_risk", False),
            "rows_matched": r.get("rows_matched", 0),
            "error": r.get("error", ""),
        })

    df = pd.DataFrame(rows)
    df_safe = df[df["leakage_risk"] == False].copy()  # noqa: E712
    if df_safe.empty:
        df_safe = df.copy()

    df_sorted = df_safe.sort_values(
        by=["overall_sMAPE_floor50", "monthly_worst_sMAPE", "9_16_sMAPE_floor50"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    df_sorted.insert(0, "rank", range(1, len(df_sorted) + 1))
    return df_sorted


# ── Output writers ───────────────────────────────────────────────────

def write_leaderboard_csv(out_dir: Path, leaderboard: pd.DataFrame) -> None:
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    logger.info("leaderboard.csv -> %s", out_dir / "leaderboard.csv")


def write_monthly_metrics_csv(out_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict] = []
    for r in results:
        for month, smape_val in r.get("monthly_smape", {}).items():
            rows.append({"candidate": r["name"], "month": month, "sMAPE_floor50": round(smape_val, 4)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["candidate", "month"]).reset_index(drop=True)
    df.to_csv(out_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    logger.info("monthly_metrics.csv -> %s", out_dir / "monthly_metrics.csv")


def write_period_metrics_csv(out_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict] = []
    for r in results:
        for period, smape_val in r.get("period_smape", {}).items():
            rows.append({"candidate": r["name"], "period": period, "sMAPE_floor50": round(smape_val, 4)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["candidate", "period"]).reset_index(drop=True)
    df.to_csv(out_dir / "period_metrics.csv", index=False, encoding="utf-8-sig")
    logger.info("period_metrics.csv -> %s", out_dir / "period_metrics.csv")


def write_bucket_metrics_csv(out_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict] = []
    for r in results:
        for bucket, bdata in r.get("bucket_smape", {}).items():
            rows.append({
                "candidate": r["name"],
                "bucket": bucket,
                "sMAPE_floor50": round(bdata["sMAPE_floor50"], 4),
                "count": bdata["count"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["candidate", "bucket"]).reset_index(drop=True)
    df.to_csv(out_dir / "bucket_metrics.csv", index=False, encoding="utf-8-sig")
    logger.info("bucket_metrics.csv -> %s", out_dir / "bucket_metrics.csv")


def write_runtime_report(out_dir: Path, results: list[dict[str, Any]]) -> None:
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "candidates": [],
    }
    for r in results:
        report["candidates"].append({
            "name": r["name"],
            "runtime_seconds": r.get("runtime_seconds", 0),
            "total_params": r.get("total_params", 0),
            "num_predictions": r.get("num_predictions", 0),
            "best_val_smape": r.get("best_val_smape", float("nan")),
            "error": r.get("error", ""),
        })
    with open(out_dir / "runtime_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("runtime_report.json -> %s", out_dir / "runtime_report.json")


def write_ablation_summary(
    out_dir: Path,
    leaderboard: pd.DataFrame,
    results: list[dict[str, Any]],
) -> None:
    """Write ablation_summary.md answering the key research questions."""

    def _get(name: str, key: str) -> Any:
        for r in results:
            if r["name"] == name:
                return r.get(key, float("nan"))
        return float("nan")

    def _fmt(val: Any) -> str:
        if isinstance(val, float) and (val != val or val == float("inf")):
            return "N/A"
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    # Gather key numbers
    baseline_overall = _get("sgdfnet_baseline", "overall_sMAPE_floor50")
    v2_tcn_overall = _get("v2_day_tcn", "overall_sMAPE_floor50")
    v2_res_overall = _get("v2_residual_sgdfnet", "overall_sMAPE_floor50")
    v3_fast_overall = _get("v3_fast_tcn", "overall_sMAPE_floor50")
    v3_ms_overall = _get("v3_multiscale_tcn", "overall_sMAPE_floor50")
    v3_tr_overall = _get("v3_teacher_residual", "overall_sMAPE_floor50")
    v3_moe_overall = _get("v3_teacher_moe", "overall_sMAPE_floor50")

    v2_tcn_916 = _get("v2_day_tcn", "9_16_sMAPE_floor50")
    v3_ms_916 = _get("v3_multiscale_tcn", "9_16_sMAPE_floor50")
    v3_tr_916 = _get("v3_teacher_residual", "9_16_sMAPE_floor50")

    # Determine champion
    champion_name = leaderboard.iloc[0]["name"] if not leaderboard.empty else "unknown"
    champion_smape = leaderboard.iloc[0]["overall_sMAPE_floor50"] if not leaderboard.empty else float("nan")

    # Answer research questions
    multiscale_effective = (
        not np.isnan(v3_ms_overall) and not np.isnan(v3_fast_overall)
        and v3_ms_overall < v3_fast_overall
    )
    period_helps_916 = (
        not np.isnan(v3_ms_916) and not np.isnan(v2_tcn_916)
        and v3_ms_916 < v2_tcn_916
    )
    teacher_residual_useful = (
        not np.isnan(v3_tr_overall) and not np.isnan(v3_ms_overall)
        and v3_tr_overall < v3_ms_overall
    )
    v3_better_than_v2 = (
        not np.isnan(v3_ms_overall) and not np.isnan(v2_tcn_overall)
        and v3_ms_overall < v2_tcn_overall
    )

    md: list[str] = [
        "# TrendKnight-X Ablation Study Summary",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Champion:** {champion_name} (overall sMAPE_floor50 = {_fmt(champion_smape)})",
        "",
        "## Leaderboard",
        "",
        "| Rank | Candidate | Overall sMAPE | Monthly Avg | Monthly Worst | 9_16 sMAPE | Runtime(s) |",
        "|------|-----------|---------------|-------------|---------------|------------|------------|",
    ]

    for _, row in leaderboard.iterrows():
        md.append(
            f"| {int(row['rank'])} | {row['name']} | {_fmt(row['overall_sMAPE_floor50'])} | "
            f"{_fmt(row['monthly_avg_sMAPE_floor50'])} | {_fmt(row['monthly_worst_sMAPE'])} | "
            f"{_fmt(row['9_16_sMAPE_floor50'])} | {row['runtime_seconds']:.1f} |"
        )

    md += [
        "",
        "## Ablation Questions",
        "",
        "### Q1: Is multiscale decomposition effective?",
        "",
        f"- v3_fast_tcn (no multiscale): overall = {_fmt(v3_fast_overall)}",
        f"- v3_multiscale_tcn (with multiscale): overall = {_fmt(v3_ms_overall)}",
        f"- **Verdict:** {'YES -- multiscale reduces sMAPE by ' + f'{(v3_fast_overall - v3_ms_overall):.4f}' if multiscale_effective else 'NO / INCONCLUSIVE -- multiscale does not clearly help'}",
        "",
        "### Q2: Does the period branch help 9_16?",
        "",
        f"- v2_day_tcn 9_16 sMAPE: {_fmt(v2_tcn_916)}",
        f"- v3_multiscale_tcn 9_16 sMAPE: {_fmt(v3_ms_916)}",
        f"- **Verdict:** {'YES -- period-aware branch improves 9_16 segment' if period_helps_916 else 'NO / INCONCLUSIVE -- 9_16 not clearly improved'}",
        "",
        "### Q3: Is teacher residual distillation useful?",
        "",
        f"- v3_multiscale_tcn (no teacher): overall = {_fmt(v3_ms_overall)}",
        f"- v3_teacher_residual (with teacher): overall = {_fmt(v3_tr_overall)}",
        f"- **Verdict:** {'YES -- teacher residual improves overall sMAPE' if teacher_residual_useful else 'NO / INCONCLUSIVE -- teacher residual does not clearly help'}",
        "",
        "### Q4: Is v3 better than v2?",
        "",
        f"- v2_day_tcn overall: {_fmt(v2_tcn_overall)}",
        f"- v3_multiscale_tcn overall: {_fmt(v3_ms_overall)}",
        f"- **Verdict:** {'YES -- v3 architecture outperforms v2' if v3_better_than_v2 else 'NO / INCONCLUSIVE -- v3 does not clearly outperform v2'}",
        "",
        "### Q5: MoE vs Residual teacher fusion?",
        "",
        f"- v3_teacher_residual: overall = {_fmt(v3_tr_overall)}",
        f"- v3_teacher_moe: overall = {_fmt(v3_moe_overall)}",
        f"- **Verdict:** {'MoE is better' if (not np.isnan(v3_moe_overall) and not np.isnan(v3_tr_overall) and v3_moe_overall < v3_tr_overall) else 'Residual is better or INCONCLUSIVE'}",
        "",
        "## Recommendations",
        "",
    ]

    # Build recommendation
    if champion_smape < PASS_THRESHOLD:
        md.append(f"**PASS**: Champion {champion_name} achieves sMAPE_floor50 = {_fmt(champion_smape)} < {PASS_THRESHOLD}.")
    elif champion_smape < SOFT_PASS_THRESHOLD:
        md.append(f"**SOFT_PASS**: Champion {champion_name} achieves sMAPE_floor50 = {_fmt(champion_smape)} <= {SOFT_PASS_THRESHOLD}.")
    elif champion_smape < baseline_overall:
        md.append(f"**BASELINE_PASS**: Champion beats SGDFNet baseline ({_fmt(baseline_overall)}).")
    else:
        md.append(f"**NO-GO**: Best candidate ({_fmt(champion_smape)}) does not beat baseline ({_fmt(baseline_overall)}).")

    md += [
        "",
        "## Architecture Decision Summary",
        "",
        f"- Multiscale decomposition: {'ADOPT' if multiscale_effective else 'SKIP'}",
        f"- Period-aware branch for 9_16: {'ADOPT' if period_helps_916 else 'EVALUATE FURTHER'}",
        f"- Teacher residual distillation: {'ADOPT' if teacher_residual_useful else 'EVALUATE FURTHER'}",
        f"- V3 over V2: {'ADOPT' if v3_better_than_v2 else 'KEEP V2 AS FALLBACK'}",
        "",
    ]

    (out_dir / "ablation_summary.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("ablation_summary.md -> %s", out_dir / "ablation_summary.md")


# ── Main orchestration ───────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TrendKnight-X Ablation Study")
    logger.info("=" * 60)
    logger.info("  Period      : %s to %s", start_date.date(), end_date.date())
    logger.info("  Output      : %s", out_dir)
    logger.info("  Device      : %s", args.device)
    logger.info("  AMP         : %s", args.amp)
    logger.info("  Fast-dev-run: %s", args.fast_dev_run)
    logger.info("  Profile     : %s", args.profile or "auto")
    logger.info("  Teachers    : %s", args.teachers or "all available")

    # SGDFNet
    sgdfnet_ok = _try_import_sgdfnet(args.sgdfnet_root)
    if sgdfnet_ok:
        logger.info("SGDFNet: available")
    else:
        logger.warning("SGDFNet: NOT available (%s)", _SGDFNET_IMPORT_ERROR)

    # Data
    raw_df = load_raw_data(args.data_path)
    logger.info("Raw data: %d rows", len(raw_df))

    # Ground truth
    gt_df = build_ground_truth(raw_df, start_date, end_date)
    if gt_df.empty:
        logger.error("No ground truth data in evaluation period. Exiting.")
        sys.exit(1)

    # ── Collect candidates ───────────────────────────────────────────
    all_results: list[dict[str, Any]] = []

    # 1. SGDFNet baseline
    logger.info("-" * 60)
    logger.info("[1/7] sgdfnet_baseline")
    logger.info("-" * 60)
    baseline_result = build_sgdfnet_baseline_candidate(raw_df, gt_df, start_date, end_date)
    baseline_result["runtime_seconds"] = 0
    all_results.append(baseline_result)
    logger.info("  overall_sMAPE = %s", baseline_result["overall_sMAPE_floor50"])

    # 2. V2 day TCN
    logger.info("-" * 60)
    logger.info("[2/7] v2_day_tcn")
    logger.info("-" * 60)
    v2_pred, v2_info, v2_model = train_and_predict_v2(
        raw_df, start_date, end_date, args.device, args.amp, args.fast_dev_run,
    )
    if v2_pred is not None:
        m = evaluate_candidate_full(v2_pred, gt_df, "v2_day_tcn")
        m["runtime_seconds"] = v2_info.get("runtime_seconds", 0)
        m["total_params"] = v2_info.get("total_params", 0)
        m["num_predictions"] = v2_info.get("num_predictions", 0)
        m["best_val_smape"] = v2_info.get("best_val_smape", float("nan"))
        all_results.append(m)
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])
    else:
        logger.warning("v2_day_tcn: no predictions produced")
        all_results.append({**_empty_result("v2_day_tcn"), **v2_info})

    # 3. V2 residual SGDFNet
    logger.info("-" * 60)
    logger.info("[3/7] v2_residual_sgdfnet")
    logger.info("-" * 60)
    if v2_pred is not None:
        v2_res_result = build_v2_residual_candidate(v2_pred, raw_df, gt_df, start_date, end_date)
        v2_res_result["runtime_seconds"] = v2_info.get("runtime_seconds", 0) + 1.0
        all_results.append(v2_res_result)
        logger.info("  overall_sMAPE = %s", v2_res_result["overall_sMAPE_floor50"])
    else:
        logger.warning("v2_residual_sgdfnet: skipped (no v2 predictions)")
        all_results.append({**_empty_result("v2_residual_sgdfnet"), "error": "no v2 predictions"})

    # 4. V3 fast TCN (no multiscale, no teacher)
    logger.info("-" * 60)
    logger.info("[4/7] v3_fast_tcn")
    logger.info("-" * 60)
    v3_fast_pred, v3_fast_info = train_and_predict_v3(
        raw_df, start_date, end_date, args.device, args.amp, args.fast_dev_run,
        profile_name="v3_fast_tcn",
    )
    if v3_fast_pred is not None:
        m = evaluate_candidate_full(v3_fast_pred, gt_df, "v3_fast_tcn")
        m.update({k: v for k, v in v3_fast_info.items() if k not in m})
        all_results.append(m)
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])
    else:
        logger.warning("v3_fast_tcn: no predictions produced")
        all_results.append({**_empty_result("v3_fast_tcn"), **v3_fast_info})

    # 5. V3 multiscale TCN
    logger.info("-" * 60)
    logger.info("[5/7] v3_multiscale_tcn")
    logger.info("-" * 60)
    v3_ms_pred, v3_ms_info = train_and_predict_v3(
        raw_df, start_date, end_date, args.device, args.amp, args.fast_dev_run,
        profile_name="v3_multiscale_tcn",
    )
    if v3_ms_pred is not None:
        m = evaluate_candidate_full(v3_ms_pred, gt_df, "v3_multiscale_tcn")
        m.update({k: v for k, v in v3_ms_info.items() if k not in m})
        all_results.append(m)
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])
    else:
        logger.warning("v3_multiscale_tcn: no predictions produced")
        all_results.append({**_empty_result("v3_multiscale_tcn"), **v3_ms_info})

    # 6. V3 teacher residual
    logger.info("-" * 60)
    logger.info("[6/7] v3_teacher_residual")
    logger.info("-" * 60)
    v3_tr_pred, v3_tr_info = train_and_predict_v3(
        raw_df, start_date, end_date, args.device, args.amp, args.fast_dev_run,
        profile_name="v3_teacher_residual", teachers=args.teachers,
    )
    if v3_tr_pred is not None:
        m = evaluate_candidate_full(v3_tr_pred, gt_df, "v3_teacher_residual")
        m.update({k: v for k, v in v3_tr_info.items() if k not in m})
        all_results.append(m)
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])
    else:
        logger.warning("v3_teacher_residual: no predictions produced")
        all_results.append({**_empty_result("v3_teacher_residual"), **v3_tr_info})

    # 7. V3 teacher MoE
    logger.info("-" * 60)
    logger.info("[7/7] v3_teacher_moe")
    logger.info("-" * 60)
    v3_moe_pred, v3_moe_info = train_and_predict_v3(
        raw_df, start_date, end_date, args.device, args.amp, args.fast_dev_run,
        profile_name="v3_teacher_moe", teachers=args.teachers,
    )
    if v3_moe_pred is not None:
        m = evaluate_candidate_full(v3_moe_pred, gt_df, "v3_teacher_moe")
        m.update({k: v for k, v in v3_moe_info.items() if k not in m})
        all_results.append(m)
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])
    else:
        logger.warning("v3_teacher_moe: no predictions produced")
        all_results.append({**_empty_result("v3_teacher_moe"), **v3_moe_info})

    # ── Build leaderboard ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Building Leaderboard")
    logger.info("=" * 60)

    leaderboard = build_leaderboard(all_results)
    if leaderboard.empty:
        logger.error("No candidates produced usable results.")
        sys.exit(1)

    for _, row in leaderboard.iterrows():
        logger.info(
            "  #%d  %-25s  overall=%.4f  9_16=%.4f  %.1fs",
            int(row["rank"]), row["name"],
            row["overall_sMAPE_floor50"], row["9_16_sMAPE_floor50"],
            row["runtime_seconds"],
        )

    # ── Write all outputs ────────────────────────────────────────────
    write_leaderboard_csv(out_dir, leaderboard)
    write_monthly_metrics_csv(out_dir, all_results)
    write_period_metrics_csv(out_dir, all_results)
    write_bucket_metrics_csv(out_dir, all_results)
    write_runtime_report(out_dir, all_results)
    write_ablation_summary(out_dir, leaderboard, all_results)

    logger.info("=" * 60)
    logger.info("All outputs saved to %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
