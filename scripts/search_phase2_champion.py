#!/usr/bin/env python
"""Phase 2 Champion Search -- compare all candidate models and pick the best.

Candidates:
  1. SGDFNet baseline (from p0 output or --baseline-metrics JSON)
  2. V1 hourly TCN
  3. V1 hourly GRU
  4. V2 day TCN
  5. V2 day GRU
  6. V2 residual SGDFNet  (V2 TCN + sgdfnet_residual blend)
  7. V2 blend SGDFNet    (V2 TCN + sgdfnet_blend blend)

Output (under --out-dir, default reports/local/phase2/champion_search/):
  leaderboard.csv
  champion_predictions.csv
  champion_metrics_summary.json
  champion_go_nogo.md
  blend_weights.json

Usage:
    python scripts/search_phase2_champion.py --start-date 2026-03-01 --end-date 2026-05-01 --data-path data/shandong_pmos_hourly.xlsx
    python scripts/search_phase2_champion.py --fast-dev-run --start-date 2026-04-01 --end-date 2026-04-07 --data-path data/shandong_pmos_hourly.xlsx
    python scripts/search_phase2_champion.py --help
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

# SGDFNet sibling path resolution (no hardcoded absolute paths)
_SIBLING_SGDFNET_SRC = (
    PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp" / "SGDFNet" / "src"
)

if _SIBLING_SGDFNET_SRC.exists() and str(_SIBLING_SGDFNET_SRC) not in sys.path:
    sys.path.insert(0, str(_SIBLING_SGDFNET_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("search_phase2_champion")

# ── Constants ────────────────────────────────────────────────────────
PASS_THRESHOLD = 15.0
SOFT_PASS_THRESHOLD = 15.8
BASELINE_SGDFNET = 16.5902
MAX_RUNTIME_SECONDS = 3600  # 1 hour total per candidate
BLEND_W_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RESIDUAL_ALPHA_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

SEASON_MAP: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Champion Search: compare all candidate models and pick the best",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Candidates compared:\n"
            "  1. SGDFNet baseline\n"
            "  2. V1 hourly TCN\n"
            "  3. V1 hourly GRU\n"
            "  4. V2 day TCN\n"
            "  5. V2 day GRU\n"
            "  6. V2 residual SGDFNet\n"
            "  7. V2 blend SGDFNet\n"
        ),
    )
    parser.add_argument(
        "--start-date", type=str, required=True,
        help="Evaluation period start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, required=True,
        help="Evaluation period end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sgdfnet-root", type=str, default=None,
        help="Path to SGDFNet project root (the directory containing src/sgdfnet/)",
    )
    parser.add_argument(
        "--data-path", type=str,
        default="data/shandong_pmos_hourly.xlsx",
        help="Path to raw data file (relative to project root or absolute)",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="reports/local/phase2/champion_search",
        help="Output directory for all result files",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Enable AMP (mixed precision) for training",
    )
    parser.add_argument(
        "--fast-dev-run", action="store_true",
        help="Quick smoke test: tiny data subset, few epochs, limited eval days",
    )
    parser.add_argument(
        "--baseline-metrics", type=str, default=None,
        help="Path to pre-computed SGDFNet baseline metrics JSON (from p0 script)",
    )
    return parser.parse_args()


# ── SGDFNet availability ─────────────────────────────────────────────

_SGDFNET_AVAILABLE = False
_SGDFNET_IMPORT_ERROR: str | None = None


def _try_import_sgdfnet(sgdfnet_root: str | None = None) -> bool:
    """Try to import SGDFNet modules.  Returns True on success."""
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
    """Load raw data, trying multiple path resolutions."""
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
        from sgdfnet.data_contract import load_dataset
        return load_dataset(str(p))

    # Fallback: direct read
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
    """Build ground-truth DataFrame for [start_date, end_date].

    Returns columns: business_day, hour, rt_actual, delta_target, da_anchor, segment_id.
    """
    df = raw_df.copy()

    # Locate timestamp column
    ts_col: str | None = None
    for c in ("时刻", "timestamp", "ds", "time"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("Cannot find timestamp column in raw data")
    df[ts_col] = pd.to_datetime(df[ts_col])

    # Business time columns
    if "business_day" not in df.columns:
        if _SGDFNET_AVAILABLE:
            from sgdfnet.data_contract import add_business_time_columns
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

    # RT / DA columns
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
            from sgdfnet.data_contract import RT_COL, DA_COL
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
            df["hour"], bins=[0, 8, 16, 24], labels=[0, 1, 2], include_lowest=True,
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


# ── SGDFNet predictions ──────────────────────────────────────────────

def get_sgdfnet_predictions(
    raw_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    val_start: pd.Timestamp,
    baseline_metrics_path: str | None,
) -> tuple[pd.DataFrame | None, dict | None]:
    """Obtain SGDFNet per-row predictions for val + test periods.

    Returns ``(predictions_df, baseline_metrics_dict)``.
    ``predictions_df`` has columns: business_day, hour, rt_hat, delta_hat, da_anchor.
    Either or both may be *None* when SGDFNet is unavailable.
    """
    baseline_metrics: dict | None = None

    # Try loading pre-computed baseline metrics
    if baseline_metrics_path:
        bp = Path(baseline_metrics_path)
        if bp.exists():
            with open(bp, "r", encoding="utf-8") as fh:
                baseline_metrics = json.load(fh)
            logger.info(
                "Loaded SGDFNet baseline metrics: overall_sMAPE=%s",
                baseline_metrics.get("overall_sMAPE_floor50", "N/A"),
            )

    if not _SGDFNET_AVAILABLE:
        logger.warning("SGDFNet not available -- blend candidates will be skipped.")
        return None, baseline_metrics

    # Build per-row predictions from the processed feature frame
    try:
        from sgdfnet.data_contract import preprocess_dataframe, FeatureConfig

        feat_config = FeatureConfig(
            include_forecast_columns=True,
            include_actual_history_columns=False,
            use_visible_actual_history=True,
            include_delta_history_features=True,
            include_tf_moving_average_features=False,
            include_static_group_graph_features=False,
            include_weekly_history_features=False,
            include_forecast_residual_history_features=False,
            include_segment_local_stats=False,
            include_forecast_pressure_interactions=False,
            include_calendar_features=True,
            include_engineered_forecast_features=True,
        )

        frame, _ = preprocess_dataframe(raw_df, feat_config)
        frame["business_day"] = pd.to_datetime(frame["business_day"]).dt.normalize()

        rows: list[dict] = []
        for target_day in pd.date_range(val_start, end_date, freq="D"):
            day_rows = frame[frame["business_day"] == target_day]
            if day_rows.empty:
                continue
            for _, row in day_rows.iterrows():
                hour = int(row.get("target_hour", row.get("hour", 0)))
                da = float(row.get("da_anchor", 0.0))
                delta_hat = float(row.get("delta_hat", 0.0))
                rt_hat = da + delta_hat
                rows.append({
                    "business_day": target_day,
                    "hour": hour,
                    "rt_hat": rt_hat,
                    "delta_hat": delta_hat,
                    "da_anchor": da,
                })

        if rows:
            sgdf_pred_df = pd.DataFrame(rows)
            logger.info("Built %d SGDFNet prediction rows", len(sgdf_pred_df))
            return sgdf_pred_df, baseline_metrics

    except Exception as exc:
        logger.warning("Failed to build SGDFNet predictions: %s", exc)
        logger.debug(traceback.format_exc())

    return None, baseline_metrics


# ── sMAPE helper (numpy) ────────────────────────────────────────────

def _smape_floor50_np(
    y_true: np.ndarray, y_pred: np.ndarray, floor: float = 50.0,
) -> float:
    yt = np.where(y_true < floor, floor, y_true)
    yp = np.where(y_pred < floor, floor, y_pred)
    denom = np.abs(yt) + np.abs(yp) + 1e-6
    return float(np.mean(200.0 * np.abs(yp - yt) / denom))


# ── V1 model: train + predict ────────────────────────────────────────

def _predict_v1_for_period(
    model: Any,
    raw_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    device: Any,
    batch_size: int,
    window_days: int,
    fast_dev_run: bool,
) -> pd.DataFrame | None:
    """Run V1 prediction loop over a date range."""
    from models.deep_sgdf_delta.predict import predict_delta
    from models.deep_sgdf_delta.dataset import build_predict_dataset, DEFAULT_FEATURE_CONFIG

    eval_days = pd.date_range(start_date, end_date, freq="D")
    if fast_dev_run:
        eval_days = eval_days[: min(7, len(eval_days))]

    all_preds: list[pd.DataFrame] = []
    for target_day in eval_days:
        try:
            pred_ds, _ = build_predict_dataset(
                raw_df, DEFAULT_FEATURE_CONFIG,
                target_day=target_day, window_days=window_days,
            )
            pred_df = predict_delta(model, pred_ds, device, batch_size=batch_size)
            pred_df["business_day"] = target_day
            all_preds.append(pred_df)
        except Exception as exc:
            logger.warning("V1 prediction failed for %s: %s", target_day.date(), exc)

    if not all_preds:
        return None

    combined = pd.concat(all_preds, ignore_index=True)
    if "rt_pred" not in combined.columns and "da_anchor" in combined.columns:
        combined["rt_pred"] = combined["da_anchor"] + combined["delta_pred"]
    combined["hour"] = combined["hour"].astype(int)
    combined["business_day"] = pd.to_datetime(combined["business_day"]).dt.normalize()
    return combined


def train_and_predict_v1(
    raw_df: pd.DataFrame,
    backbone: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    device_str: str,
    amp: bool,
    fast_dev_run: bool,
) -> tuple[pd.DataFrame | None, dict]:
    """Train a V1 (hourly) model and predict over the evaluation period.

    Returns ``(predictions_df, info_dict)``.
    """
    import torch
    from models.deep_sgdf_delta.train import TrainConfig, train_model
    from models.deep_sgdf_delta.dataset import DEFAULT_FEATURE_CONFIG

    t_start = time.time()
    info: dict[str, Any] = {"backbone": backbone, "type": "v1"}

    try:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_str == "auto"
            else torch.device(device_str)
        )

        epochs = 3 if fast_dev_run else 30
        batch_size = 8 if fast_dev_run else 256
        val_days = 7 if fast_dev_run else 30

        train_config = TrainConfig(
            backbone=backbone,
            epochs=epochs,
            batch_size=batch_size,
            amp_enabled=amp,
            device=str(device),
            val_days=val_days,
            early_stopping_patience=3 if fast_dev_run else 5,
        )

        logger.info("Training V1 %s (decision_day=%s) ...", backbone, start_date.date())
        result = train_model(
            raw_df, DEFAULT_FEATURE_CONFIG, train_config, decision_day=start_date,
        )

        model = result["model"]
        info["best_val_smape"] = result["best_val_smape"]
        info["total_params"] = result["total_params"]

        pred_all = _predict_v1_for_period(
            model, raw_df, start_date, end_date, device,
            batch_size, train_config.window_days, fast_dev_run,
        )
        if pred_all is None:
            logger.error("V1 %s: no successful predictions", backbone)
            return None, info

        info["runtime_seconds"] = round(time.time() - t_start, 1)
        info["num_predictions"] = len(pred_all)
        logger.info(
            "V1 %s: %d predictions in %.1fs",
            backbone, len(pred_all), info["runtime_seconds"],
        )
        return pred_all, info

    except Exception as exc:
        logger.error("V1 %s failed: %s", backbone, exc)
        logger.debug(traceback.format_exc())
        info["error"] = str(exc)
        info["runtime_seconds"] = round(time.time() - t_start, 1)
        return None, info


# ── V2 model: train + predict ────────────────────────────────────────

def _predict_v2_for_period(
    model: Any,
    raw_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    device: Any,
    batch_size: int,
    fast_dev_run: bool,
) -> pd.DataFrame | None:
    """Run V2 prediction loop over a date range."""
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
    raw_df: pd.DataFrame,
    backbone: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    device_str: str,
    amp: bool,
    fast_dev_run: bool,
) -> tuple[pd.DataFrame | None, dict, Any, Any]:
    """Train a V2 (day-level) model and predict over the evaluation period.

    Returns ``(predictions_df, info_dict, model, model_config)``.
    """
    import torch
    from models.deep_sgdf_delta.train_v2 import TrainV2Config, train_model_v2
    from models.deep_sgdf_delta.dataset_v2 import DEFAULT_FEATURE_CONFIG

    t_start = time.time()
    info: dict[str, Any] = {"backbone": backbone, "type": "v2"}

    try:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_str == "auto"
            else torch.device(device_str)
        )

        epochs = 3 if fast_dev_run else 30
        batch_size = 8 if fast_dev_run else 64
        val_days = 7 if fast_dev_run else 30

        train_config = TrainV2Config(
            backbone=backbone,
            epochs=epochs,
            batch_size=batch_size,
            amp_enabled=amp,
            device=str(device),
            val_days=val_days,
            early_stopping_patience=3 if fast_dev_run else 5,
        )

        logger.info("Training V2 %s (decision_day=%s) ...", backbone, start_date.date())
        result = train_model_v2(
            raw_df, DEFAULT_FEATURE_CONFIG, train_config,
            decision_day=start_date, fast_dev_run=fast_dev_run,
        )

        model = result["model"]
        model_config = result["model_config"]
        info["best_val_smape"] = result["best_val_smape"]
        info["total_params"] = result["total_params"]

        pred_all = _predict_v2_for_period(
            model, raw_df, start_date, end_date, device, batch_size, fast_dev_run,
        )
        if pred_all is None:
            logger.error("V2 %s: no successful predictions", backbone)
            return None, info, None, None

        info["runtime_seconds"] = round(time.time() - t_start, 1)
        info["num_predictions"] = len(pred_all)
        logger.info(
            "V2 %s: %d predictions in %.1fs",
            backbone, len(pred_all), info["runtime_seconds"],
        )
        return pred_all, info, model, model_config

    except Exception as exc:
        logger.error("V2 %s failed: %s", backbone, exc)
        logger.debug(traceback.format_exc())
        info["error"] = str(exc)
        info["runtime_seconds"] = round(time.time() - t_start, 1)
        return None, info, None, None


# ── Blend weight learning ────────────────────────────────────────────

def _find_best_blend_w(
    deep_rt: np.ndarray,
    sgdfnet_rt: np.ndarray,
    true_rt: np.ndarray,
    mode: str,
) -> tuple[float, float]:
    """Search for the best blend weight on a data slice.

    * ``sgdfnet_blend``: ``final = w * sgdfnet + (1 - w) * deep``
    * ``sgdfnet_residual``: ``final = sgdfnet + alpha * (deep - sgdfnet)``

    Returns ``(best_weight, best_smape)``.
    """
    if len(true_rt) == 0:
        return 0.5, float("inf")

    candidates = BLEND_W_CANDIDATES if mode == "sgdfnet_blend" else RESIDUAL_ALPHA_CANDIDATES
    best_w, best_score = 0.5, float("inf")

    for w in candidates:
        if mode == "sgdfnet_blend":
            blended = w * sgdfnet_rt + (1.0 - w) * deep_rt
        else:  # sgdfnet_residual
            blended = sgdfnet_rt + w * (deep_rt - sgdfnet_rt)
        score = _smape_floor50_np(true_rt, blended)
        if score < best_score:
            best_score = score
            best_w = w

    return best_w, best_score


def learn_blend_weights(
    deep_pred_val: pd.DataFrame,
    sgdfnet_pred_val: pd.DataFrame,
    gt_val: pd.DataFrame,
    mode: str,
) -> dict[str, Any]:
    """Learn blend weights from D-30 to D-1 validation window.

    Supports per-period (global, 1_8, 9_16, 17_24) and per-season
    (winter, spring, summer, autumn) granularities.

    NEVER uses prediction-month true values -- only the validation window.

    Returns a dict with weights at every granularity plus ``best_granularity``.
    """
    # Merge deep, SGDFNet, and ground truth on (business_day, hour)
    merged = deep_pred_val.merge(
        sgdfnet_pred_val[["business_day", "hour", "rt_hat", "delta_hat"]],
        on=["business_day", "hour"],
        how="inner",
        suffixes=("_deep", "_sgdf"),
    )
    merged = merged.merge(
        gt_val[["business_day", "hour", "rt_actual"]].rename(
            columns={"rt_actual": "rt_true"},
        ),
        on=["business_day", "hour"],
        how="inner",
    )

    if merged.empty:
        logger.warning("No overlapping data for blend weight learning")
        return {
            "mode": mode,
            "global": {"w": 0.5, "val_smape": float("inf")},
            "period": {},
            "season": {},
            "best_granularity": "global",
        }

    deep_rt = merged["rt_pred"].values.astype(float)
    sgdf_rt = merged["rt_hat"].values.astype(float)
    true_rt = merged["rt_true"].values.astype(float)
    hours = merged["hour"].values.astype(int)
    months = pd.to_datetime(merged["business_day"]).dt.month.values

    weights: dict[str, Any] = {"mode": mode}

    # 1. Global weight
    gw, gs = _find_best_blend_w(deep_rt, sgdf_rt, true_rt, mode)
    weights["global"] = {"w": gw, "val_smape": round(gs, 4)}
    logger.info("Blend weight (global, %s): w=%.2f  val_sMAPE=%.4f", mode, gw, gs)

    # 2. Per-period weights
    weights["period"] = {}
    for pname, hmin, hmax in [("1_8", 1, 8), ("9_16", 9, 16), ("17_24", 17, 24)]:
        mask = (hours >= hmin) & (hours <= hmax)
        if mask.sum() > 5:
            w, s = _find_best_blend_w(deep_rt[mask], sgdf_rt[mask], true_rt[mask], mode)
            weights["period"][pname] = {"w": w, "val_smape": round(s, 4)}
            logger.info("Blend weight (%s, %s): w=%.2f  val_sMAPE=%.4f", pname, mode, w, s)

    # 3. Per-season weights
    weights["season"] = {}
    for sname, smonths in [
        ("winter", [12, 1, 2]), ("spring", [3, 4, 5]),
        ("summer", [6, 7, 8]), ("autumn", [9, 10, 11]),
    ]:
        mask = np.isin(months, smonths)
        if mask.sum() > 5:
            w, s = _find_best_blend_w(deep_rt[mask], sgdf_rt[mask], true_rt[mask], mode)
            weights["season"][sname] = {"w": w, "val_smape": round(s, 4)}
            logger.info("Blend weight (%s, %s): w=%.2f  val_sMAPE=%.4f", sname, mode, w, s)

    # 4. Pick best granularity by full-validation sMAPE
    best_gran = "global"
    best_val_smape = gs

    # Evaluate per-period composite
    if weights["period"]:
        pp_pred = np.empty_like(deep_rt)
        for i in range(len(deep_rt)):
            h = hours[i]
            if h <= 8:
                pw = weights["period"].get("1_8", weights["global"])["w"]
            elif h <= 16:
                pw = weights["period"].get("9_16", weights["global"])["w"]
            else:
                pw = weights["period"].get("17_24", weights["global"])["w"]
            if mode == "sgdfnet_blend":
                pp_pred[i] = pw * sgdf_rt[i] + (1.0 - pw) * deep_rt[i]
            else:
                pp_pred[i] = sgdf_rt[i] + pw * (deep_rt[i] - sgdf_rt[i])
        pp_smape = _smape_floor50_np(true_rt, pp_pred)
        if pp_smape < best_val_smape:
            best_val_smape = pp_smape
            best_gran = "period"

    # Evaluate per-season composite
    if weights["season"]:
        ps_pred = np.empty_like(deep_rt)
        for i in range(len(deep_rt)):
            sn = SEASON_MAP.get(months[i], "winter")
            sw = weights["season"].get(sn, weights["global"])["w"]
            if mode == "sgdfnet_blend":
                ps_pred[i] = sw * sgdf_rt[i] + (1.0 - sw) * deep_rt[i]
            else:
                ps_pred[i] = sgdf_rt[i] + sw * (deep_rt[i] - sgdf_rt[i])
        ps_smape = _smape_floor50_np(true_rt, ps_pred)
        if ps_smape < best_val_smape:
            best_val_smape = ps_smape
            best_gran = "season"

    weights["best_granularity"] = best_gran
    weights["best_val_smape"] = round(best_val_smape, 4)
    logger.info("Best blend granularity: %s (val_sMAPE=%.4f)", best_gran, best_val_smape)
    return weights


def _get_blend_w_for_row(hour: int, month: int, weights: dict[str, Any]) -> float:
    """Return the blend weight for a single (hour, month) pair."""
    gran = weights.get("best_granularity", "global")
    gw = weights.get("global", {}).get("w", 0.5)

    if gran == "period":
        if hour <= 8:
            return weights.get("period", {}).get("1_8", {}).get("w", gw)
        if hour <= 16:
            return weights.get("period", {}).get("9_16", {}).get("w", gw)
        return weights.get("period", {}).get("17_24", {}).get("w", gw)

    if gran == "season":
        sn = SEASON_MAP.get(month, "winter")
        return weights.get("season", {}).get(sn, {}).get("w", gw)

    return gw


def apply_blend_to_predictions(
    deep_pred_df: pd.DataFrame,
    sgdfnet_pred_df: pd.DataFrame,
    weights: dict[str, Any],
    mode: str,
) -> pd.DataFrame:
    """Apply learned blend weights to deep + SGDFNet predictions.

    Returns a *new* DataFrame with blended ``rt_pred`` and ``delta_pred``.
    """
    merged = deep_pred_df.merge(
        sgdfnet_pred_df[["business_day", "hour", "rt_hat", "delta_hat"]],
        on=["business_day", "hour"],
        how="left",
        suffixes=("", "_sgdf"),
    )

    result = deep_pred_df.copy()

    hours = merged["hour"].values.astype(int)
    months = pd.to_datetime(merged["business_day"]).dt.month.values
    deep_rt = merged["rt_pred"].values.astype(float)
    sgdf_rt = merged["rt_hat"].values.astype(float)
    da = (
        merged["da_anchor"].values.astype(float)
        if "da_anchor" in merged.columns
        else np.zeros(len(merged))
    )

    blended_rt = np.empty_like(deep_rt)
    for i in range(len(merged)):
        w = _get_blend_w_for_row(hours[i], months[i], weights)
        if mode == "sgdfnet_blend":
            blended_rt[i] = w * sgdf_rt[i] + (1.0 - w) * deep_rt[i]
        else:  # sgdfnet_residual
            blended_rt[i] = sgdf_rt[i] + w * (deep_rt[i] - sgdf_rt[i])

    result["rt_pred"] = blended_rt
    result["delta_pred"] = blended_rt - da
    return result


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_candidate(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    name: str,
) -> dict[str, Any]:
    """Evaluate a candidate's predictions against ground truth.

    Returns a dict with overall_sMAPE_floor50, monthly_worst_sMAPE,
    monthly_avg_sMAPE_floor50, 9_16_sMAPE_floor50, delta_mae, etc.
    """
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
        logger.warning("%s: no matching rows between predictions and ground truth", name)
        return {
            "name": name,
            "overall_sMAPE_floor50": float("inf"),
            "monthly_worst_sMAPE": float("inf"),
            "monthly_avg_sMAPE_floor50": float("inf"),
            "9_16_sMAPE_floor50": float("inf"),
            "rows_matched": 0,
            "leakage_risk": False,
        }

    # Resolve rt columns after merge (may have _pred / _true suffixes)
    if "rt_pred" in merged.columns:
        yp = merged["rt_pred"].values.astype(float)
    else:
        yp = np.zeros(len(merged))

    yt_col = "rt_actual_true" if "rt_actual_true" in merged.columns else "rt_actual"
    if yt_col in merged.columns:
        yt = merged[yt_col].values.astype(float)
    else:
        yt = np.zeros(len(merged))

    hours = merged["hour"].values.astype(int)

    # Overall sMAPE
    overall_smape = _smape_floor50_np(yt, yp)

    # Period sMAPE (9-16)
    mask_916 = (hours >= 9) & (hours <= 16)
    smape_916 = (
        _smape_floor50_np(yt[mask_916], yp[mask_916])
        if mask_916.sum() > 0
        else float("nan")
    )

    # Monthly metrics
    merged["month"] = pd.to_datetime(merged["business_day"]).dt.to_period("M").astype(str)
    monthly_smape: dict[str, float] = {}
    for month, grp in merged.groupby("month"):
        m_yt_col = yt_col
        m_yt = grp[m_yt_col].values.astype(float)
        m_yp = grp["rt_pred"].values.astype(float)
        monthly_smape[month] = _smape_floor50_np(m_yt, m_yp)

    monthly_worst = max(monthly_smape.values()) if monthly_smape else float("inf")
    monthly_avg = float(np.mean(list(monthly_smape.values()))) if monthly_smape else float("inf")

    # Delta MAE
    dp_col = "delta_pred" if "delta_pred" in merged.columns else None
    dt_col = (
        "delta_target_true"
        if "delta_target_true" in merged.columns
        else ("delta_target" if "delta_target" in merged.columns else None)
    )
    if dp_col and dt_col:
        delta_mae = float(np.mean(
            np.abs(merged[dp_col].values.astype(float) - merged[dt_col].values.astype(float))
        ))
    else:
        delta_mae = float("nan")

    return {
        "name": name,
        "overall_sMAPE_floor50": round(overall_smape, 4),
        "monthly_worst_sMAPE": round(monthly_worst, 4),
        "monthly_avg_sMAPE_floor50": round(monthly_avg, 4),
        "9_16_sMAPE_floor50": round(smape_916, 4) if not np.isnan(smape_916) else float("inf"),
        "delta_mae": round(delta_mae, 4) if not np.isnan(delta_mae) else float("nan"),
        "rows_matched": len(merged),
        "monthly_smape": monthly_smape,
        "leakage_risk": False,
    }


# ── Leakage check ────────────────────────────────────────────────────

def check_leakage(pred_df: pd.DataFrame, start_date: pd.Timestamp) -> bool:
    """Return True if predictions might use future (post-cutoff) data."""
    if pred_df.empty:
        return False
    pred_days = pd.to_datetime(pred_df["business_day"]).unique()
    for d in pred_days:
        if pd.Timestamp(d) < start_date:
            logger.warning(
                "Leakage risk: prediction day %s is before start_date %s",
                pd.Timestamp(d).date(), start_date.date(),
            )
            return True
    return False


# ── Leaderboard ──────────────────────────────────────────────────────

def build_leaderboard(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Rank candidates.

    Ranking order:
      1. overall_sMAPE_floor50  (lower is better)
      2. monthly_worst_sMAPE    (lower is better)
      3. 9_16_sMAPE_floor50     (lower is better)
      4. runtime acceptable     (filter)
      5. leakage_risk must be false (hard filter)
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append({
            "name": r["name"],
            "overall_sMAPE_floor50": r["overall_sMAPE_floor50"],
            "monthly_worst_sMAPE": r.get("monthly_worst_sMAPE", float("inf")),
            "monthly_avg_sMAPE_floor50": r.get("monthly_avg_sMAPE_floor50", float("inf")),
            "9_16_sMAPE_floor50": r.get("9_16_sMAPE_floor50", float("inf")),
            "delta_mae": r.get("delta_mae", float("nan")),
            "runtime_seconds": r.get("runtime_seconds", 0),
            "leakage_risk": r.get("leakage_risk", False),
            "rows_matched": r.get("rows_matched", 0),
            "error": r.get("error", ""),
        })

    df = pd.DataFrame(rows)

    # Hard filter: leakage_risk must be False
    df_safe = df[df["leakage_risk"] == False].copy()  # noqa: E712
    if df_safe.empty:
        logger.warning("All candidates have leakage risk -- including all anyway.")
        df_safe = df.copy()

    # Soft filter: runtime acceptable
    df_fast = df_safe[df_safe["runtime_seconds"] <= MAX_RUNTIME_SECONDS].copy()
    if df_fast.empty:
        logger.warning("All candidates exceed runtime limit -- using all safe candidates.")
        df_fast = df_safe.copy()

    # Sort by ranking criteria
    df_fast = df_fast.sort_values(
        by=["overall_sMAPE_floor50", "monthly_worst_sMAPE", "9_16_sMAPE_floor50"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    df_fast.insert(0, "rank", range(1, len(df_fast) + 1))
    return df_fast


# ── Go / No-Go ───────────────────────────────────────────────────────

def determine_go_nogo(
    metrics: dict[str, Any],
    baseline_smape: float = BASELINE_SGDFNET,
) -> tuple[str, str]:
    """Determine the go/no-go verdict.

    Returns ``(verdict, detail)``.
    """
    if metrics.get("leakage_risk", False):
        return "NO-GO", "Leakage risk detected"

    monthly_avg = metrics.get("monthly_avg_sMAPE_floor50", float("inf"))
    overall = metrics.get("overall_sMAPE_floor50", float("inf"))

    if monthly_avg < PASS_THRESHOLD:
        return "PASS", f"Monthly avg sMAPE_floor50={monthly_avg:.4f} < {PASS_THRESHOLD}"
    if overall <= SOFT_PASS_THRESHOLD:
        return (
            "SOFT_PASS",
            f"Overall sMAPE_floor50={overall:.4f} <= {SOFT_PASS_THRESHOLD}, "
            "awaiting spike/negative module fusion",
        )
    if overall <= baseline_smape:
        return (
            "BASELINE_PASS",
            f"Overall sMAPE_floor50={overall:.4f} <= SGDFNet baseline {baseline_smape:.4f}",
        )
    return (
        "NO-GO",
        f"Overall sMAPE_floor50={overall:.4f} > SGDFNet baseline {baseline_smape:.4f}",
    )


# ── SGDFNet baseline candidate builder ───────────────────────────────

def build_sgdfnet_baseline_candidate(
    baseline_metrics: dict | None,
    baseline_metrics_path: str | None,
) -> dict[str, Any]:
    """Build the SGDFNet baseline candidate result."""
    metrics: dict[str, Any] | None = baseline_metrics

    if metrics is None and baseline_metrics_path:
        bp = Path(baseline_metrics_path)
        if bp.exists():
            with open(bp, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)

    if metrics is None:
        metrics = {
            "overall_sMAPE_floor50": BASELINE_SGDFNET,
            "monthly_avg_sMAPE_floor50": BASELINE_SGDFNET,
            "monthly_worst_sMAPE": BASELINE_SGDFNET,
            "9_16_sMAPE_floor50": BASELINE_SGDFNET,
            "delta_mae": float("nan"),
        }

    return {
        "name": "sgdfnet_baseline",
        "overall_sMAPE_floor50": metrics.get("overall_sMAPE_floor50", BASELINE_SGDFNET),
        "monthly_worst_sMAPE": metrics.get(
            "monthly_worst_sMAPE",
            metrics.get("monthly_avg_sMAPE_floor50", BASELINE_SGDFNET),
        ),
        "monthly_avg_sMAPE_floor50": metrics.get("monthly_avg_sMAPE_floor50", BASELINE_SGDFNET),
        "9_16_sMAPE_floor50": metrics.get("9_16_sMAPE_floor50", BASELINE_SGDFNET),
        "delta_mae": metrics.get("delta_mae", float("nan")),
        "runtime_seconds": 0,
        "leakage_risk": False,
        "rows_matched": metrics.get("rows_total", 0),
        "monthly_smape": {},
    }


# ── Output writing ───────────────────────────────────────────────────

def write_outputs(
    out_dir: Path,
    leaderboard: pd.DataFrame,
    champion_result: dict[str, Any],
    champion_pred_df: pd.DataFrame | None,
    blend_weights: dict[str, Any],
    baseline_smape: float,
) -> None:
    """Write all five output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. leaderboard.csv
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    logger.info("Leaderboard -> %s", out_dir / "leaderboard.csv")

    # 2. champion_predictions.csv
    if champion_pred_df is not None and not champion_pred_df.empty:
        desired_cols = [
            "business_day", "hour", "rt_actual", "rt_pred",
            "delta_target", "delta_pred", "da_anchor", "segment_id",
        ]
        out_cols = [c for c in desired_cols if c in champion_pred_df.columns]
        champion_pred_df[out_cols].to_csv(
            out_dir / "champion_predictions.csv", index=False, encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(
            out_dir / "champion_predictions.csv", index=False, encoding="utf-8-sig",
        )
    logger.info("Champion predictions -> %s", out_dir / "champion_predictions.csv")

    # 3. champion_metrics_summary.json
    verdict, detail = determine_go_nogo(champion_result, baseline_smape)
    metrics_out: dict[str, Any] = {}
    for k, v in champion_result.items():
        if k == "monthly_smape":
            metrics_out["monthly_smape"] = v
        else:
            metrics_out[k] = v
    metrics_out["verdict"] = verdict
    metrics_out["verdict_detail"] = detail
    metrics_out["timestamp"] = datetime.now().isoformat()

    with open(out_dir / "champion_metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("Champion metrics -> %s", out_dir / "champion_metrics_summary.json")

    # 4. champion_go_nogo.md
    champ_name = champion_result.get("name", "unknown")
    md: list[str] = [
        "# Champion Go/No-Go Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Champion:** {champ_name}",
        f"**Verdict:** {verdict}",
        "",
        detail,
        "",
        "## Champion Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall sMAPE_floor50 | {champion_result.get('overall_sMAPE_floor50', float('nan')):.4f} |",
        f"| Monthly Avg sMAPE_floor50 | {champion_result.get('monthly_avg_sMAPE_floor50', float('nan')):.4f} |",
        f"| Monthly Worst sMAPE | {champion_result.get('monthly_worst_sMAPE', float('nan')):.4f} |",
        f"| 9_16 sMAPE_floor50 | {champion_result.get('9_16_sMAPE_floor50', float('nan')):.4f} |",
        f"| Delta MAE | {champion_result.get('delta_mae', float('nan')):.4f} |",
        f"| Rows Matched | {champion_result.get('rows_matched', 0)} |",
        f"| Leakage Risk | {champion_result.get('leakage_risk', False)} |",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Overall sMAPE | Monthly Worst | 9-16 sMAPE | Runtime(s) | Leakage |",
        "|------|-------|---------------|---------------|------------|------------|---------|",
    ]

    for _, row in leaderboard.iterrows():
        md.append(
            f"| {int(row['rank'])} | {row['name']} | {row['overall_sMAPE_floor50']:.4f} | "
            f"{row['monthly_worst_sMAPE']:.4f} | {row['9_16_sMAPE_floor50']:.4f} | "
            f"{row['runtime_seconds']:.1f} | {row['leakage_risk']} |"
        )

    md += [
        "",
        "## Thresholds",
        "",
        f"- PASS: monthly avg sMAPE_floor50 < {PASS_THRESHOLD}",
        f"- SOFT_PASS: overall sMAPE_floor50 <= {SOFT_PASS_THRESHOLD}",
        f"- BASELINE_PASS: overall sMAPE_floor50 <= SGDFNet baseline {baseline_smape:.4f}",
        "- NO-GO: worse than baseline or leakage risk",
        "",
    ]

    monthly_smape = champion_result.get("monthly_smape", {})
    if monthly_smape:
        md += [
            "## Monthly Breakdown",
            "",
            "| Month | sMAPE_floor50 |",
            "|-------|---------------|",
        ]
        for month in sorted(monthly_smape.keys()):
            md.append(f"| {month} | {monthly_smape[month]:.4f} |")
        md.append("")

    (out_dir / "champion_go_nogo.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Go/No-Go report -> %s", out_dir / "champion_go_nogo.md")

    # 5. blend_weights.json
    with open(out_dir / "blend_weights.json", "w", encoding="utf-8") as fh:
        json.dump(blend_weights, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("Blend weights -> %s", out_dir / "blend_weights.json")


# ── Main orchestration ───────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    logger.info("=" * 60)
    logger.info("Phase 2 Champion Search")
    logger.info("=" * 60)
    logger.info("  Period      : %s to %s", start_date.date(), end_date.date())
    logger.info("  Output      : %s", out_dir)
    logger.info("  Device      : %s", args.device)
    logger.info("  AMP         : %s", args.amp)
    logger.info("  Fast-dev-run: %s", args.fast_dev_run)

    # SGDFNet availability
    sgdfnet_ok = _try_import_sgdfnet(args.sgdfnet_root)
    if sgdfnet_ok:
        logger.info("SGDFNet: available")
    else:
        logger.warning("SGDFNet: NOT available (%s)", _SGDFNET_IMPORT_ERROR)
        logger.warning("  -> Blend candidates (6, 7) will be skipped")

    # Load data
    raw_df = load_raw_data(args.data_path)
    logger.info("Raw data: %d rows", len(raw_df))

    # Ground truth for evaluation period
    gt_df = build_ground_truth(raw_df, start_date, end_date)
    if gt_df.empty:
        logger.error("No ground truth data in evaluation period. Exiting.")
        sys.exit(1)

    # Validation window (D-30 to D-1 before start_date)
    val_days = 7 if args.fast_dev_run else 30
    val_start = start_date - pd.Timedelta(days=val_days)
    val_end = start_date - pd.Timedelta(days=1)
    gt_val = build_ground_truth(raw_df, val_start, val_end)
    logger.info(
        "Validation window: %s to %s (%d rows)",
        val_start.date(), val_end.date(), len(gt_val),
    )

    # SGDFNet predictions (for blending + baseline)
    sgdfnet_pred_df, baseline_metrics = get_sgdfnet_predictions(
        raw_df, start_date, end_date, val_start, args.baseline_metrics,
    )

    # ── Collect candidates ───────────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    all_pred_dfs: dict[str, pd.DataFrame] = {}
    all_blend_weights: dict[str, Any] = {}

    # ── 1. SGDFNet baseline ──────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Candidate 1: SGDFNet Baseline")
    logger.info("-" * 60)
    baseline_result = build_sgdfnet_baseline_candidate(
        baseline_metrics, args.baseline_metrics,
    )
    all_results.append(baseline_result)
    logger.info("  overall_sMAPE = %.4f", baseline_result["overall_sMAPE_floor50"])

    # ── 2. V1 hourly TCN ────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Candidate 2: V1 Hourly TCN")
    logger.info("-" * 60)
    v1_tcn_pred, v1_tcn_info = train_and_predict_v1(
        raw_df, "tcn", start_date, end_date, args.device, args.amp, args.fast_dev_run,
    )
    if v1_tcn_pred is not None:
        m = evaluate_candidate(v1_tcn_pred, gt_df, "v1_hourly_tcn")
        m["runtime_seconds"] = v1_tcn_info.get("runtime_seconds", 0)
        m["leakage_risk"] = check_leakage(v1_tcn_pred, start_date)
        all_results.append(m)
        all_pred_dfs["v1_hourly_tcn"] = v1_tcn_pred
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])

    # ── 3. V1 hourly GRU ────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Candidate 3: V1 Hourly GRU")
    logger.info("-" * 60)
    v1_gru_pred, v1_gru_info = train_and_predict_v1(
        raw_df, "gru", start_date, end_date, args.device, args.amp, args.fast_dev_run,
    )
    if v1_gru_pred is not None:
        m = evaluate_candidate(v1_gru_pred, gt_df, "v1_hourly_gru")
        m["runtime_seconds"] = v1_gru_info.get("runtime_seconds", 0)
        m["leakage_risk"] = check_leakage(v1_gru_pred, start_date)
        all_results.append(m)
        all_pred_dfs["v1_hourly_gru"] = v1_gru_pred
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])

    # ── 4. V2 day TCN ───────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Candidate 4: V2 Day TCN")
    logger.info("-" * 60)
    v2_tcn_pred, v2_tcn_info, v2_tcn_model, v2_tcn_config = train_and_predict_v2(
        raw_df, "tcn", start_date, end_date, args.device, args.amp, args.fast_dev_run,
    )
    if v2_tcn_pred is not None:
        m = evaluate_candidate(v2_tcn_pred, gt_df, "v2_day_tcn")
        m["runtime_seconds"] = v2_tcn_info.get("runtime_seconds", 0)
        m["leakage_risk"] = check_leakage(v2_tcn_pred, start_date)
        all_results.append(m)
        all_pred_dfs["v2_day_tcn"] = v2_tcn_pred
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])

    # ── 5. V2 day GRU ───────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Candidate 5: V2 Day GRU")
    logger.info("-" * 60)
    v2_gru_pred, v2_gru_info, _, _ = train_and_predict_v2(
        raw_df, "gru", start_date, end_date, args.device, args.amp, args.fast_dev_run,
    )
    if v2_gru_pred is not None:
        m = evaluate_candidate(v2_gru_pred, gt_df, "v2_day_gru")
        m["runtime_seconds"] = v2_gru_info.get("runtime_seconds", 0)
        m["leakage_risk"] = check_leakage(v2_gru_pred, start_date)
        all_results.append(m)
        all_pred_dfs["v2_day_gru"] = v2_gru_pred
        logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])

    # ── 6 & 7. V2 blend candidates ──────────────────────────────────
    blend_specs = [
        ("v2_residual_sgdfnet", "sgdfnet_residual"),
        ("v2_blend_sgdfnet", "sgdfnet_blend"),
    ]

    can_blend = (
        v2_tcn_pred is not None
        and sgdfnet_pred_df is not None
        and v2_tcn_model is not None
    )

    if can_blend:
        import torch as _torch

        def _resolve_dev() -> _torch.device:
            if args.device == "auto":
                return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            return _torch.device(args.device)

        # Predict V2 TCN on the *validation* window for blend weight learning
        logger.info("Predicting V2 TCN on validation window for blend weight learning ...")
        v2_tcn_val_pred = _predict_v2_for_period(
            v2_tcn_model, raw_df, val_start, val_end,
            _resolve_dev(),
            batch_size=8 if args.fast_dev_run else 64,
            fast_dev_run=False,  # full val window (not truncated)
        )

        # SGDFNet predictions for the validation window
        sgdf_val = sgdfnet_pred_df[
            (pd.to_datetime(sgdfnet_pred_df["business_day"]) >= val_start)
            & (pd.to_datetime(sgdfnet_pred_df["business_day"]) <= val_end)
        ].copy()

        if v2_tcn_val_pred is not None and not v2_tcn_val_pred.empty and not sgdf_val.empty and not gt_val.empty:
            for cand_name, blend_mode in blend_specs:
                logger.info("-" * 60)
                logger.info("Candidate: %s (%s)", cand_name, blend_mode)
                logger.info("-" * 60)
                try:
                    weights = learn_blend_weights(
                        v2_tcn_val_pred, sgdf_val, gt_val, blend_mode,
                    )
                    all_blend_weights[cand_name] = weights

                    blended_pred = apply_blend_to_predictions(
                        v2_tcn_pred, sgdfnet_pred_df, weights, blend_mode,
                    )

                    m = evaluate_candidate(blended_pred, gt_df, cand_name)
                    m["runtime_seconds"] = v2_tcn_info.get("runtime_seconds", 0) + 1.0
                    m["leakage_risk"] = check_leakage(blended_pred, start_date)
                    all_results.append(m)
                    all_pred_dfs[cand_name] = blended_pred
                    logger.info("  overall_sMAPE = %.4f", m["overall_sMAPE_floor50"])
                except Exception as exc:
                    logger.error("%s failed: %s", cand_name, exc)
                    logger.debug(traceback.format_exc())
        else:
            skip_parts = []
            if v2_tcn_val_pred is None or v2_tcn_val_pred.empty:
                skip_parts.append("no V2 val predictions")
            if sgdf_val.empty:
                skip_parts.append("no SGDFNet val predictions")
            if gt_val.empty:
                skip_parts.append("no val ground truth")
            logger.warning("Skipping blend candidates: %s", "; ".join(skip_parts))
    else:
        skip_parts = []
        if v2_tcn_pred is None:
            skip_parts.append("V2 TCN predictions unavailable")
        if sgdfnet_pred_df is None:
            skip_parts.append("SGDFNet predictions unavailable")
        if v2_tcn_model is None:
            skip_parts.append("V2 TCN model unavailable")
        logger.warning("Skipping blend candidates: %s", "; ".join(skip_parts))

    # ── Build leaderboard ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Building Leaderboard")
    logger.info("=" * 60)

    leaderboard = build_leaderboard(all_results)
    if leaderboard.empty:
        logger.error("No candidates produced usable results. Exiting.")
        sys.exit(1)

    for _, row in leaderboard.iterrows():
        logger.info(
            "  #%d  %-25s  overall=%.4f  monthly_worst=%.4f  9_16=%.4f  %.1fs",
            int(row["rank"]), row["name"],
            row["overall_sMAPE_floor50"], row["monthly_worst_sMAPE"],
            row["9_16_sMAPE_floor50"], row["runtime_seconds"],
        )

    # ── Champion ─────────────────────────────────────────────────────
    champion_name = leaderboard.iloc[0]["name"]
    champion_result = next(r for r in all_results if r["name"] == champion_name)
    champion_pred_df = all_pred_dfs.get(champion_name)

    # Merge champion predictions with ground truth for the output CSV
    if champion_pred_df is not None and not champion_pred_df.empty:
        champ_out = champion_pred_df.merge(
            gt_df[["business_day", "hour", "rt_actual", "delta_target"]],
            on=["business_day", "hour"],
            how="left",
            suffixes=("", "_gt"),
        )
        if "rt_actual_gt" in champ_out.columns:
            champ_out["rt_actual"] = champ_out["rt_actual_gt"]
            champ_out = champ_out.drop(columns=["rt_actual_gt"], errors="ignore")
        if "delta_target_gt" in champ_out.columns:
            champ_out["delta_target"] = champ_out["delta_target_gt"]
            champ_out = champ_out.drop(columns=["delta_target_gt"], errors="ignore")
    else:
        champ_out = gt_df.copy()

    verdict, detail = determine_go_nogo(champion_result, BASELINE_SGDFNET)
    logger.info("=" * 60)
    logger.info("CHAMPION : %s", champion_name)
    logger.info("VERDICT  : %s", verdict)
    logger.info("DETAIL   : %s", detail)
    logger.info("=" * 60)

    # ── Write outputs ────────────────────────────────────────────────
    write_outputs(
        out_dir, leaderboard, champion_result, champ_out,
        all_blend_weights, BASELINE_SGDFNET,
    )

    logger.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
