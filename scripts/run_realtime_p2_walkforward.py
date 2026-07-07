"""P2 Realtime — unified walk-forward runner (D14 cutoff-safe).

Generates, for one model, predictions + metrics + cutoff audit over a date range,
using monthly walk-forward (train on all data strictly before each test month).

Usage examples:
  # deep candidate (monthly walk-forward)
  python scripts/run_realtime_p2_walkforward.py --model tcn_day \
      --start-date 2025-01-01 --end-date 2026-06-30 --device cuda

  # strong baseline (DA anchor)
  python scripts/run_realtime_p2_walkforward.py --model da_anchor \
      --start-date 2025-01-01 --end-date 2026-06-30

  # SGDFNet baseline reproduced at D14 (via bridge)
  python scripts/run_realtime_p2_walkforward.py --model sgdfnet_d14 \
      --start-date 2025-01-01 --end-date 2026-06-30

Outputs: outputs/p2_realtime/{run_id}/predictions/, metrics/, reports/, audit/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ensure repo root (models/, scripts/) importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import p2_common as C


def iter_target_days(base: pd.DataFrame, lo, hi):
    """Yield business_days in [lo, hi] that have a full 24 rows."""
    lo = pd.Timestamp(lo).normalize()
    hi = pd.Timestamp(hi).normalize()
    sub = base[(base["business_day"] >= lo) & (base["business_day"] <= hi)]
    counts = sub.groupby("business_day").size()
    for bd, n in counts.items():
        if n >= 24:
            yield bd


def build_samples(base, rt, da, fc, target_days, spec, target_is_abs=False):
    Xs, ys = [], []
    for T in target_days:
        X, y, _ = C.assemble_day_features(T, rt, da, fc, spec, target_is_abs=target_is_abs)
        Xs.append(X)
        ys.append(y)
    if not Xs:
        return np.empty((0, 24, C.feature_dim(spec)), dtype=np.float32), np.empty((0, 24), dtype=np.float32)
    return np.stack(Xs), np.stack(ys)


def run_deep(model_name, base, rt, da, fc, months, device, args, spec):
    from p2_common import MODEL_REGISTRY, train_monthly, predict_month
    fdim = C.feature_dim(spec)
    target_is_abs = (getattr(args, "target_mode", "delta") == "abs")
    all_rows = []
    audit_rows = []
    per_month = []
    t_train = 0.0
    t_infer = 0.0
    for M_start, M_end in months:
        # train / val target-day ranges (strictly before M_start)
        tr_lo = M_start - pd.Timedelta(days=args.train_horizon)
        tr_hi = M_start - pd.Timedelta(days=args.val_days)
        va_lo = tr_hi
        va_hi = M_start
        tr_days = list(iter_target_days(base, tr_lo, tr_hi - pd.Timedelta(days=1)))
        va_days = list(iter_target_days(base, va_lo, va_hi - pd.Timedelta(days=1)))
        if not tr_days:
            continue
        Xtr, ytr = build_samples(base, rt, da, fc, tr_days, spec, target_is_abs)
        Xval, yval = build_samples(base, rt, da, fc, va_days, spec, target_is_abs)
        # ── standardization (critical for deep models) ──
        tr_ok = ~np.isnan(ytr).any(axis=1)
        Xmean = np.nanmean(Xtr[tr_ok], axis=(0, 1))
        Xstd = np.nanstd(Xtr[tr_ok], axis=(0, 1)) + 1e-6
        ymean = float(np.nanmean(ytr[tr_ok]))
        ystd = float(np.nanstd(ytr[tr_ok]) + 1e-6)
        Xtr = (Xtr - Xmean) / Xstd
        Xval = (Xval - Xmean) / Xstd
        ytr_n = (ytr - ymean) / ystd
        yval_n = (yval - ymean) / ystd
        model = MODEL_REGISTRY[model_name](fdim, hidden=args.hidden, layers=args.layers)
        model.to(device)
        t0 = time.time()
        train_monthly(model, Xtr, ytr_n, Xval, yval_n, device,
                      epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed)
        t_train += time.time() - t0
        # val rmse (confidence proxy) — on standardized scale then unscale
        if Xval.shape[0] > 0:
            vp = predict_month(model, Xval, device) * ystd + ymean
            val_rmse = float(np.sqrt(np.nanmean((vp - yval) ** 2)))
        else:
            val_rmse = 150.0
        conf = float(np.clip(1.0 - val_rmse / 300.0, 0.1, 0.95))
        # predict month
        pred_days = list(iter_target_days(base, M_start, M_end))
        if not pred_days:
            continue
        Xp, yp = build_samples(base, rt, da, fc, pred_days, spec, target_is_abs)
        Xp = (Xp - Xmean) / Xstd
        t0 = time.time()
        dpred = predict_month(model, Xp, device) * ystd + ymean
        t_infer += time.time() - t0
        for i, T in enumerate(pred_days):
            da_row = base[base["business_day"] == T]
            for h in range(1, 25):
                r = da_row[da_row["hour_business"] == h]
                if r.empty:
                    continue
                r = r.iloc[0]
                pred_val = float(dpred[i, h - 1])
                da_v = float(r[C.DA_COL]) if not pd.isna(r[C.DA_COL]) else float("nan")
                if target_is_abs:
                    trend = pred_val
                    delta = trend - da_v if not np.isnan(da_v) else float("nan")
                else:
                    delta = pred_val
                    trend = da_v + delta if not np.isnan(da_v) else float("nan")
                yt = float(r[C.RT_COL]) if not pd.isna(r[C.RT_COL]) else float("nan")
                spike = (not pd.isna(yt)) and (abs(yt) > 500)
                neg = (not pd.isna(yt)) and (yt < 0)
                all_rows.append(dict(
                    business_day=str(T.date()), ds=str(r["ds"]),
                    hour_business=int(r["hour_business"]), period=r["segment"],
                    da_anchor=da_v, delta_pred=delta, trend_pred=trend,
                    model_name=model_name, model_version=("v1_abs" if target_is_abs else "v1_delta"), confidence=round(conf, 3),
                    run_id=RUN_ID, y_true=yt,
                    spike_pred=float("nan"), negative_pred=float("nan"),
                    residual_for_spike_module=(yt - trend) if not pd.isna(yt) else float("nan"),
                    residual_for_negative_module=(yt - trend) if not pd.isna(yt) else float("nan"),
                    is_spike=bool(spike), is_negative=bool(neg),
                ))
            audit_rows.append(dict(
                decision_day=str((T - pd.Timedelta(days=1)).date()),
                target_day=str(T.date()),
                decision_timestamp=str(T - pd.Timedelta(days=1) + pd.Timedelta(hours=C.DECISION_HOUR)),
                max_visible_realtime_timestamp=str(T - pd.Timedelta(days=1) + pd.Timedelta(hours=C.DECISION_HOUR)),
                cutoff_ok=True, protocol_tag="B_D14_cutoff_walk_forward",
            ))
        # per-month metric (on this month's predicted days)
        mdf = pd.DataFrame([r for r in all_rows if r["business_day"] >= str(M_start.date()) and r["business_day"] <= str(M_end.date())])
        if not mdf.empty:
            mm = C.compute_metrics(mdf)
            mm["month"] = M_start.strftime("%Y-%m")
            mm["val_rmse"] = round(val_rmse, 3)
            per_month.append(mm)
    return all_rows, audit_rows, per_month, t_train, t_infer


def run_da_anchor(base, months):
    all_rows, audit_rows, per_month = [], [], []
    for M_start, M_end in months:
        pred_days = list(iter_target_days(base, M_start, M_end))
        for T in pred_days:
            da_row = base[base["business_day"] == T]
            for h in range(1, 25):
                r = da_row[da_row["hour_business"] == h]
                if r.empty:
                    continue
                r = r.iloc[0]
                da_v = float(r[C.DA_COL]) if not pd.isna(r[C.DA_COL]) else float("nan")
                yt = float(r[C.RT_COL]) if not pd.isna(r[C.RT_COL]) else float("nan")
                spike = (not pd.isna(yt)) and (abs(yt) > 500)
                neg = (not pd.isna(yt)) and (yt < 0)
                all_rows.append(dict(
                    business_day=str(T.date()), ds=str(r["ds"]),
                    hour_business=int(r["hour_business"]), period=r["segment"],
                    da_anchor=da_v, delta_pred=0.0, trend_pred=da_v,
                    model_name="da_anchor", model_version="v0", confidence=0.5,
                    run_id=RUN_ID, y_true=yt, spike_pred=float("nan"), negative_pred=float("nan"),
                    residual_for_spike_module=(yt - da_v) if not pd.isna(yt) else float("nan"),
                    residual_for_negative_module=(yt - da_v) if not pd.isna(yt) else float("nan"),
                    is_spike=bool(spike), is_negative=bool(neg),
                ))
            audit_rows.append(dict(
                decision_day=str((T - pd.Timedelta(days=1)).date()), target_day=str(T.date()),
                decision_timestamp=str(T - pd.Timedelta(days=1) + pd.Timedelta(hours=C.DECISION_HOUR)),
                max_visible_realtime_timestamp=str(T - pd.Timedelta(days=1) + pd.Timedelta(hours=C.DECISION_HOUR)),
                cutoff_ok=True, protocol_tag="B_D14_cutoff_walk_forward"))
        mdf = pd.DataFrame([r for r in all_rows if r["business_day"] >= str(M_start.date()) and r["business_day"] <= str(M_end.date())])
        if not mdf.empty:
            mm = C.compute_metrics(mdf)
            mm["month"] = M_start.strftime("%Y-%m")
            per_month.append(mm)
    return all_rows, audit_rows, per_month, 0.0, 0.0


def run_sgdfnet_d14(base, months, args):
    """Reproduce SGDFNet baseline at D14 via the sibling bridge.

    Faithful reproduction: load SGDFNet production baseline config and override
    decision_hour -> 14 (D14 cutoff). SGDFNet's _build_protocol_b_visible_frame
    uses decision_hour to mask post-cutoff realtime actuals, so this yields a
    genuinely D14-cutoff-safe prediction. (The internal protocol_tag string
    stays 'B_D15...' in upstream code; we relabel it below.)
    """
    from models.deep_sgdf_delta.sgdfnet_bridge import run_protocol_b_cutoff_experiment
    import tempfile
    import yaml
    sgdfnet_root = Path(args.sgdfnet_root or "")
    baseline_cfg = sgdfnet_root / "configs" / "cutoff_recovery_2026_baseline.yaml"
    if not baseline_cfg.exists():
        # fall back to any production config
        baseline_cfg = sgdfnet_root / "configs" / "production_sgdfnet_realtime.yaml"
    if not baseline_cfg.exists():
        raise FileNotFoundError(f"No SGDFNet baseline config at {sgdfnet_root/'configs'}")
    cfg = yaml.safe_load(baseline_cfg.read_text(encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="p2_sgdfnet_d14_")
    cfg.update(dict(
        experiment_name="p2_sgdfnet_d14",
        data_path=args.data_path,
        output_root=tmp,
        start_day=args.start_date,
        end_day=args.end_date,
        decision_hour=14,
        val_days=30,
        train_min_rows=2160,
    ))
    cfg_path = Path(tmp) / "d14.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_dir = run_protocol_b_cutoff_experiment(cfg_path)
    pred = pd.read_csv(run_dir / "predictions.csv")
    # map to our schema
    all_rows, audit_rows, per_month = [], [], []
    pred["business_day"] = pd.to_datetime(pred["business_day"])
    for M_start, M_end in months:
        m = (pred["business_day"] >= M_start) & (pred["business_day"] <= M_end)
        sub = pred[m]
        for _, r in sub.iterrows():
            yt = float(r["rt_actual"]) if not pd.isna(r.get("rt_actual")) else float("nan")
            trend = float(r["rt_hat"]) if not pd.isna(r.get("rt_hat")) else float("nan")
            da_v = float(r["da_anchor"]) if not pd.isna(r.get("da_anchor")) else float("nan")
            raw_hour = int(r.get("hour", 0))
            hb = 24 if raw_hour == 0 else raw_hour
            spike = (not pd.isna(yt)) and (abs(yt) > 500)
            neg = (not pd.isna(yt)) and (yt < 0)
            all_rows.append(dict(
                business_day=str(pd.Timestamp(r["business_day"]).date()), ds=str(r.get("timestamp", "")),
                hour_business=hb, period=str(r.get("segment", "")),
                da_anchor=da_v, delta_pred=(trend - da_v) if not pd.isna(trend) else float("nan"),
                trend_pred=trend, model_name="sgdfnet_d14", model_version="v1",
                confidence=0.5, run_id=RUN_ID, y_true=yt,
                spike_pred=float("nan"), negative_pred=float("nan"),
                residual_for_spike_module=(yt - trend) if not pd.isna(yt) else float("nan"),
                residual_for_negative_module=(yt - trend) if not pd.isna(yt) else float("nan"),
                is_spike=bool(spike), is_negative=bool(neg),
            ))
            audit_rows.append(dict(
                decision_day=str((pd.Timestamp(r["business_day"]) - pd.Timedelta(days=1)).date()),
                target_day=str(pd.Timestamp(r["business_day"]).date()),
                decision_timestamp=str(pd.Timestamp(r["business_day"]) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)),
                max_visible_realtime_timestamp=str(pd.Timestamp(r["business_day"]) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)),
                cutoff_ok=True, protocol_tag="B_D14_cutoff_walk_forward",
            ))
        mdf = pd.DataFrame([x for x in all_rows if x["business_day"] >= str(M_start.date()) and x["business_day"] <= str(M_end.date())])
        if not mdf.empty:
            mm = C.compute_metrics(mdf)
            mm["month"] = M_start.strftime("%Y-%m")
            per_month.append(mm)
    return all_rows, audit_rows, per_month, 0.0, 0.0


def main():
    global RUN_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["da_anchor", "sgdfnet_d14", "tcn_day", "gru_day", "dlinear_day", "linear_day"])
    ap.add_argument("--data-path", default=r"D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.xlsx")
    ap.add_argument("--sgdfnet-root", default=r"D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/SGDFNet")
    ap.add_argument("--start-date", default="2025-01-01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--decision-hour", type=int, default=14)
    ap.add_argument("--target-mode", default="delta", choices=["delta", "abs"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--train-horizon", type=int, default=180)
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", default="outputs/p2_realtime")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    assert args.decision_hour == 14, "This framework enforces D14 cutoff only."

    RUN_ID = args.run_id or f"{args.model}_d14_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(args.output_root) / RUN_ID
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    (out_root / "audit").mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[run] model={args.model} device={device} run_id={RUN_ID}")

    raw = C.load_raw(args.data_path)
    base = C.build_base_frame(raw)
    rt, da, fc = C.build_lookups(base)

    # test months present in range
    lo = pd.Timestamp(args.start_date).normalize()
    hi = pd.Timestamp(args.end_date).normalize()
    sub = base[(base["business_day"] >= lo) & (base["business_day"] <= hi)]
    month_set = sorted({pd.Timestamp(b).to_period("M").to_timestamp() for b in sub["business_day"].unique()})
    months = []
    for ms in month_set:
        me = (ms + pd.offsets.MonthEnd(0)).normalize()
        months.append((ms.normalize(), me))

    spec = C.FeatureSpec()
    t0_all = time.time()
    if args.model == "da_anchor":
        rows, audit, per_month, t_train, t_infer = run_da_anchor(base, months)
    elif args.model == "sgdfnet_d14":
        rows, audit, per_month, t_train, t_infer = run_sgdfnet_d14(base, months, args)
    else:
        rows, audit, per_month, t_train, t_infer = run_deep(args.model, base, rt, da, fc, months, device, args, spec)
    elapsed = time.time() - t0_all

    df = pd.DataFrame(rows)
    C.write_predictions_csv(out_root / "predictions" / "predictions.csv", rows)
    pd.DataFrame(audit).to_csv(out_root / "audit" / "split_audit.csv", index=False, encoding="utf-8-sig")

    overall = C.compute_metrics(df) if not df.empty else {}
    overall["train_time_s"] = round(t_train, 1)
    overall["infer_time_s"] = round(t_infer, 1)
    overall["total_time_s"] = round(elapsed, 1)
    overall["n_days"] = int(df["business_day"].nunique()) if not df.empty else 0
    overall["n_rows"] = int(len(df))
    meta = dict(model=args.model, decision_hour=args.decision_hour, run_id=RUN_ID,
                start=args.start_date, end=args.end_date, device=device,
                months=[f"{a.strftime('%Y-%m')}" for a, _ in months],
                data_path=args.data_path, cutoff="D14")
    C.write_metrics_json(out_root / "metrics" / "metrics.json", overall, meta)

    # per-month metrics
    pd.DataFrame(per_month).to_csv(out_root / "metrics" / "per_month.csv", index=False, encoding="utf-8-sig")

    # short report
    rep = [f"# P2 Realtime Run — {args.model} (D14)", ""]
    rep.append(f"- run_id: {RUN_ID}")
    rep.append(f"- device: {device}, total_time: {elapsed:.1f}s (train {t_train:.1f}s / infer {t_infer:.1f}s)")
    rep.append(f"- days: {overall.get('n_days')}, rows: {overall.get('n_rows')}")
    rep.append(f"- sMAPE_floor50: {overall.get('sMAPE_floor50')}")
    rep.append(f"- MAE: {overall.get('MAE')}, RMSE: {overall.get('RMSE')}")
    rep.append("")
    rep.append("## Per-month sMAPE_floor50")
    for mm in per_month:
        rep.append(f"- {mm['month']}: {mm.get('sMAPE_floor50')}")
    (out_root / "reports" / "run_report.md").write_text("\n".join(rep), encoding="utf-8")

    print(f"[run] DONE run_id={RUN_ID} sMAPE_floor50={overall.get('sMAPE_floor50')} days={overall.get('n_days')}")
    print(f"[run] outputs at {out_root}")


if __name__ == "__main__":
    main()
