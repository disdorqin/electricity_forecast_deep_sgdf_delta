"""P2.2 ledger population.

After the parallel backfill produces per-day realtime model CSVs, populate the
realtime prediction + actual ledger so the faithful 2.5 GEF fusion chain can use
the trailing-window weight learner. The existing 32 winter (2026-01/02) ledger
rows remain and act as history for the 2026 spring/summer months.

Idempotent: append_predictions_to_ledger / update_actual_ledger dedup by key.

Usage:
  python p2_2_populate_ledger.py
"""
from __future__ import annotations
import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

EFM3 = "D:/作业_大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0".replace("_大创", "大创")
# Fix: explicit correct path
EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
if EFM3 not in sys.path:
    sys.path.insert(0, EFM3)

import pandas as pd
from pipelines.prediction_ledger import (
    append_predictions_to_ledger,
    update_actual_ledger,
)

REALTIME_MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
DATA_XLSX = str(Path(EFM3) / "data" / "shandong_pmos_hourly.xlsx")
LEDGER_ROOT = Path(EFM3) / "outputs" / "ledger"
RUNS_ROOT = Path(EFM3) / "outputs" / "runs"
# Test-window months that get SCORED by the analysis (history months below are
# backfilled only to provide trailing realtime history for GEF weight learning).
SCORED_MONTHS = ["2025-03", "2025-04", "2025-05", "2025-06", "2025-09", "2025-10",
                 "2026-03", "2026-04", "2026-05", "2026-06"]


def discover_days():
    """Auto-discover every day under outputs/runs/<date>/realtime/prediction
    that has all 4 model CSVs. Returns dict month -> sorted list of YYYY-MM-DD."""
    out = {}
    if not RUNS_ROOT.exists():
        return out
    for d in sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir() and len(p.name) == 10):
        pred_dir = p / "realtime" / "prediction"
        if not pred_dir.exists():
            continue
        have = all((pred_dir / f"{m}_predictions.csv").exists() for m in REALTIME_MODELS)
        if have:
            out.setdefault(d.name[:7], []).append(d.name)
    for k in out:
        out[k] = sorted(out[k])
    return out


def month_days(month):
    y, m = month.split("-")
    d0 = pd.Timestamp(year=int(y), month=int(m), day=1)
    d1 = d0 + pd.offsets.MonthEnd(1)
    return pd.date_range(d0, d1, freq="D").strftime("%Y-%m-%d").tolist()


def load_raw():
    return pd.read_excel(DATA_XLSX)


def extract_actuals(raw, target_date):
    ts_col = None
    for c in ["时刻", "ds", "timestamp", "time", "datetime"]:
        if c in raw.columns:
            ts_col = c
            break
    if ts_col is None:
        return {}
    raw = raw.copy()
    raw["ds"] = pd.to_datetime(raw[ts_col], errors="coerce")
    td = pd.Timestamp(target_date)
    start_ts = td.replace(hour=1, minute=0, second=0)
    end_ts = (td + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0)
    day = raw[(raw["ds"] >= start_ts) & (raw["ds"] <= end_ts)].copy()
    if len(day) == 0:
        return {}
    from utils.business_day import business_day_from_timestamp, hour_business_from_timestamp, infer_period
    day["business_day"] = day["ds"].apply(business_day_from_timestamp)
    day["hour_business"] = day["ds"].apply(hour_business_from_timestamp)
    day["period"] = day["hour_business"].apply(infer_period)
    out = {}
    for task, cols in [("dayahead", ["日前电价", "日前出清电价", "day_ahead_clearing_price"]),
                       ("realtime", ["实时电价", "realtime_price", "rt_price"])]:
        yc = next((c for c in cols if c in day.columns), None)
        if yc is None:
            continue
        sub = day[["ds", "business_day", "hour_business", "period", yc]].copy()
        sub["y_true"] = pd.to_numeric(day[yc], errors="coerce")
        sub["task"] = task
        sub["target_day"] = target_date
        sub = sub.dropna(subset=["y_true"])
        if len(sub):
            out[task] = sub
    return out


def main():
    t0 = time.time()
    raw = load_raw()
    days_by_month = discover_days()
    if not days_by_month:
        print("[pop] no backfilled days found under outputs/runs", flush=True)
        return
    all_days = [d for days in days_by_month.values() for d in days]
    print(f"[pop] discovered {len(all_days)} days across {len(days_by_month)} months "
          f"(scored={SCORED_MONTHS})", flush=True)
    n_pred = 0
    n_act = 0
    for mm, days in sorted(days_by_month.items()):
        for d in days:
            pred_dir = Path(EFM3) / "outputs" / "runs" / d / "realtime" / "prediction"
            pieces = []
            have_all = True
            for m in REALTIME_MODELS:
                p = pred_dir / f"{m}_predictions.csv"
                if not p.exists():
                    have_all = False
                    break
                pieces.append(pd.read_csv(p))
            if not have_all or not pieces:
                print(f"[pop] skip {d}: missing model CSVs", flush=True)
                continue
            long = pd.concat(pieces, ignore_index=True)
            res = append_predictions_to_ledger(long, LEDGER_ROOT, "realtime",
                                               source_file=str(pred_dir))
            n_pred += 1
            acts = extract_actuals(raw, d)
            for task, sub in acts.items():
                update_actual_ledger(sub, LEDGER_ROOT, task, source_file=DATA_XLSX)
                n_act += 1
    print(f"[pop] DONE in {round((time.time()-t0)/60,1)} min; "
          f"days_with_pred={n_pred}, actual_updates={n_act}", flush=True)


if __name__ == "__main__":
    main()
