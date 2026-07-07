"""P2.2 research backfill worker — processes ONE model over a set of months.

Bypasses main.py (torch segfault import chain) and ResourceScheduler (nested
ProcessPoolExecutor crash) by calling pipelines.ledger_predict._predict_model
directly in the main thread, one day at a time.

Resumable: skips any (model,day) CSV that already exists unless --force.
Robust: per-day errors are caught and recorded; the worker continues.

Output: writes outputs/runs/p2_2_worker_{model}_{chunkid}.json summary.

Usage:
  python p2_2_worker.py --model rt916 --months 2025-03,2025-04 --chunkid c1
  python p2_2_worker.py --model timemixer --months 2025-05 --chunkid t2 --force
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --- env fixes MUST be set before importing any torch-using module ---
os.environ.setdefault("OPTIM_NUM_WORKERS", "0")
os.environ.setdefault("OPTIM_PIN_MEMORY", "0")

# Pre-load torch in the MAIN thread BEFORE importing efm3.0 modules
import torch  # noqa: F401
print(f"[worker:{os.getpid()}] torch {torch.__version__} preloaded (cuda={torch.cuda.is_available()})", flush=True)

EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
if EFM3 not in sys.path:
    sys.path.insert(0, EFM3)
os.environ.setdefault("PROJECT_ROOT", EFM3)  # timesfm loader finds models/timesFM via this

import pandas as pd
from pipelines.ledger_predict import _predict_model

DATA_XLSX = str(Path(EFM3) / "data" / "shandong_pmos_hourly.xlsx")
RT_CUTOFF_HOUR = 14


def month_days(month: str):
    y, m = month.strip().split("-")
    d0 = pd.Timestamp(year=int(y), month=int(m), day=1)
    d1 = d0 + pd.offsets.MonthEnd(1)
    return pd.date_range(d0, d1, freq="D").strftime("%Y-%m-%d").tolist()


def run(model, months, chunkid, force):
    dates = []
    for mm in months:
        dates += month_days(mm)
    dates = sorted(set(dates))
    pred_dir_for = lambda d: Path(EFM3) / "outputs" / "runs" / d / "realtime" / "prediction"
    summary = {"model": model, "chunkid": chunkid, "months": months,
               "n_days": len(dates), "started_at": datetime.now(timezone.utc).isoformat(),
               "results": {}, "errors": []}
    out_path = Path(EFM3) / "outputs" / "runs" / f"p2_2_worker_{model}_{chunkid}.json"
    t_start = time.time()
    ok = 0
    for i, d in enumerate(dates, 1):
        out = pred_dir_for(d) / f"{model}_predictions.csv"
        if out.exists() and not force:
            summary["results"][d] = "cached"
            ok += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)  # _predict_model does NOT create dirs
        pd_cutoff = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        rt_cutoff = f"{pd_cutoff} {RT_CUTOFF_HOUR:02d}:00:00"
        t0 = time.time()
        try:
            _predict_model(
                model_name=model, task="realtime", target_date=d,
                data_path=DATA_XLSX, epf_root=None, allow_v2_fallback=False,
                epf_v1_mode="exact", cutoff_date=rt_cutoff,
                realtime_cutoff_hour=RT_CUTOFF_HOUR, training_months=12,
                val_ratio=0.2, timemixer_epochs=80, timemixer_patience=15,
                timemixer_batch_size=16, timemixer_full_refit=True,
                timemixer_seeds=42, seed=42, deterministic=False,
                output_path=str(out),
            )
            dt = time.time() - t0
            summary["results"][d] = f"ok({dt:.0f}s)"
            ok += 1
        except Exception as e:
            summary["results"][d] = f"ERROR: {type(e).__name__}: {e}"
            summary["errors"].append(f"{d}: {type(e).__name__}: {e}")
            traceback.print_exc()
        # periodic flush so a crash doesn't lose the whole summary
        if i % 5 == 0 or i == len(dates):
            summary["elapsed_min"] = round((time.time() - t_start) / 60, 1)
            summary["done_so_far"] = i
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"[worker {model}/{chunkid}] {i}/{len(dates)} {d}: {summary['results'][d]} (ok={ok})", flush=True)
    summary["elapsed_min"] = round((time.time() - t_start) / 60, 1)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    summary["ok_count"] = ok
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[worker {model}/{chunkid}] DONE ok={ok}/{len(dates)} in {summary['elapsed_min']} min -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--months", required=True, help="comma YYYY-MM")
    ap.add_argument("--chunkid", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    months = [m.strip() for m in a.months.split(",") if m.strip()]
    run(a.model, months, a.chunkid, a.force)


if __name__ == "__main__":
    main()
