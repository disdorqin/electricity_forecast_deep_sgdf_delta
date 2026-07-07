"""P2.2 research backfill driver — realtime-only, 4 production models.

Bypasses main.py (avoids torch segfault import chain) and the ResourceScheduler
(nested ProcessPoolExecutor + DataLoader subprocess crash on Windows) by calling
pipelines.ledger_predict._predict_model directly in the main thread.

Why realtime-only: P2.2's "DA anchor" is the raw day-ahead price (日前电价) from
the data file, NOT a model prediction. So we only need the 4 realtime model
predictions (timesfm / sgdfnet / timemixer / rt916) + actuals (read from raw data
in the analysis script). Day-ahead model backfill is unnecessary.

Resumable: skips any model CSV that already exists (unless --force).

Usage:
  python scripts/p2_2_backfill.py --start 2025-03-01 --end 2025-03-31
  python scripts/p2_2_backfill.py --months 2025-03,2025-04,2025-05
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

# --- 0) env fixes MUST be set before importing any torch-using module ---
os.environ.setdefault("OPTIM_NUM_WORKERS", "0")
os.environ.setdefault("OPTIM_PIN_MEMORY", "0")

# --- 1) pre-load torch in the MAIN thread BEFORE importing efm3.0 modules ---
import torch  # noqa: F401  (preload; avoids flaky threaded import segfault)
print(f"[backfill] torch {torch.__version__} preloaded (cuda={torch.cuda.is_available()})", flush=True)

EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
if EFM3 not in sys.path:
    sys.path.insert(0, EFM3)

import pandas as pd
from pipelines.ledger_predict import _predict_model

MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
DATA_XLSX = str(Path(EFM3) / "data" / "shandong_pmos_hourly.xlsx")


def run_day(target_date: str, models: list[str], rt_cutoff_hour: int = 14, force: bool = False):
    da_cutoff = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rt_cutoff = f"{da_cutoff} {rt_cutoff_hour:02d}:00:00"
    pred_dir = Path(EFM3) / "outputs" / "runs" / target_date / "realtime" / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model in models:
        out = pred_dir / f"{model}_predictions.csv"
        if out.exists() and not force:
            results[model] = "cached"
            continue
        t0 = time.time()
        try:
            _predict_model(
                model_name=model,
                task="realtime",
                target_date=target_date,
                data_path=DATA_XLSX,
                epf_root=None,
                allow_v2_fallback=False,
                epf_v1_mode="exact",
                cutoff_date=rt_cutoff,
                realtime_cutoff_hour=rt_cutoff_hour,
                training_months=12,
                val_ratio=0.2,
                timemixer_epochs=80,
                timemixer_patience=15,
                timemixer_batch_size=16,
                timemixer_full_refit=True,
                timemixer_seeds=42,
                seed=42,
                deterministic=False,
                output_path=str(out),
            )
            dt = time.time() - t0
            results[model] = f"ok({dt:.0f}s)"
        except Exception as e:
            results[model] = f"ERROR: {type(e).__name__}: {e}"
            traceback.print_exc()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--months", default=None, help="comma list of YYYY-MM")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--cutoff", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]

    dates = []
    if a.months:
        for mm in a.months.split(","):
            mm = mm.strip()
            y, m = mm.split("-")
            d0 = pd.Timestamp(year=int(y), month=int(m), day=1)
            d1 = d0 + pd.offsets.MonthEnd(1)
            dates += pd.date_range(d0, d1, freq="D").strftime("%Y-%m-%d").tolist()
    if a.start and a.end:
        dates += pd.date_range(pd.Timestamp(a.start), pd.Timestamp(a.end), freq="D").strftime("%Y-%m-%d").tolist()
    dates = sorted(set(dates))
    if not dates:
        print("[backfill] no dates specified", flush=True)
        return

    print(f"[backfill] {len(dates)} dates, models={models}", flush=True)
    summary = {}
    t_start = time.time()
    for i, td in enumerate(dates, 1):
        try:
            res = run_day(td, models, a.cutoff, a.force)
        except Exception as e:
            res = {"DAY_ERROR": str(e)}
            traceback.print_exc()
        summary[td] = res
        ok = [m for m, v in res.items() if v == "cached" or (isinstance(v, str) and v.startswith("ok"))]
        print(f"[backfill] {i}/{len(dates)} {td}: {res}  (ok_models={len(ok)})", flush=True)

    elapsed = time.time() - t_start
    out_path = Path(EFM3) / "outputs" / "runs" / "p2_2_backfill_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"models": models, "n_days": len(dates),
                   "elapsed_min": round(elapsed / 60, 1),
                   "started_at": datetime.now(timezone.utc).isoformat(),
                   "summary": summary}, f, indent=2, ensure_ascii=False, default=str)
    print(f"[backfill] DONE in {elapsed/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
