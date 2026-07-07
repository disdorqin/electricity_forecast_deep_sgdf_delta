"""P2.2 research backfill driver — bypasses main.py AND the ResourceScheduler.

Why: main.py's import chain + ResourceScheduler worker thread trigger a
torch C-extension segfault in this environment. We instead:
  1. pre-load torch in the MAIN thread (before anything else),
  2. call pipelines.ledger_predict._predict_model directly (no scheduler,
     no threaded set_global_seed -> import torch),
  3. write per-model CSVs into outputs/runs/{date}/{task}/prediction/.

This is research-only: it never writes submission_ready.csv.

Usage:
  python scripts/p2_2_backfill_driver.py --start 2025-03-01 --end 2025-03-31 \
      --task realtime --models timesfm,sgdfnet,timemixer,rt916
"""
from __future__ import annotations
import os, sys, argparse, json, traceback
from datetime import datetime, timezone
from pathlib import Path

# 1) Pre-load torch in the MAIN thread BEFORE importing anything from efm3.0.
import torch  # noqa: F401
print(f"[driver] torch {torch.__version__} preloaded (cuda={torch.cuda.is_available()})", flush=True)

EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
if EFM3 not in sys.path:
    sys.path.insert(0, EFM3)

import pandas as pd
from pipelines.ledger_predict import _predict_model  # clean, direct call


def run_day(target_date: str, task: str, models: list[str], rt_cutoff_hour: int = 14, force: bool = True):
    da_cutoff = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rt_cutoff = f"{da_cutoff} {rt_cutoff_hour:02d}:00:00"
    run_dir = Path(EFM3) / "outputs" / "runs" / target_date
    pred_dir = run_dir / task / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model in models:
        out = pred_dir / f"{model}_predictions.csv"
        if out.exists() and not force:
            results[model] = "cached"
            continue
        try:
            _predict_model(
                model_name=model,
                task=task,
                target_date=target_date,
                data_path=str(Path(EFM3) / "data" / "shandong_pmos_hourly.xlsx"),
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
            results[model] = "ok"
        except Exception as e:
            results[model] = f"ERROR: {type(e).__name__}: {e}"
            traceback.print_exc()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--models", default="timesfm,sgdfnet,timemixer,rt916")
    ap.add_argument("--task", default="realtime")
    ap.add_argument("--cutoff", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    # Build date range (business days only — skip if no data, the adapter will error)
    d0 = pd.Timestamp(a.start)
    d1 = pd.Timestamp(a.end)
    dates = pd.date_range(d0, d1, freq="D")
    print(f"[driver] {len(dates)} dates {a.start}..{a.end}, task={a.task}, models={models}", flush=True)

    summary = {}
    for d in dates:
        td = d.strftime("%Y-%m-%d")
        try:
            res = run_day(td, a.task, models, a.cutoff, a.force)
        except Exception as e:
            res = {"DAY_ERROR": str(e)}
            traceback.print_exc()
        summary[td] = res
        ok = [m for m, v in res.items() if v in ("ok", "cached")]
        print(f"[driver] {td}: {res}", flush=True)
    print("[driver] DONE", flush=True)
    with open(os.path.join(EFM3, "outputs", "runs", "p2_2_backfill_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"start": a.start, "end": a.end, "task": a.task, "models": models, "summary": summary},
                  f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
