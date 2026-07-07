"""P2.2 research driver: bypass main.py's segfaulting import chain.

Runs 2.5 realtime model predictions by calling pipelines.ledger_predict
internals directly (clean import context where torch import does not crash).
Writes outputs into the efm3.0 working tree (outputs/runs + outputs/ledger),
which is research-only and NEVER writes submission_ready.csv.

Usage:
  python scripts/p2_2_probe_driver.py --date 2025-03-03 --models timemixer
"""
from __future__ import annotations
import os, sys, argparse, json
from datetime import datetime, timezone
from pathlib import Path

# CRITICAL ORDER: load torch (and numpy) from site-packages FIRST, while EFM3
# is NOT yet on sys.path. If EFM3 is on the path when torch's C-extension
# initializes its global deps (OpenMP/MKL), the import segfaults intermittently.
# Pre-loading here makes the scheduler worker thread's `import torch` a no-op.
import numpy as np  # noqa: F401
import torch  # noqa: F401
print(f"[driver] torch {torch.__version__} preloaded (cuda={torch.cuda.is_available()})", flush=True)

EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
if EFM3 not in sys.path:
    sys.path.insert(0, EFM3)

import pandas as pd
from pipelines.ledger_predict import _run_model_set  # clean import


def run_one(target_date: str, task: str, models: list[str], rt_cutoff_hour: int = 14, force: bool = True):
    da_cutoff = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rt_cutoff = f"{da_cutoff} {rt_cutoff_hour:02d}:00:00"
    run_dir = os.path.join(EFM3, "outputs", "runs", target_date)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    res = _run_model_set(
        target_date=target_date,
        task=task,
        models=models,
        data_path=os.path.join(EFM3, "data", "shandong_pmos_hourly.xlsx"),
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
        run_dir=Path(run_dir),
        max_cpu=1,
        max_gpu=1,
        force=force,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--models", default="timesfm,sgdfnet,timemixer,rt916")
    ap.add_argument("--task", default="realtime")
    ap.add_argument("--cutoff", type=int, default=14)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    print(f"[probe] date={a.date} task={a.task} models={models}", flush=True)
    res = run_one(a.date, a.task, models, a.cutoff)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v) for k, v in res.items()}, ensure_ascii=False, indent=2))
    # show produced files
    d = os.path.join(EFM3, "outputs", "runs", a.date, a.task, "prediction")
    if os.path.isdir(d):
        print("FILES:", sorted(os.listdir(d)))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
