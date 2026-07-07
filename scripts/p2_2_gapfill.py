"""P2.2 gap-filler.

After the parallel supervisor finishes, scan all target months x 4 models and
relaunch a worker ONLY for (model, month) pairs that still have missing CSVs.
Workers skip cached days, so this only recomputes the gaps.

Run standalone or from the supervisor. Idempotent.
"""
from __future__ import annotations
import os
import sys
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
if EFM3 not in sys.path:
    sys.path.insert(0, EFM3)
import pandas as pd

PY = "D:/computer_download/environment/conda/epf-2/python.exe"
WS = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_deep_sgdf_delta/scripts"
MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
MONTHS = ["2025-03", "2025-04", "2025-05", "2025-06", "2025-09", "2025-10",
          "2026-03", "2026-04", "2026-05", "2026-06"]


def month_days(month):
    y, m = month.split("-")
    d0 = pd.Timestamp(year=int(y), month=int(m), day=1)
    d1 = d0 + pd.offsets.MonthEnd(1)
    return pd.date_range(d0, d1, freq="D").strftime("%Y-%m-%d").tolist()


def missing_months_for_model(model):
    miss = []
    for mm in MONTHS:
        for d in month_days(mm):
            out = Path(EFM3) / "outputs" / "runs" / d / "realtime" / "prediction" / f"{model}_predictions.csv"
            if not out.exists():
                miss.append(mm)
                break
    return miss


def main():
    t0 = time.time()
    plan = []  # (model, months_comma)
    for model in MODELS:
        mms = missing_months_for_model(model)
        if mms:
            plan.append((model, mms))
            print(f"[gapfill] {model}: missing months {mms}", flush=True)
        else:
            print(f"[gapfill] {model}: complete", flush=True)

    if not plan:
        print("[gapfill] nothing missing — all done", flush=True)
        return

    procs = []
    for model, mms in plan:
        months_csv = ",".join(mms)
        log = Path(EFM3) / "outputs" / "runs" / "p2_2_logs" / f"gap_{model}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(
            [PY, f"{WS}/p2_2_worker.py", "--model", model, "--months", months_csv, "--chunkid", f"gap_{model}"],
            stdout=open(log, "w"), stderr=subprocess.STDOUT,
        )
        procs.append((model, p))
        print(f"[gapfill] launched worker for {model} months={months_csv} pid={p.pid}", flush=True)

    for model, p in procs:
        rc = p.wait()
        print(f"[gapfill] {model} gap worker finished rc={rc}", flush=True)

    # re-scan to report remaining gaps
    total_missing = 0
    for model in MODELS:
        for mm in MONTHS:
            for d in month_days(mm):
                out = Path(EFM3) / "outputs" / "runs" / d / "realtime" / "prediction" / f"{model}_predictions.csv"
                if not out.exists():
                    total_missing += 1
    print(f"[gapfill] DONE in {round((time.time()-t0)/60,1)} min; remaining missing CSVs={total_missing}", flush=True)


if __name__ == "__main__":
    main()
