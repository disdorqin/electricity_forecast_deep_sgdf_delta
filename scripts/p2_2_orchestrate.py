"""P2.2 backfill orchestrator — safe GPU-capped parallel launcher.

Spawns p2_2_worker.py subprocesses with a semaphore cap on GPU processes so we
never exceed the RTX 4060 (8 GB) VRAM budget. rt916 is VRAM-light (~470 MiB) and
gets a generous cap; timemixer/timesfm are heavier PyTorch models and share a
smaller cap. sgdfnet is CPU-only.

After every GPU/CPU job finishes, re-runs the gap-filler to recover any crashed
days, then exits. Resumable: workers skip cached (model,day) CSVs.

Jobs cover:
  * history-only months (NOT scored): 2025-01, 2025-02  -> trailing realtime
    history so the 2025-03 test window can learn genuine 2.5 GEF weights.
  * scored test windows: 2025-03..06, 2025-09,10, 2026-03..06.
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EFM3 = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
WS = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_deep_sgdf_delta/scripts"
PY = "D:/computer_download/environment/conda/epf-2/python.exe"
LOGD = Path(EFM3) / "outputs" / "runs" / "p2_2_logs"
LOGD.mkdir(parents=True, exist_ok=True)

# Caps tuned for RTX 4060 (8 GB):
#   rt916 ~0.47 GB each -> up to 8 light concurrent = 3.8 GB
#   timemixer/timesfm ~1.0-1.5 GB each -> up to 2 heavy concurrent = 3 GB
#   total headroom ~7 GB (safe)
MAX_LIGHT = 8   # rt916
MAX_HEAVY = 2   # timemixer, timesfm
MAX_CPU = 4     # sgdfnet

# (model, months_csv, chunkid, kind)  kind in {light, heavy, cpu}
JOBS = [
    # ---- rt916 (light) : 1 job per month for fine-grained resumability ----
    ("rt916", "2025-01", "r_hist01", "light"),
    ("rt916", "2025-02", "r_hist02", "light"),
    ("rt916", "2025-03", "r01", "light"),
    ("rt916", "2025-04", "r02", "light"),
    ("rt916", "2025-05", "r03", "light"),
    ("rt916", "2025-06", "r04", "light"),
    ("rt916", "2025-09", "r05", "light"),
    ("rt916", "2025-10", "r06", "light"),
    ("rt916", "2026-03", "r07", "light"),
    ("rt916", "2026-04", "r08", "light"),
    ("rt916", "2026-05", "r09", "light"),
    ("rt916", "2026-06", "r10", "light"),
    # ---- timemixer (heavy) ----
    ("timemixer", "2025-01,2025-02,2025-03,2025-04,2025-05,2025-06", "t01", "heavy"),
    ("timemixer", "2025-09,2025-10", "t02", "heavy"),
    ("timemixer", "2026-03,2026-04,2026-05,2026-06", "t03", "heavy"),
    # ---- timesfm (heavy, but fast) ----
    ("timesfm", "2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-09,2025-10", "f01", "heavy"),
    ("timesfm", "2026-03,2026-04,2026-05,2026-06", "f02", "heavy"),
    # ---- sgdfnet (cpu) ----
    ("sgdfnet", "2025-01,2025-02,2025-03,2025-04", "s01", "cpu"),
    ("sgdfnet", "2025-05,2025-06", "s02", "cpu"),
    ("sgdfnet", "2025-09,2025-10", "s03", "cpu"),
    ("sgdfnet", "2026-03,2026-04,2026-05,2026-06", "s04", "cpu"),
]

ENV = os.environ.copy()
ENV["OPTIM_NUM_WORKERS"] = "0"
ENV["OPTIM_PIN_MEMORY"] = "0"
ENV.pop("HF_ENDPOINT", None)  # use default hf endpoint via proxy
ENV["PROJECT_ROOT"] = EFM3


def start(job):
    model, months, chunkid, kind = job
    log = LOGD / f"worker_{model}_{chunkid}.log"
    p = subprocess.Popen(
        [PY, f"{WS}/p2_2_worker.py", "--model", model, "--months", months, "--chunkid", chunkid],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, env=ENV,
    )
    return p


def main():
    pending = list(JOBS)
    running = []  # (proc, job)
    t_start = time.time()
    print(f"[orch] {datetime.now(timezone.utc).isoformat()} starting {len(pending)} jobs "
          f"(light={MAX_LIGHT} heavy={MAX_HEAVY} cpu={MAX_CPU})", flush=True)

    while pending or running:
        # reap finished
        for item in list(running):
            p, job = item
            if p.poll() is not None:
                running.remove(item)
                print(f"[orch] finished {job[0]}/{job[2]} rc={p.returncode} "
                      f"(running={len(running)})", flush=True)
        # count by kind
        n_light = sum(1 for _, j in running if j[3] == "light")
        n_heavy = sum(1 for _, j in running if j[3] == "heavy")
        n_cpu = sum(1 for _, j in running if j[3] == "cpu")
        # launch what fits
        for job in list(pending):
            kind = job[3]
            if kind == "light" and n_light < MAX_LIGHT:
                running.append((start(job), job)); pending.remove(job); n_light += 1
                print(f"[orch] launched {job[0]}/{job[2]} (light {n_light}/{MAX_LIGHT})", flush=True)
            elif kind == "heavy" and n_heavy < MAX_HEAVY:
                running.append((start(job), job)); pending.remove(job); n_heavy += 1
                print(f"[orch] launched {job[0]}/{job[2]} (heavy {n_heavy}/{MAX_HEAVY})", flush=True)
            elif kind == "cpu" and n_cpu < MAX_CPU:
                running.append((start(job), job)); pending.remove(job); n_cpu += 1
                print(f"[orch] launched {job[0]}/{job[2]} (cpu {n_cpu}/{MAX_CPU})", flush=True)
        if not running and not pending:
            break
        time.sleep(5)

    print(f"[orch] all primary jobs done in {round((time.time()-t_start)/60,1)} min; "
          f"running gap-fill", flush=True)
    subprocess.run([PY, f"{WS}/p2_2_gapfill.py"], env=ENV, check=False)
    print(f"[orch] COMPLETE at {datetime.now(timezone.utc).isoformat()} "
          f"total={round((time.time()-t_start)/60,1)} min", flush=True)


if __name__ == "__main__":
    main()
