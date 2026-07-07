"""P2.2 finalize driver — waits for the backfill orchestrator, then populates the
ledger and runs the analysis. Launched as a tracked background task so the whole
pipeline (backfill -> populate -> analyze) self-completes and notifies once.

Robustness:
  * Detects orchestrator completion by the "COMPLETE" marker in orchestrator.log.
  * Hard timeout (9.5h) prevents infinite wait if the orchestrator crashes.
  * Periodically prints progress so the run is observable in the editor.
  * On orchestrator completion, runs populate_ledger then analyze, capturing output.
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
ORCH_LOG = LOGD / "orchestrator.log"
EXPORT_DIR = Path("D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_deep_sgdf_delta/exports/efm3_candidates/realtime_ensemble/p2_2_multiseason")

ENV = os.environ.copy()
ENV["OPTIM_NUM_WORKERS"] = "0"
ENV["OPTIM_PIN_MEMORY"] = "0"
ENV.pop("HF_ENDPOINT", None)
ENV["PROJECT_ROOT"] = EFM3

MAX_WAIT = 18 * 3600
CHECK_INTERVAL = 60
PROGRESS_EVERY = 30 * 60  # print progress every 30 min


def orch_done():
    if ORCH_LOG.exists():
        try:
            txt = ORCH_LOG.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""
        return "COMPLETE" in txt
    return False


def main():
    t0 = time.time()
    print(f"[finalize] {datetime.now(timezone.utc).isoformat()} waiting for backfill orchestrator "
          f"(max_wait={MAX_WAIT/3600:.1f}h) ...", flush=True)
    last_progress = 0
    while time.time() - t0 < MAX_WAIT:
        if orch_done():
            print(f"[finalize] orchestrator COMPLETE after {round((time.time()-t0)/60,1)} min", flush=True)
            break
        if time.time() - last_progress >= PROGRESS_EVERY:
            elapsed = round((time.time() - t0) / 60, 1)
            # quick progress snapshot from orchestrator log
            snap = ""
            try:
                lines = ORCH_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
                snap = lines[-3:] if lines else []
            except Exception:
                snap = []
            print(f"[finalize] t={elapsed}min still running...", flush=True)
            for s in snap:
                print("   | " + s, flush=True)
            last_progress = time.time()
        time.sleep(CHECK_INTERVAL)
    else:
        print("[finalize] WARNING: timed out waiting for orchestrator COMPLETE; "
              "proceeding to populate+analyze with whatever data exists", flush=True)

    # snapshot backfill coverage before populating
    print(f"[finalize] backfill elapsed={round((time.time()-t0)/60,1)}min; running populate_ledger ...", flush=True)
    rp = subprocess.run([PY, f"{WS}/p2_2_populate_ledger.py"], env=ENV, capture_output=True, text=True)
    print(rp.stdout[-2500:], flush=True)
    if rp.stderr:
        print("POPULATE STDERR:\n", rp.stderr[-1500:], flush=True)

    print(f"[finalize] running analyze ...", flush=True)
    ra = subprocess.run([PY, f"{WS}/p2_2_analyze.py"], env=ENV, capture_output=True, text=True)
    print(ra.stdout[-4000:], flush=True)
    if ra.stderr:
        print("ANALYZE STDERR (tail):\n", ra.stderr[-2500:], flush=True)

    # report deliverables
    print("\n[finalize] === DELIVERABLES ===", flush=True)
    if EXPORT_DIR.exists():
        for f in sorted(EXPORT_DIR.iterdir()):
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)
    else:
        print("  EXPORT_DIR missing!", flush=True)
    print(f"\n[finalize] ALL DONE at {datetime.now(timezone.utc).isoformat()} "
          f"total={round((time.time()-t0)/60,1)} min", flush=True)


if __name__ == "__main__":
    main()
