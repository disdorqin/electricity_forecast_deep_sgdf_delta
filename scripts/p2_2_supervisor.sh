#!/usr/bin/env bash
# P2.2 parallel backfill supervisor.
# Launches one worker process per (model, month-chunk) in parallel, waits for all,
# then runs the gap-filler to recover any days skipped by a crash.
set -u
cd "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0"
export OPTIM_NUM_WORKERS=0
export OPTIM_PIN_MEMORY=0
# unset HF_ENDPOINT so huggingface/timesfm uses default endpoint via proxy (hf-mirror redirects & fails)
unset HF_ENDPOINT
PY="D:/computer_download/environment/conda/epf-2/python.exe"
WS="D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_deep_sgdf_delta/scripts"
LOGD="D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/efm3.0/outputs/runs/p2_2_logs"
mkdir -p "$LOGD"

launch() {
  local model="$1"; local months="$2"; local chunkid="$3"
  nohup "$PY" "$WS/p2_2_worker.py" --model "$model" --months "$months" --chunkid "$chunkid" \
    > "$LOGD/worker_${model}_${chunkid}.log" 2>&1 &
  echo "launched $model/$chunkid pid=$!"
}

echo "=== launching rt916 (10 month-workers) ==="
launch rt916 2025-03 r01
launch rt916 2025-04 r02
launch rt916 2025-05 r03
launch rt916 2025-06 r04
launch rt916 2025-09 r05
launch rt916 2025-10 r06
launch rt916 2026-03 r07
launch rt916 2026-04 r08
launch rt916 2026-05 r09
launch rt916 2026-06 r10

echo "=== launching timemixer (3 workers) ==="
launch timemixer 2025-03,2025-04,2025-05,2025-06 t01
launch timemixer 2025-09,2025-10 t02
launch timemixer 2026-03,2026-04,2026-05,2026-06 t03

echo "=== launching timesfm (2 workers) ==="
launch timesfm 2025-03,2025-04,2025-05,2025-06,2025-09,2025-10 f01
launch timesfm 2026-03,2026-04,2026-05,2026-06 f02

echo "=== launching sgdfnet (4 CPU workers) ==="
launch sgdfnet 2025-03,2025-04 s01
launch sgdfnet 2025-05,2025-06 s02
launch sgdfnet 2025-09,2025-10 s03
launch sgdfnet 2026-03,2026-04,2026-05,2026-06 s04

echo "=== all workers launched; waiting ==="
wait
echo "=== ALL WORKERS DONE at $(date) ==="

echo "=== running gap-fill for any missing days ==="
"$PY" "$WS/p2_2_gapfill.py"
echo "=== SUPERVISOR COMPLETE at $(date) ==="
