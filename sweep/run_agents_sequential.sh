#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_agents_sequential.sh  —  여러 sweep을 순차적으로 실행
#
# 사용법:
#   GPUS="0 1 2 3" bash sweep/run_agents_sequential.sh \
#       entity/project/SWEEP_ID_1 \
#       entity/project/SWEEP_ID_2 \
#       entity/project/SWEEP_ID_3
#
# - 각 sweep의 run이 소진되면 agent가 자동 종료 (wandb grid sweep 기준)
# - 앞 sweep 완료 → 뒤 sweep 자동 시작
# - GPUS 미지정 시 GPU 0번만 사용
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0}"
read -ra GPU_LIST <<< "$GPUS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_agents() {
    local sweep_id="$1"
    local pids=()
    log "▶ Starting agents for: $sweep_id  (GPUs: ${GPU_LIST[*]})"
    for GPU_ID in "${GPU_LIST[@]}"; do
        if command -v docker &>/dev/null; then
            GPU=$GPU_ID bash "$ROOT_DIR/run_docker.sh" wandb agent "$sweep_id" &
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent "$sweep_id" &
        fi
        pids+=($!)
    done
    wait "${pids[@]}"
    log "✓ Done: $sweep_id"
}

if [ $# -eq 0 ]; then
    echo "Usage: GPUS=\"0 1 2 3\" bash $0 SWEEP_ID_1 [SWEEP_ID_2 ...]"
    exit 1
fi

for sweep_id in "$@"; do
    run_agents "$sweep_id"
done

log "All sweeps completed."

# RunPod 환경에서만 자동 정지
if [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "RunPod detected (POD_ID=$RUNPOD_POD_ID). Stopping pod..."
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
