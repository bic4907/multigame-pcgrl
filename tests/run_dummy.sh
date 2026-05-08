#!/usr/bin/env bash
# =============================================================================
# run_dummy.sh
# =============================================================================
# Runs train_pretrained_clip.py in parallel on the specified GPU(s).
#
# Usage:
#   bash tests/run_dummy.sh all
#   bash tests/run_dummy.sh 0
#   bash tests/run_dummy.sh 0 1 2
#   bash tests/run_dummy.sh all dataset_game=dungeon
#   bash tests/run_dummy.sh 0 2 -- dataset_game=dungeon
#   bash tests/run_dummy.sh --base-seed 10 all        # seeds: 10, 11, 12, ...
#
# Leading numeric / "all" arguments select GPUs.
# --base-seed N : base seed (default 0). GPU i gets seed = base_seed + i.
# Arguments after "--" (or non-numeric/non-flag arguments) → hydra overrides.
# seed=* in hydra overrides is stripped automatically to avoid conflicts.
# PIDs are saved to tests/run_dummy.pids.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${SCRIPT_DIR}/run_dummy.pids"

# ── Parse arguments ───────────────────────────────────────────────────────────
GPU_ARGS=()
EXTRA_ARGS=()
SEPARATOR_SEEN=false
BASE_SEED=0

_ARGS=("$@")
i=0
while [[ $i -lt ${#_ARGS[@]} ]]; do
    ARG="${_ARGS[$i]}"
    if [[ "${ARG}" == "--" ]]; then
        SEPARATOR_SEEN=true
        ((i++)) || true; continue
    fi
    if ${SEPARATOR_SEEN}; then
        EXTRA_ARGS+=("${ARG}")
    elif [[ "${ARG}" == "--base-seed" ]]; then
        ((i++)) || true
        BASE_SEED="${_ARGS[$i]}"
    elif [[ "${ARG}" == "all" ]] || [[ "${ARG}" =~ ^[0-9]+$ ]]; then
        GPU_ARGS+=("${ARG}")
    else
        EXTRA_ARGS+=("${ARG}")
    fi
    ((i++)) || true
done

# Strip any seed=* passed via hydra overrides (to avoid overriding per-GPU seed)
FILTERED_EXTRA=()
for A in "${EXTRA_ARGS[@]}"; do
    [[ "${A}" =~ ^seed= ]] && continue
    FILTERED_EXTRA+=("${A}")
done
EXTRA_ARGS=("${FILTERED_EXTRA[@]+"${FILTERED_EXTRA[@]}"}")

# ── Resolve GPU list ──────────────────────────────────────────────────────────
TOTAL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ "${TOTAL_GPUS}" -eq 0 ]]; then
    echo "[ERROR] No GPUs detected via nvidia-smi." >&2
    exit 1
fi
echo "[INFO] Total GPUs available: ${TOTAL_GPUS}"

GPUS=()
if [[ ${#GPU_ARGS[@]} -eq 0 ]] || [[ "${GPU_ARGS[0]}" == "all" ]]; then
    for i in $(seq 0 $((TOTAL_GPUS - 1))); do GPUS+=("$i"); done
else
    for G in "${GPU_ARGS[@]}"; do
        if [[ "${G}" -ge "${TOTAL_GPUS}" ]]; then
            echo "[ERROR] GPU index ${G} is out of range (0~$((TOTAL_GPUS-1)))." >&2
            exit 1
        fi
        GPUS+=("${G}")
    done
fi

echo "[INFO] Target GPUs : ${GPUS[*]}"
echo "[INFO] Base seed   : ${BASE_SEED}"

# Reset PID file
: > "${PID_FILE}"

IDX=0
for GPU_IDX in "${GPUS[@]}"; do
    SEED=$((BASE_SEED + IDX))
    echo "[INFO] Starting GPU ${GPU_IDX}  seed=${SEED} (background, detached)"

    CUDA_VISIBLE_DEVICES="${GPU_IDX}" WANDB_MODE=offline \
        nohup python "${PROJECT_ROOT}/train_pretrained_clip.py" \
            seed="${SEED}" \
            exp_name="dummy" \
            overwrite=true \
            n_envs=600 \
            total_timesteps=1000000000 \
            "${EXTRA_ARGS[@]}" \
        > /dev/null 2>&1 &

    PID=$!
    disown "${PID}"
    echo "${PID}" >> "${PID_FILE}"
    echo "[INFO]   PID: ${PID}"
    ((IDX++)) || true
done

echo ""
echo "[INFO] ${#GPUS[@]} process(es) launched."
echo "[INFO] PID file: ${PID_FILE}"
echo "[INFO] To stop: bash tests/kill_dummy.sh"
