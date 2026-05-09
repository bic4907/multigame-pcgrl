#!/usr/bin/env bash
# =============================================================================
# kill_pretrained_clip.sh
# =============================================================================
# Stops processes launched by run_pretrained_clip_all_gpus.sh.
#
# Usage:
#   bash tests/kill_pretrained_clip.sh           # stop all (PID file)
#   bash tests/kill_pretrained_clip.sh all        # same as above
#   bash tests/kill_pretrained_clip.sh 0          # stop GPU 0 process only
#   bash tests/kill_pretrained_clip.sh 0 2 3      # stop GPU 0, 2, 3
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/run_dummy.pids"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Parse arguments ───────────────────────────────────────────────────────────
GPU_ARGS=()
for ARG in "$@"; do
    if [[ "${ARG}" == "all" ]] || [[ "${ARG}" =~ ^[0-9]+$ ]]; then
        GPU_ARGS+=("${ARG}")
    fi
done

_kill_pid() {
    local PID="$1"
    [[ -z "${PID}" ]] && return
    if kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}"
        echo "[INFO] Killed: PID ${PID}"
        ((KILLED++)) || true
    else
        echo "[WARN] Already gone: PID ${PID}"
        ((MISSING++)) || true
    fi
}

KILLED=0
MISSING=0

# ── Kill specific GPUs ────────────────────────────────────────────────────────
if [[ ${#GPU_ARGS[@]} -gt 0 ]] && [[ "${GPU_ARGS[0]}" != "all" ]]; then
    echo "[INFO] Stopping GPU(s): ${GPU_ARGS[*]}"
    for GPU_IDX in "${GPU_ARGS[@]}"; do
        # Try pgrep first
        PIDS=$(pgrep -f "CUDA_VISIBLE_DEVICES=${GPU_IDX}.*train_pretrained_clip" 2>/dev/null || true)
        if [[ -z "${PIDS}" ]]; then
            # Fallback: extract by line number from PID file
            if [[ -f "${PID_FILE}" ]]; then
                PIDS=$(sed -n "$((GPU_IDX + 1))p" "${PID_FILE}" || true)
            fi
        fi
        if [[ -z "${PIDS}" ]]; then
            echo "[WARN] GPU ${GPU_IDX}: no running process found."
            ((MISSING++)) || true
        else
            for PID in ${PIDS}; do _kill_pid "${PID}"; done
        fi
    done

# ── Kill all (no args or "all") ───────────────────────────────────────────────
else
    if [[ ! -f "${PID_FILE}" ]]; then
        echo "[WARN] PID file not found: ${PID_FILE}"
        echo "       No processes running or already stopped."
        exit 0
    fi
    echo "[INFO] Stopping all processes (from PID file)"
    while IFS= read -r PID; do _kill_pid "${PID}"; done < "${PID_FILE}"
    rm -f "${PID_FILE}"
fi

echo ""
echo "[INFO] Done — killed: ${KILLED} / already gone: ${MISSING}"
