#!/bin/bash
# auto_shutdown.sh
# GPU utilization(utilization)  checked every 30 seconds
# 30text(=60text) shut down the node after continuous idle readings.

IDLE_THRESHOLD=3          # Treat all GPUs as idle when utilization is at or below this percentage
CHECK_INTERVAL=30         # check interval (seconds)
MAX_IDLE_COUNT=20         # shut down after this many consecutive idle checks (30seconds × 20 = 10text)
LOG_FILE="/tmp/auto_shutdown.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Check whether nvidia-smi exists
if ! command -v nvidia-smi &>/dev/null; then
    log "ERROR: nvidia-smi not found. Exiting without shutdown."
    exit 1
fi

log "===== auto_shutdown.sh started ====="
log "  check interval : ${CHECK_INTERVAL}s"
log "  idle threshold : all GPUs util <= ${IDLE_THRESHOLD}%"
log "  shutdown after : $((MAX_IDLE_COUNT * CHECK_INTERVAL / 60)) min of consecutive idle"

idle_count=0
idle_since=""

while true; do
    # multi-GPU: read each GPU utilization and use the maximum
    # consider the node active if any GPU exceeds IDLE_THRESHOLD
    max_util=0
    gpu_utils=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    gpu_summary=""
    while IFS=',' read -r idx util; do
        idx=$(echo "$idx" | tr -d ' ')
        util=$(echo "$util" | tr -d ' ')
        gpu_summary="${gpu_summary} GPU${idx}=${util}%"
        if [ "$util" -gt "$max_util" ]; then
            max_util=$util
        fi
    done <<< "$gpu_utils"

    if [ "$max_util" -le "$IDLE_THRESHOLD" ]; then
        # record idle start time
        if [ "$idle_count" -eq 0 ]; then
            idle_since=$(date '+%Y-%m-%d %H:%M:%S')
        fi
        idle_count=$((idle_count + 1))
        remaining_sec=$(( (MAX_IDLE_COUNT - idle_count) * CHECK_INTERVAL ))
        shutdown_eta=$(date -d "+${remaining_sec} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                       || date -v "+${remaining_sec}S" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                       || echo "N/A")
        log "[IDLE]   [${gpu_summary} ] max=${max_util}% | idle since=${idle_since} | count=${idle_count}/${MAX_IDLE_COUNT} | remaining=${remaining_sec}s | ETA shutdown=${shutdown_eta}"
    else
        if [ "$idle_count" -gt 0 ]; then
            log "[ACTIVE] [${gpu_summary} ] max=${max_util}%. Resetting idle counter (was idle since ${idle_since})."
        else
            log "[ACTIVE] [${gpu_summary} ] max=${max_util}%. Watching..."
        fi
        idle_count=0
        idle_since=""
    fi

    if [ "$idle_count" -ge "$MAX_IDLE_COUNT" ]; then
        log "===== $((MAX_IDLE_COUNT * CHECK_INTERVAL / 60)) min of GPU idle detected. Stopping RunPod now... ====="
        if [ -z "$RUNPOD_POD_ID" ]; then
            log "ERROR: RUNPOD_POD_ID env var not set. Cannot stop pod."
            exit 1
        fi
        log "Running: runpodctl stop pod $RUNPOD_POD_ID"
        runpodctl stop pod "$RUNPOD_POD_ID"
        exit 0
    fi

    sleep "$CHECK_INTERVAL"
done

