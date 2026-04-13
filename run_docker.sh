#!/bin/bash

# Docker-based execution script (auto GPU allocation and logging)
# Usage: ./run_docker.sh <command> [args...]
# Example: ./run_docker.sh python train.py exp_name=test
#          ./run_docker.sh wandb login
#
# 환경 변수:
#   DOCKER_IMAGE  — 사용할 이미지 (기본: mgpcgrl:latest)
#   GPU           — 사용할 GPU 번호 강제 지정 (기본: 여유 VRAM 가장 많은 GPU 자동 선택)

# Get all command arguments
COMMAND="$@"

if [ -z "$COMMAND" ]; then
    echo "Usage: ./run_docker.sh <command> [args...]"
    echo "Example: ./run_docker.sh python train.py exp_name=test"
    exit 1
fi

# 사용할 이미지 결정 (환경변수 DOCKER_IMAGE로 덮어쓰기 가능)
docker_image="${DOCKER_IMAGE:-ghkdrmaghks/multigame}"
echo "Using Docker image: $docker_image"

# 로그 디렉토리 설정
mkdir -p output_logs error_logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="output_logs/output_${timestamp}.log"
error_log_file="error_logs/error_${timestamp}.log"

# GPU 선택 로직
echo "Searching for available GPU..."

# `nvidia-smi`로 GPU 메모리 사용량 확인
gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits)
available_gpu=$(echo "$gpu_info" | awk -F, '{if ($5 > 0) print $1 " " $5}' | sort -k2 -nr | head -n1 | cut -d' ' -f1)

# IF $CUDA_VISIBLE_DEVICES is set, use it
if [ -n "$GPU" ]; then
    available_gpu=$GPU
fi

if [ -z "$available_gpu" ]; then
    echo "No available GPU found!" | tee -a "$error_log_file"
    exit 1
fi

# 선택된 GPU 상세 정보 가져오기
selected_gpu_info=$(echo "$gpu_info" | awk -v gpu_id="$available_gpu" -F, '{if ($1 == gpu_id) print}')
selected_gpu_name=$(echo "$selected_gpu_info" | cut -d, -f2)
selected_gpu_total_mem=$(echo "$selected_gpu_info" | cut -d, -f3)
selected_gpu_used_mem=$(echo "$selected_gpu_info" | cut -d, -f4)
selected_gpu_free_mem=$(echo "$selected_gpu_info" | cut -d, -f5)

# 선택된 GPU 출력
echo "Selected GPU: $available_gpu (GPU Number: $available_gpu)" | tee -a "$log_file"
echo "GPU Details:" | tee -a "$log_file"
echo "  GPU ID: $available_gpu" | tee -a "$log_file"
echo "  Model Name: $selected_gpu_name" | tee -a "$log_file"
echo "  Total Memory: ${selected_gpu_total_mem}MiB" | tee -a "$log_file"
echo "  Used Memory: ${selected_gpu_used_mem}MiB" | tee -a "$log_file"
echo "  Free Memory: ${selected_gpu_free_mem}MiB" | tee -a "$log_file"


# 컨테이너 이름 생성 (GPU 번호 + 날짜)
date_str=$(date +"%Y%m%d%H%M%S")
container_name="multigame_gpu${available_gpu}_${date_str}"

echo "Container Name: $container_name"
echo "Output Log File: $log_file"


root_args=("traj_path")

for arg in "$@"; do
    # 인자의 이름과 값을 '='로 분리
    key=$(echo "$arg" | cut -d '=' -f 1)

    # 제외할 인자가 아닌 경우에만 처리
    for root_arg in "${root_args[@]}"; do
        if [[ "$key" == "root_arg" ]]; then
            user_param="-u $(id -u):$(id -g)"
            break
        fi
    done
done


# Docker execution command
docker_command="docker run --rm -it
    -v $(pwd):/workspace
    -w /workspace
    --gpus all
    -e CUDA_VISIBLE_DEVICES=$available_gpu
    -e XLA_PYTHON_CLIENT_PREALLOCATE=true
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=.95
    -v /mnt/nas:/mnt/nas
    -v /raid:/raid
    --env-file .env
    -v $(pwd)/.netrc:/.netrc
    --network=host
    -e HF_HOME=/workspace/cache/huggingface
    --name \"$container_name\"
    $user_param \
    $docker_image
    $COMMAND"

echo "Executing Docker command:" | tee -a "$log_file"
echo "$docker_command" | tee -a "$log_file"

# Docker 명령 실행 및 로그 기록
{
    eval $docker_command
} 2>&1 | tee -a "$log_file"

# 실행 결과 확인
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Execution failed. Check logs for details." | tee -a "$error_log_file"
    echo "Docker logs (last 10 lines of $log_file):" | tee -a "$error_log_file"
    tail -n 10 "$log_file" | tee -a "$error_log_file"
    exit $exit_code
else
    echo "Execution completed successfully." | tee -a "$log_file"
fi

exit $exit_code
