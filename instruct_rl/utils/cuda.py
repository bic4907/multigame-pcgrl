import os
import subprocess

CUDA_VERSION = None

def get_cuda_version():
    global CUDA_VERSION  # To modify the global variable
    CUDA_VERSION = False  # Default value if no valid version is found
    try:
        # Run the command and suppress console output
        result = subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        cuda_version = result.stdout
        for line in cuda_version.split("\n"):
            if "release" in line:
                version = line.split("release")[-1].strip().split(" ")[0]
                version = version.replace(",", "")
                CUDA_VERSION = float(version)
                break
    except Exception as e:
        # Handle the error gracefully
        CUDA_VERSION = False
    return CUDA_VERSION

def get_gpu_memory():
    memory_stats = dict()
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout.strip().split('\n')

        for line in output:
            total, used, free = map(int, line.split(', '))
            memory_stats = {
                'total': total,
                'used': used,
                'free': free
            }
    except Exception:
        memory_stats = {
            'total': 0,
            'used': 0,
            'free': 0
        }
    return memory_stats


if __name__ == "__main__":
    print(f'CUDA: {get_cuda_version()}')
    print(f'GPU Memory: {get_gpu_memory()}')