FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /workspace/reward

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        htop \
        nano \
        rsync \
        tmux && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir timm

COPY . .

CMD ["bash"]
