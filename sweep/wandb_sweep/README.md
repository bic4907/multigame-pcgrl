# WandB Sweep Configuration

Hyperparameter optimization using Weights & Biases sweeps with Bayesian Optimization.

## Files

- `e2e_single.yaml`: End-to-end training sweep configuration

## Quick Start

### 1. Initialize Sweep

```bash
wandb sweep --project multigame_train --entity st4889ha-gwangju-institute-of-science-and-technology sweep/e2e_single.yaml
```

Output example:
```
Created sweep with ID: abc123xyz
Run sweep agent with: wandb agent st4889ha-gwangju-institute-of-science-and-technology/multigame_train/abc123xyz
```

### 2. Run Agent

```bash
GPU=0 ./run_docker.sh wandb agent st4889ha-gwangju-institute-of-science-and-technology/multigame_train/SWEEP_ID
```

Replace `SWEEP_ID` with the actual ID from step 1.

### 3. Parallel Execution (Multiple GPUs)

```bash
# Terminal 1
GPU=0 ./run_docker.sh wandb agent st4889ha-gwangju-institute-of-science-and-technology/multigame_train/SWEEP_ID

# Terminal 2
GPU=1 ./run_docker.sh wandb agent st4889ha-gwangju-institute-of-science-and-technology/multigame_train/SWEEP_ID

# Terminal 3
GPU=2 ./run_docker.sh wandb agent st4889ha-gwangju-institute-of-science-and-technology/multigame_train/SWEEP_ID
```

## Monitoring

View sweep progress at:
```
https://wandb.ai/st4889ha-gwangju-institute-of-science-and-technology/multigame_train/sweeps/{SWEEP_ID}
```

Check running containers:
```bash
docker ps
docker logs multigame-pcgrl_gpu0_20260131
```


## References

- [WandBSweeps Documentation](https://docs.wandb.ai/guides/sweeps)


