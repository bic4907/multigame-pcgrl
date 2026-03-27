# RunPod Multi-Process Deployment

Multi-pod parallel execution with real-time monitoring and safe cleanup mechanisms.

## Features

- 🚀 **Parallel Execution**: Run multiple pods simultaneously
- 📊 **Real-time Dashboard**: Live status monitoring with activity logs
- 🛡️ **Safe Cleanup**: Automatic pod termination on interruption
- ⏱️ **Timeout Support**: Configurable execution time limits
- 🎯 **Signal Handling**: Graceful shutdown on Ctrl+C or kill signals

## Safety Mechanisms

The deployment script includes multiple safety layers to ensure all pods are properly cleaned up:

1. **Global Pod Tracking**: All created pods are registered immediately after creation
2. **Signal Handlers**: 
   - `SIGINT` (Ctrl+C): Triggers emergency cleanup
   - `SIGTERM` (kill): Triggers emergency cleanup
3. **Exit Handler**: `atexit` ensures cleanup on normal termination
4. **Exception Handling**: Cleanup on errors within pod processes
5. **Emergency Cleanup**: Terminates all tracked pods if script is interrupted

### Testing Safety

To test the cleanup mechanism:

```bash
# Start pods and press Ctrl+C during execution
python runpod/deploy.py --configs config1.yaml config2.yaml

# Press Ctrl+C - you should see:
# "⚠️  Interrupted by user (Ctrl+C)"
# "🚨 Emergency cleanup initiated..."
# "Terminating N active pod(s)..."
```

All pods will be automatically terminated even if the script is interrupted.

## Usage

### Single Pod

```bash
python runpod/deploy.py --config config.yaml
```

### Multiple Pods (Parallel)

```bash
python runpod/deploy.py --configs config1.yaml config2.yaml config3.yaml
```

### WandB Sweep Example

```bash
# Start pods for sweep
GPU=0 python runpod/deploy.py --configs sweep_config1.yaml sweep_config2.yaml

# Or with sweep command
wandb sweep --project multigame_train sweep/train.yaml --entity your-entity
python runpod/deploy.py --configs sweep_agent1.yaml sweep_agent2.yaml
```

## Configuration File Format

```yaml
name: "my-experiment"

pod:
  template_id: "zkat0jqlju"
  gpu: "NVIDIA RTX A4000"
  gpu_count: 1
  spot: false

runtime:
  cmds:
    - "cd /workspace/multigame-pcgrl"
    - "wandb login YOUR_API_KEY"
    - "python train.py n_epochs=100 batch_size=512"

options:
  timeout: true
  time_limit: 7200  # 2 hours
  terminate: true   # Auto-terminate after completion
```

## Dashboard

The dashboard shows:
- **Status Table**: Pod index, name, ID, status, progress, runtime, errors
- **Activity Log**: Real-time log of status changes (last 15 entries)

### Status Icons

- ⠋ `CREATING`: Pod is being created (with spinner animation)
- 🔧 `STARTING`: Pod is starting up
- ✅ `RUNNING`: Pod is executing commands
- ✅ `EXITED`: Pod completed successfully
- 🛑 `TERMINATED`: Pod was terminated by script
- ⏱️ `TIMEOUT`: Pod exceeded time limit
- ❌ `ERROR`: Pod encountered an error

## Summary Report

After all pods complete, a summary is displayed:
- Total pods launched
- ✅ Completed successfully
- ❌ Errors encountered
- ⏱️ Timeouts occurred

## Emergency Interruption

If you need to stop all pods immediately:

1. **Press Ctrl+C** - Triggers safe cleanup
2. **Wait for cleanup** - Script will terminate all active pods
3. **Verify on RunPod dashboard** - All pods should be stopped

The script ensures all pods are properly terminated even if:
- Python crashes
- Terminal is closed (if using nohup)
- Script is killed with SIGTERM
- Unexpected exceptions occur

## Notes

- Pods are automatically registered for cleanup upon creation
- Cleanup runs even if script crashes or is interrupted
- Each pod unregisters itself after successful termination
- Emergency cleanup is idempotent (safe to call multiple times)
- Pod creation typically takes 3-5 minutes

