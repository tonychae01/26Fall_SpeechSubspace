# Cluster GPU Workflow

User-provided cluster guidance. Read live Slurm state before each job; prior observations do not guarantee current availability.

## Shell and Environments

- Source `~/.config/cluster/shell.sh` in noninteractive shells. Interactive shells load it through `~/.bashrc`.
- Existing environment shortcuts: `cenv env-fcca` (Conda), `cenv env-ssm-fm` (venv). Do not activate an environment automatically.
- Conda environments also support `conda activate NAME`; list them with `conda info --envs`.
- Use the actual environment chosen for this project. The examples below do not select an environment for this repository.

## Before GPU Work

1. Check `gpu_status`, then `gpu_check` from the catscan login node.
2. Inspect existing allocations with `scontrol show job JOBID`. Reuse a matching allocation only after understanding its project, resources, running steps, and owner. Never cancel unrelated jobs.
3. Inside an allocation, use `gpu_here` / `nvidia-smi` to inspect actual device processes and memory.

Slurm allocation status and device utilization are different. Do not use GPUs allocated to others. If Slurm reports `Operation not permitted`, report the restriction without claiming availability was checked or an allocation was received. Do not bypass the sandbox.

## Batch Jobs

Prefer batch jobs for long project runs. Never run training on the login node.

```bash
source ~/.config/cluster/shell.sh
cd /clusterfs/bebb/users/tonychae01/projects/26Fall_SpeechSubspace
# Replace PROJECT_ENV and the entrypoint with the project's actual choices.
gpu_submit PROJECT_ENV python train.py
GPU_COUNT=3 GPU_TIME=04:00:00 gpu_submit PROJECT_ENV python train.py
```

`gpu_submit` prints a job ID and writes logs to the project's `logs/` directory. Check `squeue` and logs after submission. Do not create duplicate jobs for the same task.

Requesting three GPUs does not make single-GPU code use all three; configure the training framework accordingly. A single job requests all GPUs on one node. Separate project jobs can schedule independently.

## Interactive Jobs

Use `gpu_run 1` or `gpu_run 3`, then `cenv ENV_NAME`. Release completed interactive allocations with `exit`.

Inherited defaults are partition `RTX_4500_Ada`, 4 CPUs, 32 GB RAM, and 4 hours. These are previous configuration defaults, not verified partition limits. Check available Ada models and limits with `sinfo` and `scontrol show partition` before requesting resources. Override `GPU_PARTITION`, `GPU_CPUS`, `GPU_MEM`, and `GPU_TIME` when needed.

## Project Setup Notes (2026-11-07)

- `env-speech` is installed at `/clusterfs/bebb/users/tonychae01/envs/env-speech`.
- Use `bash 03_src/pipeline/run.sh python ...` inside `gpu_submit env-speech ...`
  to expose the pip-installed TensorFlow CUDA libraries.
- Explicitly `cd /clusterfs/bebb/users/tonychae01/projects/26Fall_SpeechSubspace`
  before submission. The login node's resolved `/pool0/clusterfs/...` path caused
  the first job to fail before log creation; the shared path worked.
- The initial restricted Slurm query failed. An approved retry succeeded; do not
  mistake that earlier failure for a permanent cluster outage.
- Live checks showed two RTX 4500 Ada GPUs per node in this partition. A three-GPU
  request on one Ada node would not fit those observed resources. Recheck live state.
