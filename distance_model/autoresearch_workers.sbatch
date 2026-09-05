#!/bin/bash
#SBATCH --job-name=ar-direct
#SBATCH --partition=gpu
#SBATCH --nodelist=amd-2
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:10:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/hulajan1/swbench/ar_20260905_direct/logs/a40_%A_%a.log
# Fixed-budget discovery kill: final validation corr > matched coat + .005 OR
# MAE < .97 * coat MAE. No pass => at most two one-knob near-miss mutations.
# Surviving models: S11 AND S13, historical heldout, 160k, seeds0/1, matched coat.
# GPU must be Slurm-allocated, non-display, free of compute processes at launch.
set -euo pipefail
ROOT=/home/hulajan1/swbench/ar_20260905_direct
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$ROOT/src"
# Resolve the physical UUID THROUGH Slurm's assigned CUDA mask, then release the
# short-lived introspection process before the worker checks GPU occupancy.
GPU_UUID=$(python3 -c 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_properties(0).uuid)')
GPU_UUID="GPU-${GPU_UUID#GPU-}"
printf 'ALLOCATED_GPU %s job=%s task=%s\n' "$GPU_UUID" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID"
sleep 2
nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
# The allocation remains idle while waiting. Training still requires zero activity,
# negligible memory and no compute processes, checked again inside the worker.
for attempt in {1..10}; do
  if python3 -c 'import sys; from autoresearch_loop import gpu_idle; idle,reason=gpu_idle(sys.argv[1]); print(reason,flush=True); sys.exit(0 if idle else 1)' "$GPU_UUID"; then
    break
  fi
  sleep 10
done
python3 autoresearch_loop.py worker --root "$ROOT" --group a40 --gpu "$GPU_UUID" \
  --ready-file "$ROOT/prepare_G13_s0_validation.done"
printf 'WORKER_DONE job=%s task=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID"
