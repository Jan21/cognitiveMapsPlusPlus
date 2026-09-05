#!/bin/bash
#SBATCH --job-name=sok-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=dgx-2
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/hulajan1/swbench/sokoban_ar_20260905/logs/smoke_%A_%a.log
set -euo pipefail
ROOT=/home/hulajan1/swbench/sokoban_ar_20260905
PY="$ROOT/venv/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$ROOT/src:$ROOT/crtr"
cd "$ROOT/src"
sha256sum --check "$ROOT/src/SHA256SUMS"
test -f "$ROOT/banks/READY"
GPU_UUID=$("$PY" -c 'import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_properties(0).uuid)')
GPU_UUID="GPU-${GPU_UUID#GPU-}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
exec 9>"$ROOT/gpu-$GPU_UUID.lock"
flock -n 9
sleep 2
"$PY" -c 'import sys; from autoresearch_loop import gpu_idle; ok,why=gpu_idle(sys.argv[1]); print(why,flush=True); sys.exit(0 if ok else 1)' "$GPU_UUID"
KIND=joint
SKIND=joint
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then KIND=supervised; SKIND=supervised-ours; fi
PREFIX="$ROOT/artifacts/smoke_${SLURM_ARRAY_JOB_ID}_${KIND}"
"$PY" -u sokoban_joint.py --model "$KIND" --train-bank "$ROOT/banks/train.npz" \
  --val-bank "$ROOT/banks/val.npz" --test-bank "$ROOT/banks/smoke_test.npz" \
  --out "$PREFIX" --crtr-root "$ROOT/crtr" --steps 40 --bs 128 --evalevery 20 \
  --device cuda --seed 777
for MODE in greedy search; do
  "$PY" -u sokoban_solve.py --crtr-root "$ROOT/crtr" --checkpoint "$PREFIX.pt" \
    --kind "$SKIND" --mode "$MODE" --n-jobs 2 --max-tree-size 64 \
    --output-dir "${PREFIX}_${MODE}"
done
printf 'SMOKE_DONE %s GPU=%s\n' "$KIND" "$GPU_UUID"
