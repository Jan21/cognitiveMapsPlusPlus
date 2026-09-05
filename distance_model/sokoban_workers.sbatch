#!/bin/bash
#SBATCH --job-name=sok-joint
#SBATCH --partition=gpu
#SBATCH --nodelist=dgx-2
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/hulajan1/swbench/sokoban_ar_20260905/logs/worker_%A_%a.log
set -euo pipefail
ROOT=/home/hulajan1/swbench/sokoban_ar_20260905
PY="$ROOT/venv/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Do not inherit unrelated research projects via PYTHONPATH.
export PYTHONPATH="$ROOT/src:$ROOT/crtr"
cd "$ROOT/src"
sha256sum --check "$ROOT/src/SHA256SUMS"
test -f "$ROOT/banks/READY"
GPU_UUID=$("$PY" -c 'import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_properties(0).uuid)')
GPU_UUID="GPU-${GPU_UUID#GPU-}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
exec 9>"$ROOT/gpu-$GPU_UUID.lock"
flock -n 9
sleep 2
for attempt in {1..10}; do
  if "$PY" -c 'import sys; from autoresearch_loop import gpu_idle; ok, why=gpu_idle(sys.argv[1]); print(why,flush=True); sys.exit(0 if ok else 1)' "$GPU_UUID"; then
    break
  fi
  sleep 10
done
"$PY" -c 'import sys; from autoresearch_loop import gpu_idle; ok,why=gpu_idle(sys.argv[1]); print(why,flush=True); sys.exit(0 if ok else 1)' "$GPU_UUID"
printf 'ALLOCATED_GPU %s job=%s task=%s\n' "$GPU_UUID" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID"
NAMES=(joint_bfs_s0 supervised_bfs_s0 joint_bfs_s1 supervised_bfs_s1 joint_gap_s0 supervised_gap_s0 joint_noattn_bfs_s0 supervised_shipped)
KINDS=(joint supervised joint supervised joint supervised joint supervised)
SEEDS=(0 0 1 1 0 0 0 0)
TASK=${SLURM_ARRAY_TASK_ID}
NAME=${NAMES[$TASK]}
KIND=${KINDS[$TASK]}
PREFIX="$ROOT/artifacts/$NAME"
TARGETS=bfs
TRAIN="$ROOT/banks/train.npz"
VAL="$ROOT/banks/val.npz"
if [[ "$TASK" == 4 || "$TASK" == 5 ]]; then
  TARGETS=gap
  TRAIN="$ROOT/banks/train_gap.npz"
  VAL="$ROOT/banks/val_gap.npz"
fi
ARGS=(--model "$KIND" --val-bank "$VAL" --test-bank "$ROOT/banks/test.npz"
  --out "$PREFIX" --crtr-root "$ROOT/crtr" --targets "$TARGETS" --device cuda)
if [[ "$TASK" == 7 ]]; then
  CKPT="$ROOT/crtr/example_checkpoints/sokoban/supervised.pt"
  timeout --signal=TERM --kill-after=30 1h "$PY" -u sokoban_joint.py "${ARGS[@]}" \
    --eval-only --checkpoint "$CKPT"
  SOLVE_KIND=supervised
else
  EXTRA=()
  if [[ "$TASK" == 6 ]]; then EXTRA=(--attention-heads 0); fi
  timeout --signal=TERM --kill-after=30 8h "$PY" -u sokoban_joint.py "${ARGS[@]}" \
    --train-bank "$TRAIN" --steps 80000 --bs 128 --seed "${SEEDS[$TASK]}" \
    --lr .001 --warmup 2000 --gradclip 1 --evalevery 10000 "${EXTRA[@]}"
  CKPT="$PREFIX.pt"
  SOLVE_KIND="$KIND"
  if [[ "$KIND" == supervised ]]; then SOLVE_KIND=supervised-ours; fi
fi
test -f "$PREFIX.json"
printf 'TRAIN_OR_EVAL_DONE %s\n' "$NAME"
for MODE in greedy search; do
  LIMIT=13h
  if [[ "$MODE" == greedy ]]; then LIMIT=1h; fi
  timeout --signal=TERM --kill-after=30 "$LIMIT" "$PY" -u sokoban_solve.py \
    --crtr-root "$ROOT/crtr" --checkpoint "$CKPT" --kind "$SOLVE_KIND" \
    --mode "$MODE" --n-jobs 1000 --output-dir "$ROOT/artifacts/${NAME}_${MODE}" \
    > "$ROOT/logs/${NAME}_${MODE}.log" 2>&1
  printf 'SOLVE_DONE %s %s\n' "$NAME" "$MODE"
done
printf 'WORKER_DONE %s\n' "$NAME"
