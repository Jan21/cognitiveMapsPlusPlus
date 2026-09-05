#!/bin/bash
#SBATCH --job-name=crtr-extra
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/hulajan1/swbench/crtr_extra_20260905/logs/worker_%A_%a.log
# Submit with explicit typed --gres, --nodelist and --array; never bare GPU indices.
set -euo pipefail
BENCH_ROOT=/home/hulajan1/swbench/crtr_extra_20260905
REFERENCE_ROOT=/home/hulajan1/swbench/sokoban_ar_20260905
PY="$REFERENCE_ROOT/venv/bin/python"
ENV=${1:?requires lights or digit}
PHASE=${2:-full}
[[ "$ENV" == lights || "$ENV" == digit ]]
[[ "$PHASE" == smoke || "$PHASE" == full ]]
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$BENCH_ROOT/src:$REFERENCE_ROOT/crtr"
cd "$BENCH_ROOT/src"
sha256sum --check SHA256SUMS
test -f "$BENCH_ROOT/banks/$ENV/manifest.json"
GPU_UUID=$("$PY" -c 'import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_properties(0).uuid)')
GPU_UUID="GPU-${GPU_UUID#GPU-}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
exec 9>"$BENCH_ROOT/gpu-$GPU_UUID.lock"
flock -n 9
sleep 2
for attempt in {1..10}; do
  if "$PY" -c 'import sys; from autoresearch_loop import gpu_idle; ok,why=gpu_idle(sys.argv[1]); print(why,flush=True); sys.exit(0 if ok else 1)' "$GPU_UUID"; then
    break
  fi
  sleep 10
done
"$PY" -c 'import sys; from autoresearch_loop import gpu_idle; ok,why=gpu_idle(sys.argv[1]); print(why,flush=True); sys.exit(0 if ok else 1)' "$GPU_UUID"
printf 'ALLOCATED_GPU %s job=%s task=%s env=%s phase=%s\n' "$GPU_UUID" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$ENV" "$PHASE"
KINDS=(joint supervised joint supervised dense dense)
SEEDS=(0 0 1 1 0 1)
TASK=${SLURM_ARRAY_TASK_ID}
KIND=${KINDS[$TASK]}
SEED=${SEEDS[$TASK]}
[[ "$ENV" == lights || "$TASK" -lt 4 ]]
BANK="$BENCH_ROOT/banks/$ENV"
NAME="${ENV}_${KIND}_exact_s${SEED}"
STEPS=80000
EVALEVERY=10000
EVALBS=128
TEST="$BANK/test.npz"
CASES="$BANK/cases.npz"
NJOBS=1000
CAP=6000
if [[ "$PHASE" == smoke ]]; then
  NAME="smoke_${SLURM_ARRAY_JOB_ID}_${NAME}"
  STEPS=40
  EVALEVERY=0
  TEST="$BANK/smoke_test.npz"
  CASES="$BANK/smoke_cases.npz"
  NJOBS=2
  CAP=64
fi
PREFIX="$BENCH_ROOT/artifacts/$NAME"
timeout --signal=TERM --kill-after=30 9h "$PY" -u crtr_extra_train.py \
  --env "$ENV" --model "$KIND" --targets exact --train-bank "$BANK/train.npz" \
  --val-bank "$BANK/validation.npz" --test-bank "$TEST" --out "$PREFIX" \
  --crtr-root "$REFERENCE_ROOT/crtr" --steps "$STEPS" --bs 128 --eval-bs "$EVALBS" \
  --evalevery "$EVALEVERY" --seed "$SEED" --device cuda
test -f "$PREFIX.json"
printf 'TRAIN_DONE %s\n' "$NAME"
FAILED=0
for MODE in greedy search; do
  LIMIT=13h
  if [[ "$MODE" == greedy ]]; then LIMIT=1h; fi
  if timeout --signal=TERM --kill-after=30 "$LIMIT" "$PY" -u crtr_extra_solve.py \
    --env "$ENV" --crtr-root "$REFERENCE_ROOT/crtr" --checkpoint "$PREFIX.pt" \
    --cases "$CASES" --mode "$MODE" --n-jobs "$NJOBS" --max-tree-size "$CAP" \
    --out "${PREFIX}_${MODE}" --device cuda \
    > "$BENCH_ROOT/logs/${NAME}_${MODE}.log" 2>&1; then
    printf 'SOLVE_DONE %s %s\n' "$NAME" "$MODE"
  else
    CODE=$?
    printf 'SOLVE_INCOMPLETE %s %s exit=%s\n' "$NAME" "$MODE" "$CODE"
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then exit 1; fi
printf 'WORKER_DONE %s\n' "$NAME"
