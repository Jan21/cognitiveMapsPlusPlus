#!/bin/bash
set -euo pipefail
# GPU smoke only: verifies full trainer execution, memory, throughput, artifacts.
# These 200-step results are never candidates or comparable baseline evidence.
ROOT=/home/hulajan1/swbench/ar_20260905_direct
GPU_UUID="$1"
ARM="$2"
exec 9>"$ROOT/smoke-${GPU_UUID}.lock"
flock -n 9
case "$GPU_UUID" in GPU-*) ;; *) exit 2;; esac
CARD=$(nvidia-smi -i "$GPU_UUID" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits)
case "$CARD" in *Display*|*1080*) exit 3;; esac
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ "$APPS" == *"$GPU_UUID"* ]]; then exit 4; fi
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$ROOT/src"
A=(--train --enc pureimage --G 11 --ngate 3 --nlever 2 --nchute 1
   --nmaps 32 --poolq 12 --Rmax 36 --bfsmax 38360 --steps 200 --bs 128
   --evalevery 100 --evalbs 128 --nobaseline --gradclip 1 --warmup 20
   --cnnw 64 --cnnk 3 --cnndepth 3 --d 128 --heads 4 --layers 2 --T 4
   --lr 1e-3 --seed 0 --split map --tag "smoke_${ARM}"
   --save "$ROOT/artifacts/smoke_${ARM}" --dumppred "$ROOT/artifacts/smoke_${ARM}")
case "$ARM" in
  joint) A+=(--research-model joint --T 8);;
  context) A+=(--research-model context);;
  coat) A+=(--extonly coat --extw 64);;
  pixels) A+=(--readout pixels --layers 1);;
  *) exit 5;;
esac
python3 autoresearch_trial.py --pool-cache "$ROOT/pools" --eval-bank validation "${A[@]}"
printf 'SMOKE_DONE %s\n' "$ARM"
