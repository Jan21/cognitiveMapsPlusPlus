#!/bin/bash
#SBATCH --job-name=extra-bank
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/hulajan1/swbench/crtr_extra_20260905/logs/prepare_%A_%a.log
set -euo pipefail
BENCH_ROOT=/home/hulajan1/swbench/crtr_extra_20260905
PY=/home/hulajan1/swbench/sokoban_ar_20260905/venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$BENCH_ROOT/prep_src"
sha256sum --check SHA256SUMS
ENVS=(lights digit)
ENV=${ENVS[$SLURM_ARRAY_TASK_ID]}
"$PY" -u crtr_extra_prepare.py --env "$ENV" --out "$BENCH_ROOT/banks/$ENV"
test -f "$BENCH_ROOT/banks/$ENV/manifest.json"
printf 'PREPARE_DONE %s\n' "$ENV"
