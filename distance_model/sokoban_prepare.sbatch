#!/bin/bash
#SBATCH --job-name=sok-bank
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/hulajan1/swbench/sokoban_ar_20260905/logs/prepare_%j.log
set -euo pipefail
ROOT=/home/hulajan1/swbench/sokoban_ar_20260905
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$ROOT/venv/bin/python" -u "$ROOT/src/sokoban_prepare.py" --root "$ROOT"
test -f "$ROOT/banks/READY"
printf 'PREPARE_DONE\n'
