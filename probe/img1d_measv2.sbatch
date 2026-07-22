#!/bin/bash
#SBATCH --job-name=m1dv2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/m1dv2_%a.out
#SBATCH --array=0-7
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
BASE="--N 5 --G 48 --steps 20000 --seed 0"
CFG=$SLURM_ARRAY_TASK_ID
case $CFG in
  0) ARGS="$BASE --factored --W 16 --L 1500 --qlo 0.05 --qhi 0.6" ;;                 # baseline reproduce
  1) ARGS="$BASE --factored --W 24 --L 4000 --M 150000 --qlo 0.02 --qhi 0.75" ;;     # denser+wider band
  2) ARGS="$BASE --factored --W 32 --L 6000 --M 200000 --qlo 0.02 --qhi 0.80" ;;     # denser++
  3) ARGS="$BASE            --W 24 --L 4000 --M 150000 --qlo 0.02 --qhi 0.75" ;;      # image, denser+wider
  4) ARGS="$BASE            --W 32 --L 6000 --M 200000 --qlo 0.02 --qhi 0.80" ;;      # image, denser++
  5) ARGS="$BASE --factored --W 16 --L 4000 --M 150000 --qlo 0.02 --qhi 0.75" ;;     # band only (W=16)
  6) ARGS="$BASE --factored --keygate --W 24 --L 4000 --M 150000 --qlo 0.02 --qhi 0.75" ;;
  7) ARGS="$BASE            --keygate --W 24 --L 4000 --M 150000 --qlo 0.02 --qhi 0.75" ;;
esac
echo "=== CFG $CFG: $ARGS ==="
python mini20_image1d.py $ARGS
echo "=== DONE $CFG ==="
