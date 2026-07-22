#!/bin/bash
#SBATCH --job-name=f1d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/f1d_%a.out
#SBATCH --array=0-3
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
CFG=$SLURM_ARRAY_TASK_ID
case $CFG in
  0) ARGS="--N 5 --G 48 --steps 20000 --seed 0 --factored" ;;
  1) ARGS="--N 5 --G 48 --steps 20000 --seed 1 --factored" ;;
  2) ARGS="--N 3 --G 32 --steps 20000 --seed 0 --factored" ;;
  3) ARGS="--N 5 --G 48 --steps 20000 --seed 0 --factored --keygate" ;;
esac
echo "=== CFG $CFG: $ARGS ==="
python mini20_image1d.py $ARGS
echo "=== DONE $CFG ==="
