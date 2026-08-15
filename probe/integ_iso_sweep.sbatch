#!/bin/bash
#SBATCH --job-name=igIso
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igIso_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
B="--keymech control --topo grid --norm pre --lr 2e-3 --pbank 0.4 --rvgt 3 --iso 2 --isoscales 3 --steps 40000 --nquery 70 --nag 4"
C=$SLURM_ARRAY_TASK_ID
case $C in
  # image-encoder confirm at G6 (does iso transfer off factored?)
  0) A="--enc hgather --G 6 --Rmax 12 --T 18 --seed 0" ;;
  1) A="--enc hgather --G 6 --Rmax 12 --T 18 --seed 1" ;;
  2) A="--enc hgather --G 6 --Rmax 12 --T 18 --seed 2" ;;
  3) A="--enc percomp --G 6 --Rmax 12 --T 18 --seed 0" ;;
  4) A="--enc percomp --G 6 --Rmax 12 --T 18 --seed 1" ;;
  5) A="--enc percomp --G 6 --Rmax 12 --T 18 --seed 2" ;;
  # bigger grids, factored
  6) A="--enc factored --G 8  --Rmax 16 --T 22 --seed 0" ;;
  7) A="--enc factored --G 8  --Rmax 16 --T 22 --seed 1" ;;
  8) A="--enc factored --G 10 --Rmax 20 --T 26 --seed 0" ;;
  9) A="--enc factored --G 10 --Rmax 20 --T 26 --seed 1" ;;
  # bigger grid + image encoder
  10) A="--enc hgather --G 8 --Rmax 16 --T 22 --seed 0" ;;
  11) A="--enc hgather --G 8 --Rmax 16 --T 22 --seed 1" ;;
esac
echo "CFG=$C ARGS=$A"; python3 integ_image.py $B $A
