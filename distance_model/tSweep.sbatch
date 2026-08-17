#!/bin/bash
#SBATCH --job-name=tSweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-9%8
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/tSweep_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# Probe: does the recall depth T matter on the image in-range structural task? integ was never run with T!=14.
# integ best image config (L2 d256), vary T in {2,4,8,14,20}, both splits, seed 0. Baseline = T14 (0.788/0.786).
BASE="python3 switchyard.py --train --enc image --readout xattn --cnnw 64 --cnndepth 2 --heads 4 --nobaseline --lr 1e-3 --layers 2 --d 256 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --latentnorm 0"
A=()
for T in 2 4 8 14 20; do
  A+=("$BASE --T $T --split map  --tag t_${T}_map")
  A+=("$BASE --T $T --split wire --tag t_${T}_wire")
done
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
