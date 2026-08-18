#!/bin/bash
#SBATCH --job-name=pureDesign
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-23%8
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureDesign_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# DESIGN questions on the rung where pure image trains (L1 = gates open, crate pushable, unseen maps):
# how many recall steps T? re-inject goal/start or not? how many slots? 2 seeds each.
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --gatesopen --readout xattn --nobaseline --lr 2e-3"
A=()
for sd in 0 1; do for T in 1 2 4 8 14; do for rc in 0 1; do A+=("$P --slots 8 --T $T --norecall $rc --seed $sd --tag pd_L1_T${T}_rc${rc}_s${sd}"); done; done; done
for sd in 0 1; do for K in 4 16; do A+=("$P --slots $K --T 4 --seed $sd --tag pd_L1_s${K}_T4_s${sd}"); done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
