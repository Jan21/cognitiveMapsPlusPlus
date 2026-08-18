#!/bin/bash
#SBATCH --job-name=slotDiag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-5%6
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/slotDiag_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# What do the slots DO? per-slot attention entropy + which entity type each slot lands on (test maps), 3x3 vs 1x1, L1.5 / L2 / L3.
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --readout xattn --slots 8 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --seed 0 --T 4 --nobaseline --lr 2e-3"
A=(
 "$B --cnnk 3 --nlever 0 --noplate --tag sd_L1h_k3" "$B --cnnk 1 --nlever 0 --noplate --tag sd_L1h_k1"
 "$B --cnnk 3 --wire1 --noplate    --tag sd_L2_k3"  "$B --cnnk 1 --wire1 --noplate    --tag sd_L2_k1"
 "$B --cnnk 3 --noplate            --tag sd_L3_k3"  "$B --cnnk 1 --noplate            --tag sd_L3_k1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
