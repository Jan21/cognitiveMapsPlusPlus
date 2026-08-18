#!/bin/bash
#SBATCH --job-name=pureCeil
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3%4
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureCeil_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# CEILING probe: pure-image integ at L2/L3 with Hungarian-matched SUPERVISED binding (slots <-> true entity cells).
# If this reaches ~0.9 at L2, binding is THE bottleneck (recon-slot is the right lever); if ~0.75, the CNN features are.
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --slots 8 --readout xattn --T 4 --nobaseline --lr 2e-3 --supbind 1.0"
A=( "$P --wire1 --noplate --seed 0 --tag pc_L2_sup_s0" "$P --wire1 --noplate --seed 1 --tag pc_L2_sup_s1" "$P --seed 0 --tag pc_L3_sup_s0" "$P --seed 1 --tag pc_L3_sup_s1" )
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
