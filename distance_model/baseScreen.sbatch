#!/bin/bash
#SBATCH --job-name=baseScreen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-23%8
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/baseScreen_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# PHASE A screen: fair baseline comparison on the image structural setup (map split, 8k steps).
# Same encoder (image CNN+xattn, cnnw64/depth2), same data/loss/steps for every head; only head + {lr, latentnorm} vary.
# Purpose: (1) does latentnorm fix the MRN nan? (2) which lr each head needs. Ranks configs -> Phase B full-length.
B="python3 switchyard.py --train --enc image --readout xattn --cnnw 64 --cnndepth 2 --d 128 --heads 4 --T 14 --nmaps 200 --poolq 2000 --steps 8000 --gradclip 1.0 --split map"
A=()
for lr in 2e-3 1e-3 3e-4; do
  for ln in 0 1; do
    A+=("$B --lr $lr --latentnorm $ln --tag scr_intsc_lr${lr}_ln${ln}")   # integ + scalar (one run trains both)
    A+=("$B --symonly --lr $lr --latentnorm $ln --tag scr_sym_lr${lr}_ln${ln}")
    A+=("$B --iqeonly --lr $lr --latentnorm $ln --tag scr_iqe_lr${lr}_ln${ln}")
    A+=("$B --mrnonly --lr $lr --latentnorm $ln --tag scr_mrn_lr${lr}_ln${ln}")
  done
done
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
