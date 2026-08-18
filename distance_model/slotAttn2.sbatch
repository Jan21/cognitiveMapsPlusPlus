#!/bin/bash
#SBATCH --job-name=slotAttn2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-11%12
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/slotAttn2_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# slot attention v2 for integ: it COLLAPSED at lr 2e-3 / no clip (7 slots one-hot on background). Recipe fix: gradclip 1 + lower lr + warmup.
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --readout slotattn --slots 8 --slotiters 3 --nobaseline --T 4 --gradclip 1.0"
L2="--wire1 --noplate"; L1h="--nlever 0 --noplate"; L3="--noplate"
A=(
 "$B $L2 --lr 1e-3 --seed 0 --tag sa2_L2_lr1e3_s0"       "$B $L2 --lr 1e-3 --seed 1 --tag sa2_L2_lr1e3_s1"
 "$B $L2 --lr 5e-4 --seed 0 --tag sa2_L2_lr5e4_s0"       "$B $L2 --lr 5e-4 --seed 1 --tag sa2_L2_lr5e4_s1"
 "$B $L2 --lr 1e-3 --warmup 2000 --seed 0 --tag sa2_L2_lr1e3_wu_s0"  "$B $L2 --lr 1e-3 --warmup 2000 --seed 1 --tag sa2_L2_lr1e3_wu_s1"
 "$B $L2 --lr 1e-3 --warmup 2000 --slotiters 1 --seed 0 --tag sa2_L2_lr1e3_wu_i1_s0"
 "$B $L2 --lr 1e-3 --warmup 2000 --recon slot --seed 0 --tag sa2_L2_lr1e3_wu_recon_s0"
 "$B $L2 --lr 1e-3 --warmup 2000 --T 1 --seed 0 --tag sa2_L2_lr1e3_wu_T1_s0"
 "$B $L1h --lr 1e-3 --warmup 2000 --seed 0 --tag sa2_L1h_lr1e3_wu_s0"
 "$B $L3 --lr 1e-3 --warmup 2000 --seed 0 --tag sa2_L3_lr1e3_wu_s0"
 "$B $L2 --lr 1e-3 --warmup 2000 --cnnk 3 --seed 0 --tag sa2_L2_lr1e3_wu_k3_s0"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
