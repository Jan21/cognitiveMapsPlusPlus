#!/bin/bash
#SBATCH --job-name=slotAttn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-15%16
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/slotAttn_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# COMPETITIVE ITERATIVE SLOT ATTENTION (softmax over slots per pixel, GRU update, k iters) on the pure-image path.
# Refs @L2/200 maps/40k: one-shot xattn slots k1 = 0.896/0.850 (T1), 0.896/0.741 (T4); iqe 0.85; factored 0.923.
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --readout slotattn"
L2="--wire1 --noplate"; L1h="--nlever 0 --noplate"; L3="--noplate"
A=(
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --tag sa_L2_i3s8_k1_T4_s0"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 1 --tag sa_L2_i3s8_k1_T4_s1"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --T 1 --nobaseline --lr 2e-3 --seed 0 --tag sa_L2_i3s8_k1_T1_s0"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 1 --T 4 --nobaseline --lr 2e-3 --seed 0 --tag sa_L2_i1s8_k1_T4_s0"
 "$B $L2 --cnnk 1 --slots 12 --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --tag sa_L2_i3s12_k1_T4_s0"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --recon slot --tag sa_L2_i3s8_k1_T4_recon_s0"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --slotnoise 1 --tag sa_L2_i3s8_k1_T4_noise_s0"
 "$B $L2 --cnnk 3 --slots 8  --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --tag sa_L2_i3s8_k3_T4_s0"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --seed 0 --tag sa_L2_i3s8_k1_iqe_s0"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --seed 0 --tag sa_L2_i3s8_k1_scalar_s0"
 "$B $L2 --cnnk 1 --slots 8  --slotiters 3 --symonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --seed 0 --tag sa_L2_i3s8_k1_sym_s0"
 "$B $L1h --cnnk 1 --slots 8 --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --tag sa_L1h_i3s8_k1_T4_s0"
 "$B $L1h --cnnk 3 --slots 8 --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --tag sa_L1h_i3s8_k3_T4_s0"
 "$B $L3 --cnnk 1 --slots 8  --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 0 --tag sa_L3_i3s8_k1_T4_s0"
 "$B $L3 --cnnk 1 --slots 8  --slotiters 3 --T 4 --nobaseline --lr 2e-3 --seed 1 --tag sa_L3_i3s8_k1_T4_s1"
 "$B $L3 --cnnk 1 --slots 8  --slotiters 3 --iqeonly --baselayers 4 --lr 1e-3 --gradclip 1.0 --seed 0 --tag sa_L3_i3s8_k1_iqe_s0"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
