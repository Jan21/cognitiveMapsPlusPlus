#!/bin/bash
#SBATCH --job-name=baseFull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-15%8
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/baseFull_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# PHASE B1: full 80k, image structural, split map, seed 0. latentnorm OFF (screen: it hurts metric heads),
# gradclip + nan-guard on. Per-head capacity tune: depth {2,4} x width d {128,256}. lr from screen
# (metric heads 1e-3, mrn 2e-3). Same encoder/data/loss/steps for every head. Winners -> B2 (wire split + seeds).
BASE="python3 switchyard.py --train --enc image --readout xattn --cnnw 64 --cnndepth 2 --heads 4 --T 14 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --latentnorm 0 --split map"
A=()
for depth in 2 4; do
  for dd in 128 256; do
    A+=("$BASE --lr 1e-3 --d $dd --layers $depth --baselayers $depth --tag full_intsc_L${depth}_d${dd}")   # integ + scalar
    A+=("$BASE --symonly --lr 1e-3 --d $dd --baselayers $depth --tag full_sym_L${depth}_d${dd}")
    A+=("$BASE --iqeonly --lr 1e-3 --d $dd --baselayers $depth --tag full_iqe_L${depth}_d${dd}")
    A+=("$BASE --mrnonly --lr 2e-3 --d $dd --baselayers $depth --tag full_mrn_L${depth}_d${dd}")
  done
done
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
