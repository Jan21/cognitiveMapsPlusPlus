#!/bin/bash
#SBATCH --job-name=baseSeeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-24%8
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/baseSeeds_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# PHASE B2: each head at its B1-best capacity (best-recipe-per-model), error bars on BOTH axes.
# map already has seed0 from B1 -> add seeds 1,2. wire is fresh -> seeds 0,1,2. 5 heads x 5 runs = 25.
BASE="python3 switchyard.py --train --enc image --readout xattn --cnnw 64 --cnndepth 2 --heads 4 --T 14 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --latentnorm 0"
HEAD=(
 "--nobaseline --lr 1e-3 --layers 2 --d 256|integ"      # B1 best: L2 d256
 "--scalaronly --lr 1e-3 --baselayers 2 --d 128|scalar" # B1 best: L2 d128
 "--symonly    --lr 1e-3 --baselayers 2 --d 128|sym"    # B1 best: L2 d128
 "--iqeonly    --lr 1e-3 --baselayers 4 --d 128|iqe"    # B1 best: L4 d128
 "--mrnonly    --lr 2e-3 --baselayers 2 --d 128|mrn"    # B1 best: L2 d128 (d256 diverges)
)
A=()
for hc in "${HEAD[@]}"; do
  cfg="${hc%%|*}"; nm="${hc##*|}"
  for sd in 1 2;   do A+=("$BASE $cfg --split map  --seed $sd --tag b2_${nm}_map_s${sd}"); done
  for sd in 0 1 2; do A+=("$BASE $cfg --split wire --seed $sd --tag b2_${nm}_wire_s${sd}"); done
done
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
