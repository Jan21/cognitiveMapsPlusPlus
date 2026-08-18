#!/bin/bash
#SBATCH --job-name=pureLadK1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-71%14
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureLadK1_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# PURE-IMAGE LADDER, FIXED PERCEPTION: --cnnk 1 (per-pixel MLP, no spatial blur; still image-only). All six rungs,
# integ (slots8 T4, and T1 / T14 variants) vs iqe / mrn / sym / scalar on the same slots, 2 seeds, 40k, unseen maps.
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --readout xattn --slots 8"
R=("--gatesopen --nopush|L0" "--gatesopen|L1" "--wire1 --noplate|L2" "--noplate|L3" "--nchute 0|L4" "|L5")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"; for sd in 0 1; do
  A+=("$P $f --seed $sd --T 4  --nobaseline --lr 2e-3 --tag k1l_${n}_integ_T4_s${sd}")
  A+=("$P $f --seed $sd --T 1  --nobaseline --lr 2e-3 --tag k1l_${n}_integ_T1_s${sd}")
  A+=("$P $f --seed $sd --T 14 --nobaseline --lr 2e-3 --tag k1l_${n}_integ_T14_s${sd}")
  A+=("$P $f --seed $sd --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag k1l_${n}_iqe_s${sd}")
  A+=("$P $f --seed $sd --symonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag k1l_${n}_sym_s${sd}")
  A+=("$P $f --seed $sd --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag k1l_${n}_scalar_s${sd}")
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
