#!/bin/bash
#SBATCH --job-name=bindUnsup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-23%8
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/bindUnsup_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# UNSUPERVISED BINDING probe (switchyard, CNN+xattn, integ at its own recipe). Supervised cell aux gave +0.10; can we get it
# from the model's OWN INPUT? recon tied (slot i decodes its entity channel), recon slot (slots jointly+exclusively explain
# the entity mask, no identity given), hard straight-through attention, slot+hard. References: none, supervised markeraux.
B="python3 switchyard.py --train --d 128 --layers 4 --T 14 --nmaps 200 --poolq 2000 --steps 80000 --enc image --nobaseline --readout xattn"
V=("--recon none|none" "--recon tied|tied" "--recon slot|slot" "--hardattn 1|hard" "--recon slot --hardattn 1|slothard" "--markeraux 1.0|sup")
A=()
for v in "${V[@]}"; do f="${v%%|*}"; n="${v##*|}"; for sp in map wire; do for sd in 0 1; do
  A+=("$B $f --split $sp --seed $sd --tag bu_${n}_${sp}_s${sd}")
done; done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
