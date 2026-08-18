#!/bin/bash
#SBATCH --job-name=bindPure
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-19%16
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/bindPure_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# BINDING on the current best pure-image model (1x1 one-shot slots, safe recipe), where the gap to symbolic remains (L3 ~0.12, L5 ~0.15):
# ceiling = Hungarian-supervised binding; unsupervised = recon-slot aux, attention-entropy sharpening, slots >= entities (12/16). 80k.
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --readout xattn --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --split map --gradclip 1.0 --warmup 2000 --lr 1e-3 --nobaseline --T 4"
A=()
for r in "--noplate|L3" "|L5"; do f="${r%%|*}"; n="${r##*|}"
  A+=("$B $f --slots 8  --seed 0 --supbind 1.0        --tag bp_${n}_sup_s8_s0")
  A+=("$B $f --slots 12 --seed 0 --supbind 1.0        --tag bp_${n}_sup_s12_s0")
  A+=("$B $f --slots 8  --seed 0 --recon slot         --tag bp_${n}_recon_s8_s0")
  A+=("$B $f --slots 12 --seed 0 --recon slot         --tag bp_${n}_recon_s12_s0")
  A+=("$B $f --slots 8  --seed 0 --attnent 0.05       --tag bp_${n}_ent05_s8_s0")
  A+=("$B $f --slots 8  --seed 0 --attnent 0.2        --tag bp_${n}_ent2_s8_s0")
  A+=("$B $f --slots 12 --seed 0                      --tag bp_${n}_plain_s12_s0")
  A+=("$B $f --slots 16 --seed 0                      --tag bp_${n}_plain_s16_s0")
  A+=("$B $f --slots 12 --seed 0 --recon slot --attnent 0.05 --tag bp_${n}_reconent_s12_s0")
  A+=("$B $f --slots 12 --seed 1 --recon slot         --tag bp_${n}_recon_s12_s1")
done
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
