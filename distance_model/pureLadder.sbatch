#!/bin/bash
#SBATCH --job-name=pureLadder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-39%12
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureLadder_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# PURE-IMAGE screen: input = rendered start + goal images ONLY (walls, entities, gate bits, wiring, chute dir all as pixels).
# Encoder -> K slots (or all pixels) -> integrator / baselines. No symbolic tokens anywhere. Complexity ladder (unseen maps):
#  L0 gates open + crate static | L1 gates open | L2 wire1 + no plate | L3 full switchyard.
# Variants: recall on/off, T 4 vs 14, 8 vs 16 slots, slots vs pixel tokens; baselines on the SAME slots; factored integ = ceiling ref.
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --seed 0"
R=("--gatesopen --nopush|L0" "--gatesopen|L1" "--wire1 --noplate|L2" "|L3")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"
  A+=("$P $f --readout xattn --slots 8  --T 14 --nobaseline --lr 2e-3 --tag pl_${n}_integ_s8_T14")
  A+=("$P $f --readout xattn --slots 8  --T 14 --nobaseline --lr 2e-3 --norecall 1 --tag pl_${n}_integ_s8_T14_norecall")
  A+=("$P $f --readout xattn --slots 8  --T 4  --nobaseline --lr 2e-3 --tag pl_${n}_integ_s8_T4")
  A+=("$P $f --readout xattn --slots 16 --T 14 --nobaseline --lr 2e-3 --tag pl_${n}_integ_s16_T14")
  A+=("$P $f --readout pixels           --T 14 --nobaseline --lr 2e-3 --tag pl_${n}_integ_pix_T14")
  A+=("$P $f --readout xattn --slots 8 --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag pl_${n}_iqe_s8")
  A+=("$P $f --readout xattn --slots 8 --mrnonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag pl_${n}_mrn_s8")
  A+=("$P $f --readout xattn --slots 8 --symonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag pl_${n}_sym_s8")
  A+=("$P $f --readout xattn --slots 8 --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag pl_${n}_scalar_s8")
  A+=("python3 switchyard.py --train --enc factored --d 128 --layers 4 --heads 4 --T 14 --nmaps 200 --poolq 2000 --steps 40000 --split map --seed 0 $f --nobaseline --lr 2e-3 --tag pl_${n}_integ_FACTORED_ref")
done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
