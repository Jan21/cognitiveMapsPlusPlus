#!/bin/bash
#SBATCH --job-name=cwImg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-58%12
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/cwImg_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
export PYTORCH_ALLOC_CONF=expandable_segments:True
# CRATEWORLD, IMAGE (CNN + xattn, switchyard-aligned) held-out CONFIG generalization: does the integrator's edge
# on unseen coupling structures (combo / links2 / dofhi) survive a proper image encoder + fairly tuned baselines?
# integ at its own recipe (run_best.sbatch: lr 2e-3, L4 d128 T14, no clip) x 3 splits x 2 seeds.
# baselines iqe / mrn / sym_attn / scalar x depth {2,4} x lr {1e-3, 2e-3} x 3 splits, seed 0. Same encoder/data/loss/80k.
B="python3 integ_distance.py --nag 4 --G 6 --Rmax 12 --T 14 --d 128 --heads 4 --poolq 2000 --nquery 80 --inject 1 --enc image --cnnw 64 --cnndepth 2 --steps 80000"
A=()
for ho in combo links2 dofhi; do
  for sd in 0 1; do A+=("$B --arch integ --layers 4 --lr 2e-3 --heldout $ho --seed $sd"); done
done
for arch in iqe mrn sym_attn scalar; do
  for L in 2 4; do for lr in 1e-3 2e-3; do for ho in combo links2 dofhi; do
    A+=("$B --arch $arch --layers $L --lr $lr --heldout $ho --seed 0")
  done; done; done
done
# switchyard fair-table completion: integ at its OWN recipe (verbatim original) wire seed 2, and T20 seeds 1,2
S="python3 switchyard.py --train --d 128 --layers 4 --T 14 --nmaps 200 --poolq 2000 --steps 80000 --enc image --nobaseline --readout xattn"
A+=("$S --split wire --seed 2 --tag rep_orig_wire_s2")
A+=("$S --T 20 --split map  --seed 1 --tag rep_T20_map_s1")
A+=("$S --T 20 --split wire --seed 1 --tag rep_T20_wire_s1")
A+=("$S --T 20 --split map  --seed 2 --tag rep_T20_map_s2")
A+=("$S --T 20 --split wire --seed 2 --tag rep_T20_wire_s2")
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
