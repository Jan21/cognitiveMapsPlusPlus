#!/bin/bash
#SBATCH --job-name=pureFix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-9%10
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureFix_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# Two interventions on the pure-image L2/L3 wall (ladder refs @200 maps/40k: L2 integ T4 0.762, L3 0.671; factored L2 0.923):
#  --recon slot : K slots reconstruct ALL entity channels exclusively (unsupervised many-entity binding)
#  --wirepath 1 : draw lever->gate wiring as an L-path in the lever channel (spatially traceable pairing)
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --seed 0 --slots 8"
A=()
for r in "--wire1 --noplate|L2" "|L3"; do f="${r%%|*}"; n="${r##*|}"
  A+=("$P $f --readout xattn --T 4 --nobaseline --lr 2e-3 --recon slot               --tag pf_${n}_integ_T4_recon")
  A+=("$P $f --readout xattn --T 4 --nobaseline --lr 2e-3 --wirepath 1               --tag pf_${n}_integ_T4_wire")
  A+=("$P $f --readout xattn --T 4 --nobaseline --lr 2e-3 --recon slot --wirepath 1  --tag pf_${n}_integ_T4_reconwire")
  A+=("$P $f --readout pixels --T 4 --nobaseline --lr 2e-3 --wirepath 1              --tag pf_${n}_integ_pix_T4_wire")
  A+=("$P $f --readout xattn --symonly --baselayers 4 --lr 1e-3 --gradclip 1.0 --recon slot --tag pf_${n}_sym_recon")
done
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
