#!/bin/bash
#SBATCH --job-name=igSub8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-23%24
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub8_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# ROBUSTNESS: firm the headline (supervised head corr_head + head_mae) to n5-7 across all held-out regimes. add vs submodx, seeds 4-6.
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --dofsup 1.0 --steps 20000 --inject 1 --d 128 --layers 4"
A=(
 "--gatekey add     --heldout combo  --seed 4"
 "--gatekey add     --heldout combo  --seed 5"
 "--gatekey add     --heldout combo  --seed 6"
 "--gatekey add     --heldout links2 --seed 4"
 "--gatekey add     --heldout links2 --seed 5"
 "--gatekey add     --heldout links2 --seed 6"
 "--gatekey add     --heldout dofhi  --seed 4"
 "--gatekey add     --heldout dofhi  --seed 5"
 "--gatekey add     --heldout dofhi  --seed 6"
 "--gatekey add     --heldout dofhi2 --seed 4"
 "--gatekey add     --heldout dofhi2 --seed 5"
 "--gatekey add     --heldout dofhi2 --seed 6"
 "--gatekey submodx --heldout combo  --seed 4"
 "--gatekey submodx --heldout combo  --seed 5"
 "--gatekey submodx --heldout combo  --seed 6"
 "--gatekey submodx --heldout links2 --seed 4"
 "--gatekey submodx --heldout links2 --seed 5"
 "--gatekey submodx --heldout links2 --seed 6"
 "--gatekey submodx --heldout dofhi  --seed 4"
 "--gatekey submodx --heldout dofhi  --seed 5"
 "--gatekey submodx --heldout dofhi  --seed 6"
 "--gatekey submodx --heldout dofhi2 --seed 4"
 "--gatekey submodx --heldout dofhi2 --seed 5"
 "--gatekey submodx --heldout dofhi2 --seed 6"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
