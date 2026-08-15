#!/bin/bash
#SBATCH --job-name=igSub6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-11%12
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub6_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# CALIBRATION: head_mae (magnitude, not just rank) across all held-out regimes + finer extrapolation dofhi2 (3 values). 20k steps.
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --dofsup 1.0 --steps 20000 --inject 1 --d 128 --layers 4"
A=(
 "--gatekey add     --heldout combo  --seed 0"
 "--gatekey add     --heldout combo  --seed 1"
 "--gatekey add     --heldout links2 --seed 0"
 "--gatekey add     --heldout links2 --seed 1"
 "--gatekey add     --heldout dofhi  --seed 0"
 "--gatekey add     --heldout dofhi  --seed 1"
 "--gatekey add     --heldout dofhi2 --seed 0"
 "--gatekey add     --heldout dofhi2 --seed 1"
 "--gatekey submod  --heldout combo  --seed 0"
 "--gatekey submod  --heldout links2 --seed 0"
 "--gatekey submod  --heldout dofhi  --seed 0"
 "--gatekey submod  --heldout dofhi2 --seed 0"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
