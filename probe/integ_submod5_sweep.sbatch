#!/bin/bash
#SBATCH --job-name=igSub5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-11%12
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub5_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# EXTRAPOLATION (--heldout dofhi: train dof<=6, test dof>=7). Does supervised head extrapolate to unseen DOF levels? 20k steps (head converges fast).
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --heldout dofhi --steps 20000 --inject 1 --d 128 --layers 4"
A=(
 "--gatekey add     --dofsup 1.0 --seed 0"
 "--gatekey add     --dofsup 1.0 --seed 1"
 "--gatekey add     --dofsup 1.0 --seed 2"
 "--gatekey submod  --dofsup 1.0 --seed 0"
 "--gatekey submod  --dofsup 1.0 --seed 1"
 "--gatekey submod  --dofsup 1.0 --seed 2"
 "--gatekey submodx --dofsup 1.0 --seed 0"
 "--gatekey submodx --dofsup 1.0 --seed 1"
 "--gatekey submodx --dofsup 1.0 --seed 2"
 "--gatekey add     --dofsup 0.0 --seed 0"
 "--gatekey add     --dofsup 0.0 --seed 1"
 "--gatekey submod  --dofsup 0.0 --seed 0"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
