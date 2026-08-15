#!/bin/bash
#SBATCH --job-name=igN5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=3:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-7%8
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igN5_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# SCALING: does supervised-head finding hold at N=5 (5 agents, DOF up to 10)? add path (submod too slow at N=5). heldout links2 + dof-extrapolation.
B="--nag 5 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --gatekey add --steps 20000 --inject 1 --d 128 --layers 4"
A=(
 "--dofsup 1.0 --heldout links2 --seed 0"
 "--dofsup 1.0 --heldout links2 --seed 1"
 "--dofsup 1.0 --heldout links2 --seed 2"
 "--dofsup 1.0 --heldout dofhi  --seed 0"
 "--dofsup 1.0 --heldout dofhi  --seed 1"
 "--dofsup 1.0 --heldout dofhi2 --seed 0"
 "--dofsup 1.0 --heldout dofhi2 --seed 1"
 "--dofsup 0.0 --heldout links2 --seed 0"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
