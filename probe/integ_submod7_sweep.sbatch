#!/bin/bash
#SBATCH --job-name=igSub7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-8%9
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub7_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# CONFIRM the one soft claim: submod/submodx better CALIBRATED than add on the HARD links2 split (head_mae). n4 via seeds 1-3 (+seed0 from wave6/4).
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --dofsup 1.0 --heldout links2 --steps 20000 --inject 1 --d 128 --layers 4"
A=(
 "--gatekey add     --seed 1"
 "--gatekey add     --seed 2"
 "--gatekey add     --seed 3"
 "--gatekey submod  --seed 1"
 "--gatekey submod  --seed 2"
 "--gatekey submod  --seed 3"
 "--gatekey submodx --seed 1"
 "--gatekey submodx --seed 2"
 "--gatekey submodx --seed 3"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
