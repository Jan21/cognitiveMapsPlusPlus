#!/bin/bash
#SBATCH --job-name=igSub4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-17%18
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub4_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# HARD split (--heldout links2: train <=1 active link, test >=2). Does supervised head still compose? add vs submod vs submodx.
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --heldout links2 --steps 40000 --inject 1 --d 128 --layers 4"
A=(
 "--gatekey add     --dofsup 1.0 --seed 0"
 "--gatekey add     --dofsup 1.0 --seed 1"
 "--gatekey add     --dofsup 1.0 --seed 2"
 "--gatekey add     --dofsup 1.0 --seed 3"
 "--gatekey submod  --dofsup 1.0 --seed 0"
 "--gatekey submod  --dofsup 1.0 --seed 1"
 "--gatekey submod  --dofsup 1.0 --seed 2"
 "--gatekey submod  --dofsup 1.0 --seed 3"
 "--gatekey submodx --dofsup 1.0 --seed 0"
 "--gatekey submodx --dofsup 1.0 --seed 1"
 "--gatekey submodx --dofsup 1.0 --seed 2"
 "--gatekey submodx --dofsup 1.0 --seed 3"
 "--gatekey add     --dofsup 0.0 --seed 0"
 "--gatekey add     --dofsup 0.0 --seed 1"
 "--gatekey submod  --dofsup 0.0 --seed 0"
 "--gatekey submod  --dofsup 0.0 --seed 1"
 "--gatekey submodx --dofsup 0.0 --seed 0"
 "--gatekey submodx --dofsup 0.0 --seed 1"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
