#!/bin/bash
#SBATCH --job-name=igSub2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-23%24
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub2_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --heldout combo --steps 40000 --inject 1 --d 128 --layers 4"
A=(
 # confirm submod/submodx vs add at n8 (seeds 4-7, combine with wave1 seeds 0-3)
 "--gatekey add     --seed 4"
 "--gatekey add     --seed 5"
 "--gatekey add     --seed 6"
 "--gatekey add     --seed 7"
 "--gatekey submod  --seed 4"
 "--gatekey submod  --seed 5"
 "--gatekey submod  --seed 6"
 "--gatekey submod  --seed 7"
 "--gatekey submodx --seed 4"
 "--gatekey submodx --seed 5"
 "--gatekey submodx --seed 6"
 "--gatekey submodx --seed 7"
 # confirm DOF-supervision variance-kill + best combo at n6 (seeds 2-5, combine with wave1 seeds 0-1)
 "--gatekey add     --dofsup 0.5 --seed 2"
 "--gatekey add     --dofsup 0.5 --seed 3"
 "--gatekey add     --dofsup 0.5 --seed 4"
 "--gatekey add     --dofsup 0.5 --seed 5"
 "--gatekey submodx --dofsup 0.5 --seed 2"
 "--gatekey submodx --dofsup 0.5 --seed 3"
 "--gatekey submodx --dofsup 0.5 --seed 4"
 "--gatekey submodx --dofsup 0.5 --seed 5"
 "--gatekey submod  --dofsup 0.5 --seed 2"
 "--gatekey submod  --dofsup 0.5 --seed 3"
 "--gatekey submod  --dofsup 0.5 --seed 4"
 "--gatekey submod  --dofsup 0.5 --seed 5"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
