#!/bin/bash
#SBATCH --job-name=igSub
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-19%20
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --heldout combo --steps 40000 --inject 1 --d 128 --layers 4"
A=(
 # --- load-bearing: gatekey ablation at n4 (does submodular readout beat additive?) ---
 "--gatekey add     --seed 0"
 "--gatekey add     --seed 1"
 "--gatekey add     --seed 2"
 "--gatekey add     --seed 3"
 "--gatekey submod  --seed 0"
 "--gatekey submod  --seed 1"
 "--gatekey submod  --seed 2"
 "--gatekey submod  --seed 3"
 "--gatekey submodx --seed 0"
 "--gatekey submodx --seed 1"
 "--gatekey submodx --seed 2"
 "--gatekey submodx --seed 3"
 # --- parallel ideas (n2) ---
 "--gatekey submod  --pool sum --seed 0"          # ablate: does intersection(min) matter vs plain sum?
 "--gatekey submod  --pool sum --seed 1"
 "--gatekey add     --dofsup 0.5 --seed 0"        # direct DOF supervision on additive token
 "--gatekey add     --dofsup 0.5 --seed 1"
 "--gatekey submod  --dofsup 0.5 --seed 0"        # both fixes: structured token + direct supervision
 "--gatekey submod  --dofsup 0.5 --seed 1"
 "--gatekey submodx --dofsup 0.5 --seed 0"        # explicit structure + supervision (upper bound)
 "--gatekey submodx --dofsup 0.5 --seed 1"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
