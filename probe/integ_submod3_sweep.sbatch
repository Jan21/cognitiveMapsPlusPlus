#!/bin/bash
#SBATCH --job-name=igSub3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-23%24
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igSub3_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# KEY experiment: does the supervised DOF-head COMPOSE to held-out combos, and is structured(submod/submodx) >> additive?
# reports corr_vgt / corr_mle / corr_head. grid gatekey{add,submod,submodx} x dofsup{0.5,1.0} x seeds{0-3}
B="--nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --heldout combo --steps 40000 --inject 1 --d 128 --layers 4"
A=(
 "--gatekey add     --dofsup 0.5 --seed 0"
 "--gatekey add     --dofsup 0.5 --seed 1"
 "--gatekey add     --dofsup 1.0 --seed 0"
 "--gatekey add     --dofsup 1.0 --seed 1"
 "--gatekey add     --dofsup 1.0 --seed 2"
 "--gatekey add     --dofsup 1.0 --seed 3"
 "--gatekey submod  --dofsup 0.5 --seed 0"
 "--gatekey submod  --dofsup 0.5 --seed 1"
 "--gatekey submod  --dofsup 1.0 --seed 0"
 "--gatekey submod  --dofsup 1.0 --seed 1"
 "--gatekey submod  --dofsup 1.0 --seed 2"
 "--gatekey submod  --dofsup 1.0 --seed 3"
 "--gatekey submodx --dofsup 0.5 --seed 0"
 "--gatekey submodx --dofsup 0.5 --seed 1"
 "--gatekey submodx --dofsup 1.0 --seed 0"
 "--gatekey submodx --dofsup 1.0 --seed 1"
 "--gatekey submodx --dofsup 1.0 --seed 2"
 "--gatekey submodx --dofsup 1.0 --seed 3"
 # head with NO metric loss weight change -- also a couple no-dofsup structured (head untrained baseline, corr_head~0)
 "--gatekey submodx --dofsup 0.0 --seed 0"
 "--gatekey submodx --dofsup 0.0 --seed 1"
 "--gatekey add     --dofsup 0.0 --seed 0"
 "--gatekey add     --dofsup 0.0 --seed 1"
 "--gatekey submod  --dofsup 0.0 --seed 0"
 "--gatekey submod  --dofsup 0.0 --seed 1"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
