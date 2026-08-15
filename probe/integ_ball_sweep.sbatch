#!/bin/bash
#SBATCH --job-name=igBall
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-17%18
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igBall_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# LABEL-FREE stratification: distance-only models. Does clean-ball VGT (corr_vgtball) recover the oracle ~0.97?
# Does near-shell weighting (--iso) sharpen local fidelity? Compare jitter-VGT vs ball-VGT. NO dofsup.
B="--dofsup 0 --nag 4 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --heldout combo --steps 20000 --inject 1 --d 128 --layers 4"
A=(
 # near-shell weighting sweep (local-fidelity lever), G=6, add
 "--G 6 --gatekey add --iso 0 --seed 0"
 "--G 6 --gatekey add --iso 0 --seed 1"
 "--G 6 --gatekey add --iso 0 --seed 2"
 "--G 6 --gatekey add --iso 1 --seed 0"
 "--G 6 --gatekey add --iso 1 --seed 1"
 "--G 6 --gatekey add --iso 3 --seed 0"
 "--G 6 --gatekey add --iso 3 --seed 1"
 "--G 6 --gatekey add --iso 3 --seed 2"
 "--G 6 --gatekey add --iso 6 --seed 0"
 "--G 6 --gatekey add --iso 6 --seed 1"
 # structured token: does submodular readout help the GEOMETRY stratify (label-free)?
 "--G 6 --gatekey submodx --iso 3 --seed 0"
 "--G 6 --gatekey submodx --iso 3 --seed 1"
 # bigger grid empirical check (oracle says worse, but test the model)
 "--G 8 --gatekey add --iso 3 --seed 0"
 "--G 8 --gatekey add --iso 3 --seed 1"
 # hard split, label-free
 "--G 6 --gatekey add     --iso 3 --heldout links2 --seed 0"
 "--G 6 --gatekey add     --iso 3 --heldout links2 --seed 1"
 "--G 6 --gatekey submodx --iso 3 --heldout links2 --seed 0"
 "--G 6 --gatekey submodx --iso 3 --heldout links2 --seed 1"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
