#!/bin/bash
#SBATCH --job-name=igCtrl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-5%6
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igCtrl_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# CONTROL: does VGT survive when the bank is padded with 2000 random distractors (all regimes)? tests model separates near from far.
B="--dofsup 0 --nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --gatekey add --inject 1 --iso 0 --d 128 --layers 4 --steps 20000 --nctrl 2000"
A=(
 "--heldout combo  --seed 0"
 "--heldout combo  --seed 1"
 "--heldout combo  --seed 2"
 "--heldout links2 --seed 0"
 "--heldout links2 --seed 1"
 "--heldout links2 --seed 2"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
