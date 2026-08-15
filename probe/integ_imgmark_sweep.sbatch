#!/bin/bash
#SBATCH --job-name=igMark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-12%13
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igMark_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Make marker (learned attention binding) work: does more training let it bind? + resolve bmask flaky seed.
B="--dofsup 0 --nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --gatekey add --inject 0 --iso 0 --d 128 --layers 4"
A=(
 "--enc marker --heldout combo  --steps 40000 --seed 0"
 "--enc marker --heldout combo  --steps 40000 --seed 1"
 "--enc marker --heldout combo  --steps 40000 --seed 2"
 "--enc marker --heldout links2 --steps 40000 --seed 0"
 "--enc marker --heldout links2 --steps 40000 --seed 1"
 "--enc marker --heldout links2 --steps 40000 --seed 2"
 "--enc marker --heldout combo  --steps 80000 --seed 0"
 "--enc marker --heldout combo  --steps 80000 --seed 1"
 "--enc marker --heldout links2 --steps 80000 --seed 0"
 "--enc marker --heldout links2 --steps 80000 --seed 1"
 "--enc bmask  --heldout combo  --steps 40000 --seed 0"
 "--enc bmask  --heldout combo  --steps 40000 --seed 1"
 "--enc bmask  --heldout combo  --steps 40000 --seed 2"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
