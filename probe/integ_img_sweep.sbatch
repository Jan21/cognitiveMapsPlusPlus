#!/bin/bash
#SBATCH --job-name=igImg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-15%16
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/igImg_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
export PYTORCH_ALLOC_CONF=expandable_segments:True
# IMAGE PORT: does label-free on-manifold VGT stratification survive image input? factored (ref) vs bmask (lossless) vs marker (learned bind). distance-only.
B="--dofsup 0 --nag 4 --G 6 --Rmax 12 --T 14 --poolq 2000 --nquery 70 --nbank 2000 --rvgt 4.5 --lr 2e-3 --gatekey add --inject 0 --iso 0 --d 128 --layers 4 --steps 20000"
A=(
 "--enc factored --heldout combo  --seed 0"
 "--enc factored --heldout combo  --seed 1"
 "--enc factored --heldout links2 --seed 0"
 "--enc factored --heldout links2 --seed 1"
 "--enc bmask    --heldout combo  --seed 0"
 "--enc bmask    --heldout combo  --seed 1"
 "--enc bmask    --heldout combo  --seed 2"
 "--enc bmask    --heldout links2 --seed 0"
 "--enc bmask    --heldout links2 --seed 1"
 "--enc bmask    --heldout links2 --seed 2"
 "--enc marker   --heldout combo  --seed 0"
 "--enc marker   --heldout combo  --seed 1"
 "--enc marker   --heldout combo  --seed 2"
 "--enc marker   --heldout links2 --seed 0"
 "--enc marker   --heldout links2 --seed 1"
 "--enc marker   --heldout links2 --seed 2"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_links.py $B ${A[$C]}
