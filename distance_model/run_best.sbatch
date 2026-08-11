#!/bin/bash
#SBATCH --job-name=distModel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-17%18
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/dm_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Best distance-prediction experiments: accuracy + generalization. factored / bmask / marker (image) x heldout splits.
B="--nag 4 --G 6 --Rmax 12 --T 14 --d 128 --layers 4 --heads 4 --poolq 2000 --nquery 80 --lr 2e-3 --inject 1"
A=(
 # factored reference (fast) -- all splits, 2 seeds
 "--enc factored --heldout combo  --steps 40000 --seed 0"
 "--enc factored --heldout combo  --steps 40000 --seed 1"
 "--enc factored --heldout links2 --steps 40000 --seed 0"
 "--enc factored --heldout links2 --steps 40000 --seed 1"
 "--enc factored --heldout dofhi  --steps 40000 --seed 0"
 "--enc factored --heldout dofhi  --steps 40000 --seed 1"
 # bmask (lossless image) -- combo + links2
 "--enc bmask    --heldout combo  --steps 40000 --seed 0"
 "--enc bmask    --heldout combo  --steps 40000 --seed 1"
 "--enc bmask    --heldout links2 --steps 40000 --seed 0"
 "--enc bmask    --heldout links2 --steps 40000 --seed 1"
 # marker (learned image binding) -- needs more steps
 "--enc marker   --heldout combo  --steps 80000 --seed 0"
 "--enc marker   --heldout combo  --steps 80000 --seed 1"
 "--enc marker   --heldout links2 --steps 80000 --seed 0"
 "--enc marker   --heldout links2 --steps 80000 --seed 1"
 "--enc marker   --heldout dofhi  --steps 80000 --seed 0"
 "--enc marker   --heldout dofhi  --steps 80000 --seed 1"
 # factored dofhi2 (extra extrapolation point)
 "--enc factored --heldout dofhi2 --steps 40000 --seed 0"
 "--enc factored --heldout dofhi2 --steps 40000 --seed 1"
)
C=$SLURM_ARRAY_TASK_ID
echo "CFG=$C [${A[$C]}]"
python3 integ_distance.py $B ${A[$C]}
