#!/bin/bash
#SBATCH --job-name=cwImg2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-14%15
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/cwImg2_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
export PYTORCH_ALLOC_CONF=expandable_segments:True
# PHASE 2 of the crateworld IMAGE config-gen benchmark: seed 1 at each baseline's BEST (depth, lr) from cwImg,
# + integ seed 2. Gives error bars for the final table.
B="python3 integ_distance.py --nag 4 --G 6 --Rmax 12 --T 14 --d 128 --heads 4 --poolq 2000 --nquery 80 --inject 1 --enc image --cnnw 64 --cnndepth 2 --steps 80000"
A=(
 "$B --arch integ    --layers 4 --lr 2e-3 --heldout combo  --seed 2"
 "$B --arch integ    --layers 4 --lr 2e-3 --heldout links2 --seed 2"
 "$B --arch integ    --layers 4 --lr 2e-3 --heldout dofhi  --seed 2"
 "$B --arch iqe      --layers 4 --lr 1e-3 --heldout combo  --seed 1"
 "$B --arch iqe      --layers 4 --lr 1e-3 --heldout links2 --seed 1"
 "$B --arch iqe      --layers 4 --lr 1e-3 --heldout dofhi  --seed 1"
 "$B --arch mrn      --layers 2 --lr 1e-3 --heldout combo  --seed 1"
 "$B --arch mrn      --layers 2 --lr 2e-3 --heldout links2 --seed 1"
 "$B --arch mrn      --layers 2 --lr 2e-3 --heldout dofhi  --seed 1"
 "$B --arch sym_attn --layers 4 --lr 1e-3 --heldout combo  --seed 1"
 "$B --arch sym_attn --layers 2 --lr 2e-3 --heldout links2 --seed 1"
 "$B --arch sym_attn --layers 4 --lr 1e-3 --heldout dofhi  --seed 1"
 "$B --arch scalar   --layers 2 --lr 1e-3 --heldout combo  --seed 1"
 "$B --arch scalar   --layers 2 --lr 1e-3 --heldout links2 --seed 1"
 "$B --arch scalar   --layers 4 --lr 1e-3 --heldout dofhi  --seed 1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
