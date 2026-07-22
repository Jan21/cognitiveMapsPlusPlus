#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-3
#SBATCH --time=01:40:00
#SBATCH --job-name=imgesc
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/esc_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/esc_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
NAG=(3 4 4 5)
Gs=(16 16 20 16)
SAVE=(1 0 0 0)
i=$SLURM_ARRAY_TASK_ID
BASE="--encoder marker --steps 30000 --d 64 --margin 8 --seed 0"
if [ "${SAVE[$i]}" = "1" ]; then
  python3 -u mini17_image.py $BASE --Nag ${NAG[$i]} --G ${Gs[$i]} --save --save_path marker_best.pt
else
  python3 -u mini17_image.py $BASE --Nag ${NAG[$i]} --G ${Gs[$i]}
fi
