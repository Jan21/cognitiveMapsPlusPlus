#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-3
#SBATCH --time=01:10:00
#SBATCH --job-name=imgrob
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/rob_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/rob_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
SEED=(1 2 3 1)
NAG=(3 3 3 4)
Gs=(16 16 16 20)
i=$SLURM_ARRAY_TASK_ID
python3 -u mini17_image.py --encoder marker --Nag ${NAG[$i]} --G ${Gs[$i]} --steps 30000 --d 64 --margin 8 --seed ${SEED[$i]}
