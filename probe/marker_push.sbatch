#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --job-name=mrkpush
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/mrk_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/mrk_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
Gs=(8 12 16 20 12 16 12 16)
STEPS=(40000 40000 40000 40000 60000 60000 40000 40000)
Dd=(64 64 64 64 64 64 96 96)
i=$SLURM_ARRAY_TASK_ID
python3 -u mini17_image.py --encoder marker --G ${Gs[$i]} --steps ${STEPS[$i]} --d ${Dd[$i]} --heads 4 --margin 8 --seed 0
