#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-7
#SBATCH --time=01:40:00
#SBATCH --job-name=imgsweep
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/img_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/img_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
Gs=(8 8 12 12 16 16 12 12)
STEPS=(20000 35000 20000 35000 20000 35000 35000 35000)
HEADS=(4 4 4 4 4 4 8 4)
Dd=(64 64 64 64 64 64 64 96)
i=$SLURM_ARRAY_TASK_ID
python3 -u mini17_image.py --G ${Gs[$i]} --steps ${STEPS[$i]} --heads ${HEADS[$i]} --d ${Dd[$i]} --margin 8 --seed 0
