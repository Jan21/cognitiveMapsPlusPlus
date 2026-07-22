#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-8
#SBATCH --time=01:40:00
#SBATCH --job-name=imgenc
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/imgenc_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/imgenc_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
ENC=(marker marker marker gather gather gather mha mha mha)
Gs=(8 12 16 8 12 16 8 12 16)
i=$SLURM_ARRAY_TASK_ID
python3 -u mini17_image.py --encoder ${ENC[$i]} --G ${Gs[$i]} --steps 30000 --d 64 --heads 4 --margin 8 --seed 0
