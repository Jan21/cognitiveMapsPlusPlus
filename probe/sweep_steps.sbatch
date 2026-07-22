#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-7
#SBATCH --time=00:50:00
#SBATCH --job-name=stepsweep
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/step_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/step_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
STEPS=(16000 24000 32000 48000 16000 24000 32000 48000)
SEED=(0 0 0 0 1 1 1 1)
i=$SLURM_ARRAY_TASK_ID
python3 -u sweep_worker.py --margin 5 --lam_iso 3 --K 6 --d 48 --G 48 --steps ${STEPS[$i]} --seed ${SEED[$i]}
