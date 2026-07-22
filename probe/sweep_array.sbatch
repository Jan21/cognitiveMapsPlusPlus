#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-79%24
#SBATCH --time=00:25:00
#SBATCH --job-name=isosweep
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/sweep_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/sweep_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
python3 -u sweep_worker.py --grid_index $SLURM_ARRAY_TASK_ID --steps 8000 --seed 0
