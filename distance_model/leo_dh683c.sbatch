#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=dh683c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/dh683c_%a.out
# DECODE HEAD T1 FOLLOW-UP (large bed): T1 is the screen standout (0.893 at lr 5e-4 vs 0.70-0.75 for every T4
# recipe). Give it an lr curve (3e-4 / 1e-3 / 2e-3, seed 0) and the seeds for the standing winner (5e-4, seeds 1-2).
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 1 --objch 1 --nobaseline --split map --decodehead 1 --evalevery 8000"
A=(
 "$B --lr 3e-4 --seed 0 --tag dhc_T1_lr3e4_s0"
 "$B --lr 1e-3 --seed 0 --tag dhc_T1_lr1e3_s0"
 "$B --lr 2e-3 --seed 0 --tag dhc_T1_lr2e3_s0"
 "$B --lr 5e-4 --seed 1 --tag dhc_T1_lr5e4_s1"
 "$B --lr 5e-4 --seed 2 --tag dhc_T1_lr5e4_s2"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
