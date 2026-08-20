#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=dh683w
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/dh683w_%a.out
# DECODE HEAD, WIRE SPLIT at the large bed: its tuned config (T1, lr 5e-4) has only map-split runs; 3 seeds on
# rewired causality to complete the comparison chart. Map: 0.864 +- 0.025; accumulation wire ref: 0.943 +- 0.020.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 1 --lr 5e-4 --objch 1 --nobaseline --split wire --decodehead 1 --evalevery 8000"
A=(
 "$B --seed 0 --tag dh683w_s0"
 "$B --seed 1 --tag dh683w_s1"
 "$B --seed 2 --tag dh683w_s2"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
