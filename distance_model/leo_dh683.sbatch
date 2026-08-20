#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=dh683
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/dh683_%a.out
# DECODE HEAD AT THE LARGE IMAGE BED (683 maps / poolq 6800 = 167k pairs): the missing cell of the readout ladder.
# Reference tie happened at 512 maps / 501k pairs (symbolic); if the decode head's problem is data hunger it should
# close on the large image bed (accumulation there: 0.953 +- 0.007). lr 5e-4 (its viable lr on the lean bed) and
# lr 1e-3 (dead-softplus at lean data; more data may stabilize), 2 seeds each.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --objch 1 --nobaseline --split map --decodehead 1 --evalevery 8000"
A=(
 "$B --lr 5e-4 --seed 0 --tag dh683_lr5e4_s0"
 "$B --lr 5e-4 --seed 1 --tag dh683_lr5e4_s1"
 "$B --lr 1e-3 --seed 0 --tag dh683_lr1e3_s0"
 "$B --lr 1e-3 --seed 1 --tag dh683_lr1e3_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
