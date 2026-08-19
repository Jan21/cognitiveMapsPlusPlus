#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=dhfix2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/dhfix2_%a.out
# DECODE-HEAD SYMBOLIC CONTROL, ROUND 2: the reference tie ran the decode head at lr 1e-3 (integrate at 2e-3);
# our first symbolic control used lr 2e-3 and collapsed (dead softplus, constant output). Rerun our decodehead
# on the symbolic bed at lr 1e-3 / 5e-4 x 2 seeds. Accumulation reference on this bed: 0.904 +- 0.007.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
S="python3 switchyard.py --train --enc factored --d 128 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --split map --nobaseline --layers 4 --T 14 --decodehead 1 --evalevery 8000"
A=(
 "$S --lr 1e-3 --seed 0 --tag dhfix2_sym_lr1e3_s0"
 "$S --lr 1e-3 --seed 1 --tag dhfix2_sym_lr1e3_s1"
 "$S --lr 5e-4 --seed 0 --tag dhfix2_sym_lr5e4_s0"
 "$S --lr 5e-4 --seed 1 --tag dhfix2_sym_lr5e4_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
