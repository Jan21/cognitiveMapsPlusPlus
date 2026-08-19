#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=spfix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/spfix_%a.out
# SCALAR-HEAD SOFTPLUS PROBE (fairness): the reference concat_mlp applies softplus to its output; our Scalar
# baseline did not. The two unconstrained-output heads (scalar, decode head) are exactly the two that collapse
# on the image bed. Rerun the scalar baseline at its L5 best variants (scalarO unconstrained, scalarMO matched,
# both with the objectness plane) WITH softplus (--scalarsp 1), map split, 2 seeds. No-softplus references:
# scalarO 0.721 +- 0.025, scalarMO 0.730 +- 0.046 (phase B', 3 seeds).
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000 --split map --scalaronly --scalarsp 1 --objch 1"
U="--readout pixels --cnnk 3 --coordconv 1 --cnndepth 4 --cnnw 128 --basepool flat --baselayers 4 --d 128 --lr 1e-3"
M="--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3"
A=(
 "$B $U --seed 0 --tag spfix_scalarO_s0"
 "$B $U --seed 1 --tag spfix_scalarO_s1"
 "$B $M --seed 0 --tag spfix_scalarMO_s0"
 "$B $M --seed 1 --tag spfix_scalarMO_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
