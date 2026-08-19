#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=dhfix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/dhfix_%a.out
# DECODE-HEAD FIX PROBE: our --decodehead was missing the softplus that the reference scalar_mlp applies
# (reference: softplus(head(mean(tok)))). Question 1: does softplus rescue the decode head on the IMAGE setup
# (previous: train MAE 4-6, corr 0.39-0.75)? Question 2 (control, never run): does our decodehead implementation
# reproduce the reference tie on the SYMBOLIC setup (accumulation 0.904; reference decode head tied there)?
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
I="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --objch 1 --nobaseline --split map --decodehead 1"
S="python3 switchyard.py --train --enc factored --d 128 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --split map --nobaseline --layers 4 --T 14 --lr 2e-3 --decodehead 1"
A=(
 "$I --lr 1e-3 --seed 0 --tag dhfix_img_lr1e3_s0"
 "$I --lr 1e-3 --seed 1 --tag dhfix_img_lr1e3_s1"
 "$I --lr 5e-4 --seed 0 --tag dhfix_img_lr5e4_s0"
 "$I --lr 5e-4 --seed 1 --tag dhfix_img_lr5e4_s1"
 "$S --seed 0 --tag dhfix_sym_s0"
 "$S --seed 1 --tag dhfix_sym_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
