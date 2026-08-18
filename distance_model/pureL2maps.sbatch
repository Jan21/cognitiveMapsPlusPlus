#!/bin/bash
#SBATCH --job-name=pureL2maps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-6%7
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureL2maps_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# L2 (wire1 + noplate) pure-image: train MAE ~0.8 but test MAE ~3.5 on unseen maps at 200 maps / 40k -> memorising maps.
# Probe: is it DIVERSITY (800 maps) or STEPS (80k)? Fair set on the same slots.
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --poolq 2000 --split map --seed 0 --wire1 --noplate --readout xattn --slots 8"
A=(
 "$P --nmaps 800 --steps 80000 --T 4  --nobaseline --lr 2e-3 --tag l2m_integ_T4_n800"
 "$P --nmaps 800 --steps 80000 --T 14 --nobaseline --lr 2e-3 --tag l2m_integ_T14_n800"
 "$P --nmaps 800 --steps 80000 --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag l2m_iqe_n800"
 "$P --nmaps 800 --steps 80000 --symonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag l2m_sym_n800"
 "$P --nmaps 800 --steps 80000 --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag l2m_scalar_n800"
 "$P --nmaps 200 --steps 80000 --T 4  --nobaseline --lr 2e-3 --tag l2m_integ_T4_n200_80k"      # steps-only control
 "$P --nmaps 800 --steps 80000 --T 4  --nobaseline --lr 2e-3 --readout pixels --tag l2m_integ_pix_T4_n800"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
