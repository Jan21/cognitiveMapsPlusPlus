#!/bin/bash
#SBATCH --job-name=pureK1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-5%6
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureK1_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# Is the 3x3 CNN the L2 bottleneck? --cnnk 1 = per-pixel MLP on the raw 12 channels (+pos), no spatial mixing; still image-only.
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --seed 0 --wire1 --noplate"
A=(
 "$P --readout fgpix  --T 4 --nobaseline --lr 2e-3 --tag k1_L2_fg_integ_T4"
 "$P --readout pixels --T 4 --nobaseline --lr 2e-3 --tag k1_L2_pix_integ_T4"
 "$P --readout xattn --slots 8 --T 4 --nobaseline --lr 2e-3 --tag k1_L2_slots_integ_T4"
 "$P --readout fgpix  --T 1 --nobaseline --lr 2e-3 --tag k1_L2_fg_integ_T1"
 "$P --readout fgpix  --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag k1_L2_fg_scalar"
 "$P --readout fgpix  --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag k1_L2_fg_iqe"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
