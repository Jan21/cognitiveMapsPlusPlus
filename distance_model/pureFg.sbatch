#!/bin/bash
#SBATCH --job-name=pureFg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-11%12
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pureFg_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# L2 fix candidates, still image-only: fgpix = tokens at FOREGROUND pixels (exact per-entity tokens derived from the picture,
# no labels, no identity); coordconv = x/y channels into the CNN. Refs @200/40k: L2 slots T4 0.762, pixels T14 0.791, factored 0.923.
P="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --seed 0"
L2="--wire1 --noplate"
A=(
 "$P $L2 --readout fgpix --T 4  --nobaseline --lr 2e-3                 --tag fg_L2_integ_T4"
 "$P $L2 --readout fgpix --T 14 --nobaseline --lr 2e-3                 --tag fg_L2_integ_T14"
 "$P $L2 --readout fgpix --T 4  --nobaseline --lr 2e-3 --coordconv 1   --tag fg_L2_integ_T4_cc"
 "$P $L2 --readout xattn --slots 8 --T 4 --nobaseline --lr 2e-3 --coordconv 1  --tag fg_L2_integ_slots_T4_cc"
 "$P $L2 --readout pixels --T 4 --nobaseline --lr 2e-3 --coordconv 1  --tag fg_L2_integ_pix_T4_cc"
 "$P $L2 --readout fgpix --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag fg_L2_iqe"
 "$P $L2 --readout fgpix --symonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag fg_L2_sym"
 "$P $L2 --readout fgpix --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag fg_L2_scalar"
 "$P $L2 --readout fgpix --mrnonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag fg_L2_mrn"
 "$P     --readout fgpix --T 4  --nobaseline --lr 2e-3                 --tag fg_L3_integ_T4"
 "$P     --readout fgpix --T 14 --nobaseline --lr 2e-3                 --tag fg_L3_integ_T14"
 "$P     --readout fgpix --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag fg_L3_scalar"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
