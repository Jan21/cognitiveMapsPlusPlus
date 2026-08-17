#!/bin/bash
#SBATCH --job-name=gwBench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-19%10
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/gridworld/gwBench_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/gridworld
# GRAND EXPERIMENT (keys-&-doors gridworld): integ vs IQE/MRN/sym/scalar on the SAME CNN+xattn reader,
# same data/loss/steps. Baselines were NEVER run on this bed. Two beds at the original headline recipes:
#  F = fixed 8x8 K1, train D<=8, test to 19, 90k, Ttest22  (integ: near 0.033, flat 19 = FULL length-gen)
#  R = random maps=1200, train D<=8, unseen maps, 45k, Ttest24 (integ: near 0.136 on unseen maps)
F="python3 gw_probe.py --layout fixed  --H 8 --W 8 --K 1 --D 8 --Ttest 22 --steps 90000 --lr 1e-3 --gradclip 1.0"
R="python3 gw_probe.py --layout random --H 8 --W 8 --K 1 --D 8 --Ttest 24 --maps 1200 --steps 45000 --lr 1e-3 --gradclip 1.0"
A=(
 "$F --arch integ --seed 0 --tag gwF_integ_s0"
 "$F --arch integ --seed 1 --tag gwF_integ_s1"
 "$F --arch sym    --baselayers 2 --tag gwF_sym_L2"
 "$F --arch sym    --baselayers 4 --tag gwF_sym_L4"
 "$F --arch iqe    --baselayers 2 --tag gwF_iqe_L2"
 "$F --arch iqe    --baselayers 4 --tag gwF_iqe_L4"
 "$F --arch mrn    --baselayers 2 --tag gwF_mrn_L2"
 "$F --arch mrn    --baselayers 4 --tag gwF_mrn_L4"
 "$F --arch scalar --baselayers 2 --tag gwF_scalar_L2"
 "$F --arch scalar --baselayers 4 --tag gwF_scalar_L4"
 "$R --arch integ --seed 0 --tag gwR_integ_s0"
 "$R --arch integ --seed 1 --tag gwR_integ_s1"
 "$R --arch sym    --baselayers 2 --tag gwR_sym_L2"
 "$R --arch sym    --baselayers 4 --tag gwR_sym_L4"
 "$R --arch iqe    --baselayers 2 --tag gwR_iqe_L2"
 "$R --arch iqe    --baselayers 4 --tag gwR_iqe_L4"
 "$R --arch mrn    --baselayers 2 --tag gwR_mrn_L2"
 "$R --arch mrn    --baselayers 4 --tag gwR_mrn_L4"
 "$R --arch scalar --baselayers 2 --tag gwR_scalar_L2"
 "$R --arch scalar --baselayers 4 --tag gwR_scalar_L4"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
