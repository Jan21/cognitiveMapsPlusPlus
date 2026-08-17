#!/bin/bash
# CRATEWORLD LENGTH-GEN (nag2 G14, train dist<=8, test to 25): FAIR-TUNE the baselines under the same
# bellman+bellwarm+ematarget recipe that gave integ 0.74 (with bellrad) / 0.65 (without).
# Prior baseline numbers (MRN 0.64, IQE 0.47) used one depth (4) and one lr (1e-3). Sweep depth x lr here,
# and add sym_attn (never run on this axis with bellman). Runs on the OSU phd boxes (RTX 4060Ti, uv venv).
# usage: cwlen_phd_runner.sh <box-index 0..2>
cd ~/crateworld || exit 1
PY=~/cmppenv/bin/python
B="--enc factored --d 128 --heads 4 --inject 1 --steps 80000 --nag 2 --G 14 --Rmax 25 --Rtrain 8 --maxnodes 400000 --poolq 1200 --nquery 60 --T 24 --bellman 1.0 --bellwarm 1 --ematarget 1 --seed 0"
run() { CUDA_VISIBLE_DEVICES=0 $PY integ_distance.py $B $1 > "cwlen_$2.out" 2>&1; }
case "$1" in
 0) ( run "--arch iqe --layers 2 --lr 1e-3" iqe_L2_lr1e3 ; run "--arch iqe --layers 6 --lr 1e-3" iqe_L6_lr1e3 ) &
    ( run "--arch iqe --layers 4 --lr 3e-4" iqe_L4_lr3e4 ; run "--arch iqe --layers 4 --lr 2e-3" iqe_L4_lr2e3 ) & ;;
 1) ( run "--arch mrn --layers 2 --lr 1e-3" mrn_L2_lr1e3 ; run "--arch mrn --layers 6 --lr 1e-3" mrn_L6_lr1e3 ) &
    ( run "--arch mrn --layers 4 --lr 3e-4" mrn_L4_lr3e4 ; run "--arch mrn --layers 4 --lr 2e-3" mrn_L4_lr2e3 ) & ;;
 2) ( run "--arch sym_attn --layers 4 --lr 1e-3" sym_L4_lr1e3 ; run "--arch sym_attn --layers 2 --lr 1e-3" sym_L2_lr1e3 ) &
    ( run "--arch sym_attn --layers 4 --lr 3e-4" sym_L4_lr3e4 ; run "--arch sym_attn --layers 4 --lr 1e-3 --bellrad 1" sym_L4_bellrad ) & ;;
esac
wait
echo ALL_DONE > cwlen_runner_$1.done
