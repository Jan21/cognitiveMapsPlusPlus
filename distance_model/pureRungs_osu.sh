#!/bin/bash
# dgx-osu: the two ladder rungs missing from pureLadder -- L3' (--noplate: XOR multi-gate wiring) and L4' (--nchute 0: full minus chute).
# Compute GPUs 2 and 4 only (index 3 = NVIDIA DGX Display, never used). 3 concurrent per GPU.
cd ~/switchyard || exit 1
PY=./venv/bin/python
P="--train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --seed 0 --readout xattn --slots 8"
run() { CUDA_VISIBLE_DEVICES=$1 $PY switchyard.py $P $2 > "pr_$3.out" 2>&1; }
( run 2 "--noplate --T 4  --nobaseline --lr 2e-3" L3x_integ_T4;  run 2 "--nchute 0 --T 4  --nobaseline --lr 2e-3" L4x_integ_T4 ) &
( run 2 "--noplate --T 14 --nobaseline --lr 2e-3" L3x_integ_T14; run 2 "--nchute 0 --T 14 --nobaseline --lr 2e-3" L4x_integ_T14 ) &
( run 2 "--noplate --symonly --baselayers 4 --lr 1e-3 --gradclip 1.0" L3x_sym; run 2 "--nchute 0 --symonly --baselayers 4 --lr 1e-3 --gradclip 1.0" L4x_sym ) &
( run 4 "--noplate --iqeonly --baselayers 4 --lr 1e-3 --gradclip 1.0" L3x_iqe; run 4 "--nchute 0 --iqeonly --baselayers 4 --lr 1e-3 --gradclip 1.0" L4x_iqe ) &
( run 4 "--noplate --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0" L3x_scalar; run 4 "--nchute 0 --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0" L4x_scalar ) &
wait; echo DONE > pr_runner.done
