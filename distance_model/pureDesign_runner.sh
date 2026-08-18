#!/bin/bash
# usage: pureDesign_runner.sh <chunk 0..2>  -- L1 pure-image design sweep on a phd box: T x recall x seed, slots. 8 configs per chunk, 2 concurrent.
cd ~/switchyard || exit 1
PY=~/cmppenv/bin/python
P="--train --enc pureimage --cnnw 64 --cnndepth 2 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map --gatesopen --readout xattn --nobaseline --lr 2e-3"
A=()
for sd in 0 1; do for T in 1 2 4 8 14; do for rc in 0 1; do A+=("--slots 8 --T $T --norecall $rc --seed $sd --tag pd_L1_T${T}_rc${rc}_s${sd}"); done; done; done
for sd in 0 1; do for K in 4 16; do A+=("--slots $K --T 4 --seed $sd --tag pd_L1_s${K}_T4_s${sd}"); done; done
CH=$1; i=0
for idx in $(seq $((CH*8)) $((CH*8+7))); do
  cfg="${A[$idx]}"; tag=$(echo "$cfg" | grep -oE "pd_[A-Za-z0-9_]+")
  CUDA_VISIBLE_DEVICES=0 $PY switchyard.py $P $cfg > "$tag.out" 2>&1 &
  i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait; echo DONE > pd_chunk_$CH.done
