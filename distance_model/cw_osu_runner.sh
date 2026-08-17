#!/bin/bash
# CRATEWORLD image (marker) held-out-config comparison, on dgx-osu (venv torch 2.7.1 + torchqmet).
# integ (control; prior: test corr 0.98-0.997) vs IQE / MRN / sym_attn on the SAME encoder + same
# data/loss/steps. Original integ recipe (run_best.sbatch): --nag 4 --G 6 --Rmax 12 --T 14 --d 128 --layers 4
# --heads 4 --poolq 2000 --nquery 80 --lr 2e-3 --inject 1 --enc marker --steps 80000. Baselines use --layers 4 too.
cd ~/crateworld || exit 1
PY=~/switchyard/venv/bin/python
B="--nag 4 --G 6 --Rmax 12 --T 14 --d 128 --layers 4 --heads 4 --poolq 2000 --nquery 80 --lr 2e-3 --inject 1 --enc marker --steps 80000 --seed 0"
run() { CUDA_VISIBLE_DEVICES=$1 $PY integ_distance.py $B $2 > "cw_$3.out" 2>&1; }
# GPU 3 (free): 4 concurrent
( run 3 "--arch integ    --heldout combo"  integ_combo
  run 3 "--arch mrn      --heldout combo  --gradclip 1.0" mrnclip_combo ) &
( run 3 "--arch iqe      --heldout combo"  iqe_combo
  run 3 "--arch mrn      --heldout links2 --gradclip 1.0" mrnclip_links2 ) &
( run 3 "--arch mrn      --heldout combo"  mrn_combo
  run 3 "--arch mrn      --heldout dofhi  --gradclip 1.0" mrnclip_dofhi ) &
( run 3 "--arch sym_attn --heldout combo"  sym_combo ) &
# GPU 2 (shared): 2 concurrent
( run 2 "--arch integ    --heldout links2" integ_links2
  run 2 "--arch mrn      --heldout links2" mrn_links2 ) &
( run 2 "--arch iqe      --heldout links2" iqe_links2
  run 2 "--arch sym_attn --heldout links2" sym_links2 ) &
# GPU 4 (shared): 2 concurrent
( run 4 "--arch integ    --heldout dofhi"  integ_dofhi
  run 4 "--arch mrn      --heldout dofhi"  mrn_dofhi ) &
( run 4 "--arch iqe      --heldout dofhi"  iqe_dofhi
  run 4 "--arch sym_attn --heldout dofhi"  sym_dofhi ) &
wait
echo ALL_DONE > cw_runner.done
