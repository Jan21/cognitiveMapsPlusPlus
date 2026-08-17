#!/bin/bash
# CRATEWORLD MARKER-canvas held-out config bed: seed 1 for the baselines (IQE / MRN unclipped / sym_attn) + the
# never-run scalar control (seeds 0,1). integ marker already has 2 seeds (run_best / dgx_dm) + dgx-osu controls.
# Same recipe as run_best.sbatch: --nag 4 --G 6 --Rmax 12 --T 14 --d 128 --layers 4 --heads 4 --poolq 2000
# --nquery 80 --lr 2e-3 --inject 1 --enc marker --steps 80000.  usage: cwmk_phd_runner.sh <box 0..2>
cd ~/crateworld || exit 1
PY=~/cmppenv/bin/python
B="--nag 4 --G 6 --Rmax 12 --T 14 --d 128 --layers 4 --heads 4 --poolq 2000 --nquery 80 --lr 2e-3 --inject 1 --enc marker --steps 80000"
run() { CUDA_VISIBLE_DEVICES=0 $PY integ_distance.py $B $1 > "cwmk_$2.out" 2>&1; }
case "$1" in
 0) ( run "--arch iqe --heldout combo  --seed 1" iqe_combo_s1 ;  run "--arch iqe --heldout links2 --seed 1" iqe_links2_s1 ) &
    ( run "--arch iqe --heldout dofhi  --seed 1" iqe_dofhi_s1 ;  run "--arch scalar --heldout combo --seed 0" scalar_combo_s0 ) &
    ( run "--arch scalar --heldout combo --seed 1" scalar_combo_s1 ) & ;;
 1) ( run "--arch mrn --heldout combo  --seed 1" mrn_combo_s1 ;  run "--arch mrn --heldout links2 --seed 1" mrn_links2_s1 ) &
    ( run "--arch mrn --heldout dofhi  --seed 1" mrn_dofhi_s1 ;  run "--arch scalar --heldout links2 --seed 0" scalar_links2_s0 ) &
    ( run "--arch scalar --heldout links2 --seed 1" scalar_links2_s1 ) & ;;
 2) ( run "--arch sym_attn --heldout combo  --seed 1" sym_combo_s1 ;  run "--arch sym_attn --heldout links2 --seed 1" sym_links2_s1 ) &
    ( run "--arch sym_attn --heldout dofhi  --seed 1" sym_dofhi_s1 ;  run "--arch scalar --heldout dofhi --seed 0" scalar_dofhi_s0 ) &
    ( run "--arch scalar --heldout dofhi --seed 1" scalar_dofhi_s1 ) & ;;
esac
wait
echo ALL_DONE > cwmk_runner_$1.done
