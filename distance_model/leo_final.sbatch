#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=finalLeo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-116%117
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/finalLeo_%a.out
# FINAL pure-image ladders (1x1 one-shot slots, safe recipe, 80k, 3 seeds):
#  MAP  ladder L3..L6 rerun with 12 slots (8 < #entities there): integ T4 -> 12 runs
#  WIRE ladder L0..L6, integ T4 12 slots vs iqe/sym/scalar at their ON-BED tuned configs
#     (iqe: L4 lr1e-3 (L2 best) / L6 lr2e-3 (L3 best) -> use L6 lr2e-3 for L3+, L4 lr1e-3 for L0-L2;
#      sym: L2 lr2e-3; scalar: L2 lr5e-4 (L2 best) / L6 lr1e-3 (L3 best)) -> 7 rungs x 4 heads x 3 seeds = 84
#  plus MAP L3..L6 iqe/scalar at tuned configs, 3 seeds -> 24  (sym default already ~tuned)
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --readout xattn --d 128 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000"
R=("--gatesopen --nopush|L0" "--gatesopen|L1" "--wire1 --noplate|L2" "--noplate|L3" "--nchute 0|L4" "|L5" "--ngate 4 --nlever 3|L6")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"
  case $n in L0|L1|L2) IQ="--baselayers 4 --lr 1e-3"; SC="--baselayers 2 --lr 5e-4";; *) IQ="--baselayers 6 --lr 2e-3"; SC="--baselayers 6 --lr 1e-3";; esac
  for sd in 0 1 2; do
    A+=("$B --split wire $f --slots 12 --layers 4 --nobaseline --T 4 --lr 1e-3 --seed $sd --tag fw2_${n}_integ_s${sd}")
    A+=("$B --split wire $f --slots 8 --iqeonly    $IQ --seed $sd --tag fw2_${n}_iqe_s${sd}")
    A+=("$B --split wire $f --slots 8 --symonly    --baselayers 2 --lr 2e-3 --seed $sd --tag fw2_${n}_sym_s${sd}")
    A+=("$B --split wire $f --slots 8 --scalaronly $SC --seed $sd --tag fw2_${n}_scalar_s${sd}")
  done
  case $n in L3|L4|L5|L6) for sd in 0 1 2; do
    A+=("$B --split map $f --slots 12 --layers 4 --nobaseline --T 4 --lr 1e-3 --seed $sd --tag fm2_${n}_integ_s${sd}")
    A+=("$B --split map $f --slots 8 --iqeonly    $IQ --seed $sd --tag fm2_${n}_iqe_s${sd}")
    A+=("$B --split map $f --slots 8 --scalaronly $SC --seed $sd --tag fm2_${n}_scalar_s${sd}")
  done;; esac
done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
srun ${A[$SLURM_ARRAY_TASK_ID]}
