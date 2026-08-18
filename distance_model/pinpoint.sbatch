#!/bin/bash
#SBATCH --job-name=pinpoint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-27%14
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/pinpoint_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# PINPOINT the L1 -> L2 failure of pure-image slots. Intermediate rungs:
#  L1    = gates open, levers inert                     (--gatesopen)
#  L1.5  = gates with random bits, NO levers            (--nlever 0 --noplate)   -> must READ gate bits from pixels
#  L1.75 = one gate, one lever, wire1                   (--ngate 1 --nlever 1 --wire1 --noplate) -> trivial pairing
#  L2    = 3 gates, 2 levers, wire1                     (--wire1 --noplate)
# x {slots 3x3 CNN (failing), slots 1x1 (fix), factored ref, scalar-1x1, iqe-1x1}; slots with 2 seeds. 200 maps, 40k, unseen maps.
B="python3 switchyard.py --train --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 40000 --split map"
PI="--enc pureimage --cnnw 64 --cnndepth 2 --readout xattn --slots 8"
R=("--gatesopen|L1" "--nlever 0 --noplate|L1h" "--ngate 1 --nlever 1 --wire1 --noplate|L1q" "--wire1 --noplate|L2")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"
  for sd in 0 1; do
    A+=("$B $PI --cnnk 3 $f --T 4 --nobaseline --lr 2e-3 --seed $sd --tag pp_${n}_slots_k3_s${sd}")
    A+=("$B $PI --cnnk 1 $f --T 4 --nobaseline --lr 2e-3 --seed $sd --tag pp_${n}_slots_k1_s${sd}")
  done
  A+=("$B --enc factored $f --T 14 --nobaseline --lr 2e-3 --seed 0 --tag pp_${n}_factored_ref")
  A+=("$B $PI --cnnk 1 $f --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --seed 0 --tag pp_${n}_scalar_k1")
  A+=("$B $PI --cnnk 1 $f --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --seed 0 --tag pp_${n}_iqe_k1")
done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
