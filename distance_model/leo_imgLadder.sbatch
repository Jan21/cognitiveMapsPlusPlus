#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=imgLadderLeo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-104%105
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/imgLadderLeo_%a.out
# PURE-IMAGE (1x1 per-pixel encoder, one-shot slots) interdependence ladder L0..L6 with the SAFE recipe
# (lr 1e-3, gradclip 1, warmup 2k, which cured seed bimodality) -- integ T4 / T1 vs iqe / sym / scalar, 3 seeds, 80k, unseen maps.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --readout xattn --slots 8 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --split map --gradclip 1.0 --warmup 2000 --lr 1e-3"
R=("--gatesopen --nopush|L0" "--gatesopen|L1" "--wire1 --noplate|L2" "--noplate|L3" "--nchute 0|L4" "|L5" "--ngate 4 --nlever 3|L6")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"; for sd in 0 1 2; do
  A+=("$B $f --seed $sd --nobaseline --T 4 --tag img_${n}_integ_T4_s${sd}")
  A+=("$B $f --seed $sd --nobaseline --T 1 --tag img_${n}_integ_T1_s${sd}")
  A+=("$B $f --seed $sd --iqeonly    --baselayers 4 --tag img_${n}_iqe_s${sd}")
  A+=("$B $f --seed $sd --symonly    --baselayers 4 --tag img_${n}_sym_s${sd}")
  A+=("$B $f --seed $sd --scalaronly --baselayers 4 --tag img_${n}_scalar_s${sd}")
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
srun ${A[$SLURM_ARRAY_TASK_ID]}
