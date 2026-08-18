#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=symLadderLeo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-104%105
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/symLadderLeo_%a.out
# SYMBOLIC (factored) interdependence ladder with FAIRLY TUNED baselines, 3 seeds, 80k, unseen maps:
# rungs L0..L5 + L6 = deeper coupling (4 gates, 3 levers, dense XOR wiring). integ own recipe vs iqe/mrn/sym/scalar at their tuned recipes.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc factored --d 128 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --split map"
R=("--gatesopen --nopush|L0" "--gatesopen|L1" "--wire1 --noplate|L2" "--noplate|L3" "--nchute 0|L4" "|L5" "--ngate 4 --nlever 3|L6")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"; for sd in 0 1 2; do
  A+=("$B $f --seed $sd --nobaseline --layers 4 --T 14 --lr 2e-3 --tag sym_${n}_integ_s${sd}")
  A+=("$B $f --seed $sd --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag sym_${n}_iqe_s${sd}")
  A+=("$B $f --seed $sd --mrnonly    --baselayers 2 --lr 1e-3 --gradclip 1.0 --tag sym_${n}_mrn_s${sd}")
  A+=("$B $f --seed $sd --symonly    --baselayers 2 --lr 1e-3 --gradclip 1.0 --tag sym_${n}_sym_s${sd}")
  A+=("$B $f --seed $sd --scalaronly --baselayers 2 --lr 1e-3 --gradclip 1.0 --tag sym_${n}_scalar_s${sd}")
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
srun ${A[$SLURM_ARRAY_TASK_ID]}
