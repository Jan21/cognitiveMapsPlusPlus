#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=slotAttnLeo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-53%54
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/slotAttnLeo_%a.out
# LEONARDO: competitive iterative slot attention at scale -- 80k steps, 3 seeds, L1.5 / L2 / L3, all variants + baselines.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --split map --readout slotattn --slots 8 --slotiters 3"
R=("--nlever 0 --noplate|L1h" "--wire1 --noplate|L2" "--noplate|L3")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"; for sd in 0 1 2; do
  A+=("$B $f --seed $sd --T 4 --nobaseline --lr 2e-3 --tag leo_${n}_sa_T4_s${sd}")
  A+=("$B $f --seed $sd --T 1 --nobaseline --lr 2e-3 --tag leo_${n}_sa_T1_s${sd}")
  A+=("$B $f --seed $sd --T 4 --nobaseline --lr 2e-3 --recon slot --tag leo_${n}_sa_T4_recon_s${sd}")
  A+=("$B $f --seed $sd --T 4 --nobaseline --lr 2e-3 --readout xattn --tag leo_${n}_xattn_T4_s${sd}")
  A+=("$B $f --seed $sd --iqeonly    --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag leo_${n}_sa_iqe_s${sd}")
  A+=("$B $f --seed $sd --scalaronly --baselayers 4 --lr 1e-3 --gradclip 1.0 --tag leo_${n}_sa_scalar_s${sd}")
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
srun ${A[$SLURM_ARRAY_TASK_ID]}
