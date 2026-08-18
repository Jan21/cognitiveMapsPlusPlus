#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=slotAttn2Leo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-53%54
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/slotAttn2Leo_%a.out
# Slot attention v2 (recipe fix: lr 1e-3, gradclip 1, warmup 2k) at scale: 80k, 3 seeds, L1.5/L2/L3.
# integ: slotattn T4 | slotattn T1 | slotattn+recon | slotattn iters1 | one-shot xattn (same recipe) ; baselines: iqe on slotattn tokens.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --d 128 --layers 4 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --split map --slots 8 --gradclip 1.0 --warmup 2000"
R=("--nlever 0 --noplate|L1h" "--wire1 --noplate|L2" "--noplate|L3")
A=()
for r in "${R[@]}"; do f="${r%%|*}"; n="${r##*|}"; for sd in 0 1 2; do
  A+=("$B $f --seed $sd --readout slotattn --slotiters 3 --T 4 --nobaseline --lr 1e-3 --tag leo2_${n}_sa_T4_s${sd}")
  A+=("$B $f --seed $sd --readout slotattn --slotiters 3 --T 1 --nobaseline --lr 1e-3 --tag leo2_${n}_sa_T1_s${sd}")
  A+=("$B $f --seed $sd --readout slotattn --slotiters 3 --T 4 --nobaseline --lr 1e-3 --recon slot --tag leo2_${n}_sa_T4_recon_s${sd}")
  A+=("$B $f --seed $sd --readout slotattn --slotiters 1 --T 4 --nobaseline --lr 1e-3 --tag leo2_${n}_sa_i1_T4_s${sd}")
  A+=("$B $f --seed $sd --readout xattn --T 4 --nobaseline --lr 1e-3 --tag leo2_${n}_xattn_T4_s${sd}")
  A+=("$B $f --seed $sd --readout slotattn --slotiters 3 --iqeonly --baselayers 4 --lr 1e-3 --tag leo2_${n}_sa_iqe_s${sd}")
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
srun ${A[$SLURM_ARRAY_TASK_ID]}
