#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=fastLeo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-125%126
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/fastLeo_%a.out
# Shortest path to "beats a well-tuned baseline" on the pure-image bed (1x1 one-shot slots, safe recipe, 80k, unseen maps):
#  A) baseline tuning ON THIS BED: iqe/sym/scalar x depth {2,4,6} x lr {5e-4,1e-3,2e-3} at L2, L3 (seed 0)      -> 54
#  B) integ recipe at L3: T {1,4,8} x lr {5e-4,1e-3} x slots {8,12}, 2 seeds                                     -> 24
#  C) WIRE split (rewired causality) at L2/L3/L5: integ T4 (8 & 12 slots) vs iqe/sym/scalar, 3 seeds             -> 45 (+3 integ s12 L5)
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --cnnw 64 --cnndepth 2 --cnnk 1 --readout xattn --d 128 --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000"
A=()
for r in "--wire1 --noplate|L2" "--noplate|L3"; do f="${r%%|*}"; n="${r##*|}"
  for h in iqeonly symonly scalaronly; do for L in 2 4 6; do for lr in 5e-4 1e-3 2e-3; do
    A+=("$B --split map --slots 8 $f --$h --baselayers $L --lr $lr --seed 0 --tag ft_${n}_${h%only}_L${L}_lr${lr}")
  done; done; done
done
for T in 1 4 8; do for lr in 5e-4 1e-3; do for K in 8 12; do for sd in 0 1; do
  A+=("$B --split map --noplate --slots $K --layers 4 --nobaseline --T $T --lr $lr --seed $sd --tag fi_L3_T${T}_lr${lr}_s${K}_s${sd}")
done; done; done; done
for r in "--wire1 --noplate|L2" "--noplate|L3" "|L5"; do f="${r%%|*}"; n="${r##*|}"; for sd in 0 1 2; do
  A+=("$B --split wire $f --slots 8  --layers 4 --nobaseline --T 4 --lr 1e-3 --seed $sd --tag fw_${n}_integ_s8_s${sd}")
  A+=("$B --split wire $f --slots 12 --layers 4 --nobaseline --T 4 --lr 1e-3 --seed $sd --tag fw_${n}_integ_s12_s${sd}")
  A+=("$B --split wire $f --slots 8 --iqeonly    --baselayers 4 --lr 1e-3 --seed $sd --tag fw_${n}_iqe_s${sd}")
  A+=("$B --split wire $f --slots 8 --symonly    --baselayers 4 --lr 1e-3 --seed $sd --tag fw_${n}_sym_s${sd}")
  A+=("$B --split wire $f --slots 8 --scalaronly --baselayers 4 --lr 1e-3 --seed $sd --tag fw_${n}_scalar_s${sd}")
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
srun ${A[$SLURM_ARRAY_TASK_ID]}
