#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=win683
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-27%28
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/win683_%a.out
# SEEDS WAVE for the matched-bed (683 maps / poolq 6800) screen winners: per-head best config, both splits,
# seeds 0-2 (map seed 0 rerun for uniformity), plus an objch-off check of each winner (map, seed 0).
# Screen winners (seed 0, map): scalar sl64_bl2_d128_lr1e-3 0.884 | iqe sl64_bl6_d128_lr2e-3 0.861 |
# sym px32_mean0_d256_lr1e-3 0.788 | mrn sl64_bl2_d128_lr2e-3 0.780. integ ref 0.953/0.943 (3 seeds).
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 --gradclip 1.0 --warmup 2000 --evalevery 8000"
IQE="--readout xattn --slots 12 --cnnk 1 --cnnw 64 --basepool mean --baselayers 6 --d 128 --lr 2e-3 --iqeonly"
MRN="--readout xattn --slots 12 --cnnk 1 --cnnw 64 --basepool mean --baselayers 2 --d 128 --lr 2e-3 --mrnonly"
SYM="--readout pixels --cnnk 1 --cnnw 32 --coordconv 1 --basepool mean --baselayers 0 --d 256 --lr 1e-3 --symonly"
SCA="--readout xattn --slots 12 --cnnk 1 --cnnw 64 --basepool mean --baselayers 2 --d 128 --lr 1e-3 --scalaronly"
A=()
for sd in 0 1 2; do for sp in map wire; do
  A+=("$B $IQE --objch 1 --split $sp --seed $sd --tag w683_${sp}_iqe_s${sd}")
  A+=("$B $MRN --objch 1 --split $sp --seed $sd --tag w683_${sp}_mrn_s${sd}")
  A+=("$B $SYM --objch 1 --split $sp --seed $sd --tag w683_${sp}_sym_s${sd}")
  A+=("$B $SCA --objch 1 --split $sp --seed $sd --tag w683_${sp}_scalar_s${sd}")
done; done
A+=("$B $IQE --objch 0 --split map --seed 0 --tag w683_map_iqe_noobj_s0")
A+=("$B $MRN --objch 0 --split map --seed 0 --tag w683_map_mrn_noobj_s0")
A+=("$B $SYM --objch 0 --split map --seed 0 --tag w683_map_sym_noobj_s0")
A+=("$B $SCA --objch 0 --split map --seed 0 --tag w683_map_scalar_noobj_s0")
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
