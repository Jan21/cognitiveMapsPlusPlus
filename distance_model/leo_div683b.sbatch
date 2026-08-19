#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=div683b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/div683b_%a.out
# 683-MAP ADDENDUM: the slot-encoder (param-matched) variants of IQE and sym. Reason: at 683 maps the lone earlier
# control (iqe on the slot encoder, seed 0) hit 0.836 while the "best-at-200-maps" 3x3-flatten iqeO lands at 0.767;
# variant ranking flips with map diversity (flatten memorizes layouts). Best baseline at 683 must be searched, not
# assumed from the 200-map tuning. Both with objectness plane, both splits, 3 seeds, eval curve on.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000 --evalevery 4000"
IQM="--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3 --objch 1 --iqeonly"
SYM="--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3 --objch 1 --symonly"
A=()
for sd in 0 1 2; do for sp in map wire; do
  A+=("$B $IQM --split $sp --seed $sd --tag d683_${sp}_iqeMO_s${sd}")
  A+=("$B $SYM --split $sp --seed $sd --tag d683_${sp}_symMO_s${sd}")
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
