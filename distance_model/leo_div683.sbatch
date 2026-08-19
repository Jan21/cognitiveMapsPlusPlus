#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=div683
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-35%36
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/div683_%a.out
# MAP-DIVERSITY CAMPAIGN AT 683 MAPS (matches the imgDiv integ probe): ALL baselines at their L5-best phase-B'
# variants, both splits, 3 seeds, 80k, + best-checkpoint tracking (--evalevery 4000 -> best_mae/best_corr in RESULT).
# integ rerun with the eval curve. scalar in two arms: phase-B' protocol (no softplus) at the per-split winning
# variant (map scalarMO / wire scalarO) AND the same variant with --scalarsp 1 (reference concat_mlp parity).
# mrn runs as mrnO on both splits (L5 winners: map mrn 0.729 vs wire mrnO 0.733; gap within noise).
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000 --evalevery 4000"
INTEG="--cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --lr 1e-3 --objch 1 --nobaseline"
IQE="--readout pixels --cnnk 3 --coordconv 1 --cnndepth 3 --cnnw 128 --basepool flat --baselayers 0 --d 128 --lr 1e-3 --objch 1 --iqeonly"
MRN="--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 2e-3 --objch 1 --mrnonly"
SYM="--readout pixels --cnnk 1 --coordconv 1 --basepool flat --baselayers 0 --d 256 --lr 1e-3 --symonly"
SCU="--readout pixels --cnnk 3 --coordconv 1 --cnndepth 4 --cnnw 128 --basepool flat --baselayers 4 --d 128 --lr 1e-3 --objch 1 --scalaronly"
SCM="--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3 --objch 1 --scalaronly"
A=()
for sd in 0 1 2; do for sp in map wire; do
  A+=("$B $INTEG --split $sp --seed $sd --tag d683_${sp}_integ_s${sd}")
  A+=("$B $IQE   --split $sp --seed $sd --tag d683_${sp}_iqeO_s${sd}")
  A+=("$B $MRN   --split $sp --seed $sd --tag d683_${sp}_mrnO_s${sd}")
  A+=("$B $SYM   --split $sp --seed $sd --tag d683_${sp}_sym_s${sd}")
done; done
for sd in 0 1 2; do
  A+=("$B $SCM --split map  --seed $sd --tag d683_map_scalarMO_s${sd}")
  A+=("$B $SCU --split wire --seed $sd --tag d683_wire_scalarO_s${sd}")
  A+=("$B $SCM --scalarsp 1 --split map  --seed $sd --tag d683_map_scalarMOsp_s${sd}")
  A+=("$B $SCU --scalarsp 1 --split wire --seed $sd --tag d683_wire_scalarOsp_s${sd}")
done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
