#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=int683T1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/int683T1_%a.out
# ACCUMULATION AT T1 ON THE LARGE BED: the decode head's tuned optimum is T1 (0.864 +- 0.025); to compare readouts
# at the decode head's own recurrence depth, run the accumulation integrator at T1 here (lean-bed T1 was 0.857,
# within noise of T4 0.860; T4 large-bed reference 0.953 +- 0.007).
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 1 --lr 1e-3 --objch 1 --nobaseline --split map --evalevery 8000"
A=(
 "$B --seed 0 --tag int683T1_s0"
 "$B --seed 1 --tag int683T1_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
