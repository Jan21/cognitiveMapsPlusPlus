#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=chk683
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/chk683_%a.out
# REGRESSION BISECT: div683 integ at 683 maps = 0.844 +- 0.016 but the earlier imgDiv probe = 0.953 +- 0.007 with
# IDENTICAL flags (RESULT dicts differ only in tag). Only difference: code version (decode-softplus + scalarsp +
# evalevery edits landed in between). Four cells: {pre-edit file, current file} x {evalevery 0, evalevery 4000}
# minus the pre-edit+eval cell (flag does not exist there), plus a seed-1 replication on the pre-edit file.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="--train --enc pureimage --heads 4 --nmaps 683 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --lr 1e-3 --objch 1 --nobaseline --split map"
A=(
 "python3 switchyard.py $B --evalevery 0 --seed 0 --tag chk_cur_ev0_s0"
 "python3 switchyard.py $B --evalevery 4000 --seed 0 --tag chk_cur_ev4k_s0"
 "python3 switchyard_pre_edits.py $B --seed 0 --tag chk_old_s0"
 "python3 switchyard_pre_edits.py $B --seed 1 --tag chk_old_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
