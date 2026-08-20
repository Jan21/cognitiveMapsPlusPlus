#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=int683no
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/int683no_%a.out
# INTEG WITHOUT THE OBJECTNESS PLANE at the large bed (683 maps / 167k pairs): the one integ cell missing there.
# With objch: 0.953 +- 0.007 map / 0.943 +- 0.020 wire. At 200 maps the plane was worth +0.017 map / 0 wire;
# more data may make the hint redundant. Both splits, seeds 0-1 (extend to seed 2 if the gap is interesting).
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --lr 1e-3 --objch 0 --nobaseline --evalevery 8000"
A=(
 "$B --split map  --seed 0 --tag int683no_map_s0"
 "$B --split map  --seed 1 --tag int683no_map_s1"
 "$B --split wire --seed 0 --tag int683no_wire_s0"
 "$B --split wire --seed 1 --tag int683no_wire_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
