#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=int650
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/int650_%a.out
# INTEGRATOR AT 650 MAPS / ~150k PAIRS (poolq 6100; pool_train ~ 24.5 x poolq): user-requested bed, final image
# recipe, both splits, 3 seeds. Nearby reference: 683 maps / 167k pairs -> 0.953 +- 0.007 map / 0.943 +- 0.020 wire.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 650 --poolq 6100 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --lr 1e-3 --objch 1 --nobaseline --evalevery 8000"
A=(
 "$B --split map  --seed 0 --tag int650_map_s0"
 "$B --split map  --seed 1 --tag int650_map_s1"
 "$B --split map  --seed 2 --tag int650_map_s2"
 "$B --split wire --seed 0 --tag int650_wire_s0"
 "$B --split wire --seed 1 --tag int650_wire_s1"
 "$B --split wire --seed 2 --tag int650_wire_s2"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
