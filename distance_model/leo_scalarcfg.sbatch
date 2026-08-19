#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=scalarCfg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/scalarCfg_%a.out
# Probe of the colleague's CONFIGS.md scalar recipe: factored + seewalls 1, 4 layers, lr 1e-3, 60k, high diversity.
# Compare vs our tuned scalar (L2 depth, no seewalls, 150 maps, 80k -> 0.796 at L5 map) and integ 0.904.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc factored --scalaronly --baselayers 4 --d 128 --heads 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --poolq 2000 --steps 60000 --split map"
A=(
 "$B --nmaps 200 --seewalls 1 --seed 0 --tag sc_cfg_n200_sw1_s0"
 "$B --nmaps 200 --seewalls 1 --seed 1 --tag sc_cfg_n200_sw1_s1"
 "$B --nmaps 200 --seewalls 1 --seed 2 --tag sc_cfg_n200_sw1_s2"
 "$B --nmaps 560 --seewalls 1 --seed 0 --tag sc_cfg_n560_sw1_s0"
 "$B --nmaps 560 --seewalls 1 --seed 1 --tag sc_cfg_n560_sw1_s1"
 "$B --nmaps 560 --seewalls 0 --seed 0 --tag sc_cfg_n560_sw0_s0"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
srun ${A[$SLURM_ARRAY_TASK_ID]}
