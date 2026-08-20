#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=dh683b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/dh683b_%a.out
# DECODE-HEAD RECIPE SCREEN AT THE LARGE IMAGE BED (683 maps / 167k pairs): before claiming the -0.14 readout gap
# is intrinsic, give the decode head its own tuning pass here (accumulation runs its tuned recipe; at this bed the
# decode head has only seen lr 5e-4 / 1e-3). Around its viable point (lr 5e-4 -> 0.810/0.815, train MAE 2.9,
# underfit): lr 3e-4/7e-4, T 1/8, warm-up 8k, cosine, 120k steps, batch 256. Seed 0; winner gets 2 more seeds.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
B="python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 --gradclip 1.0 --warmup 2000 --cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --objch 1 --nobaseline --split map --decodehead 1 --evalevery 8000 --seed 0"
A=(
 "$B --lr 3e-4 --tag dhb_lr3e4"
 "$B --lr 7e-4 --tag dhb_lr7e4"
 "$B --lr 5e-4 --T 1 --tag dhb_T1"
 "$B --lr 5e-4 --T 8 --tag dhb_T8"
 "$B --lr 5e-4 --warmup 8000 --tag dhb_wu8k"
 "$B --lr 5e-4 --cosine 1 --tag dhb_cos"
 "$B --lr 5e-4 --steps 120000 --tag dhb_120k"
 "$B --lr 5e-4 --bs 256 --tag dhb_bs256"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
