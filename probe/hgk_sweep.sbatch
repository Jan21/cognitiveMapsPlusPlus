#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --array=0-7
#SBATCH --time=01:10:00
#SBATCH --job-name=hgk
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/hgk_out/%a.out
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/hgk_out/%a.err
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
Gs=(12 16 20 12 16 20 16 16)
STEPS=(15000 15000 15000 25000 25000 25000 25000 40000)
DBG=(0 0 0 0 1 0 0 0)
i=$SLURM_ARRAY_TASK_ID
if [ "${DBG[$i]}" = "1" ]; then
  python3 -u mini17_image.py --encoder hgather --keygate --G ${Gs[$i]} --steps ${STEPS[$i]} --d 64 --margin 8 --seed 0 --save --save_path hgk_best.pt
  python3 -u mini19_dof2debug.py --encoder hgather --keygate --G ${Gs[$i]} --d 64 --ckpt hgk_best.pt
else
  python3 -u mini17_image.py --encoder hgather --keygate --G ${Gs[$i]} --steps ${STEPS[$i]} --d 64 --margin 8 --seed 0
fi
