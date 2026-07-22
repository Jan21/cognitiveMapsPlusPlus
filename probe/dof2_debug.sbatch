#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --time=00:50:00
#SBATCH --job-name=dof2dbg
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/probe/dof2_out.txt
#SBATCH --error=/home/hulajan1/cognitiveMapsPlusPlus/probe/dof2_err.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hulajan1/cognitiveMapsPlusPlus/probe
python3 -u mini17_image.py --encoder marker --G 12 --steps 30000 --d 64 --heads 4 --margin 8 --save --save_path marker_g12.pt
python3 -u mini19_dof2debug.py --encoder marker --G 12 --d 64 --heads 4 --ckpt marker_g12.pt
