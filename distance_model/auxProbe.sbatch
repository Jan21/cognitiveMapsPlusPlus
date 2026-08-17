#!/bin/bash
#SBATCH --job-name=auxProbe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3%4
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/auxProbe_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# UPPER-BOUND PROBE: is binding the bottleneck on the image (CNN+xattn) path?
# Same integ image config as the fair benchmark (baseline markeraux=0 -> 0.788 map / 0.786 wire),
# + SUPERVISED binding aux (decode xattn bound token -> true worker/crate cell). If this lifts toward
# ~0.88-0.91, binding IS the bottleneck and an unsupervised binding route is worth building. If flat, it isn't.
BASE="python3 switchyard.py --train --enc image --readout xattn --cnnw 64 --cnndepth 2 --heads 4 --nobaseline --lr 1e-3 --layers 2 --d 256 --T 14 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --latentnorm 0"
A=(
 "$BASE --markeraux 1.0 --split map  --tag aux_w1_map"
 "$BASE --markeraux 1.0 --split wire --tag aux_w1_wire"
 "$BASE --markeraux 0.5 --split map  --tag aux_w05_map"
 "$BASE --markeraux 0.5 --split wire --tag aux_w05_wire"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
