#!/bin/bash
#SBATCH --job-name=replicate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-8%9
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/replicate_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# REPLICATION of the original CNN+xattn integ result (imgCNN.sbatch: 0.827/0.807/0.785 map s0/1/2, 0.850/0.827 wire s0/1)
# with the CURRENT code. Command line VERBATIM from imgCNN.sbatch (defaults: lr 2e-3, cnnw 32, no gradclip, heads 4).
# Then single-knob isolations: what did the fair benchmark's lr 1e-3 / gradclip 1.0 cost integ?
ORIG="python3 switchyard.py --train --d 128 --layers 4 --T 14 --nmaps 200 --poolq 2000 --steps 80000 --enc image --nobaseline --readout xattn"
A=(
 "$ORIG --split map  --seed 0 --tag rep_orig_map_s0"      # expect ~0.827
 "$ORIG --split map  --seed 1 --tag rep_orig_map_s1"      # expect ~0.807
 "$ORIG --split map  --seed 2 --tag rep_orig_map_s2"      # expect ~0.785
 "$ORIG --split wire --seed 0 --tag rep_orig_wire_s0"     # expect ~0.850
 "$ORIG --split wire --seed 1 --tag rep_orig_wire_s1"     # expect ~0.827
 "$ORIG --split map  --seed 0 --gradclip 1.0 --tag rep_clip_map_s0"      # isolate gradclip
 "$ORIG --split wire --seed 0 --gradclip 1.0 --tag rep_clip_wire_s0"
 "$ORIG --split map  --seed 0 --lr 1e-3 --tag rep_lr1e3_map_s0"          # isolate lr
 "$ORIG --split wire --seed 0 --lr 1e-3 --tag rep_lr1e3_wire_s0"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
