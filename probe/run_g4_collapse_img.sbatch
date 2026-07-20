#!/bin/bash
#SBATCH --job-name=g4_col_img
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=probe/results/g4colimg_%j.out

# E7: image encoder + COLLAPSE loss (illegal 1-step move -> same point). Test whether the image
# encoder can learn to quotient immovable axes so graph-free emb-NN reads the DOF ladder.
set -e
cd "$HOME/cognitiveMapsPlusPlus"; mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
export PYTORCH_ALLOC_CONF=expandable_segments:True
$PY -u probe/stratified_gated_probe.py --nagents 4 --G 24 --steps 30000 --eval_every 3000 \
    --K 12 --R 12 --cap 60000 --reps 3 --encoder image --lam_collapse 1.0 \
    --emb_W 6 --emb_M 60000 --emb_k 2000 --d_model 96 --D 64 --slots 16 --seed 1 --device cuda \
    --save_model factored_vis/gated4_collapse_img.pt
echo "ALL DONE"
