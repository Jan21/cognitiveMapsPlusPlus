#!/bin/bash
#SBATCH --job-name=g4_fact
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=probe/results/g4fact_%j.out

# E5: knob-gated FACTORED encoder (collapse by architecture). Two gate modes:
#   learned  - gate derived from knob via sigmoid MLP, taught by the collapse loss
#   hardwired- gate zeroes immovable axes exactly (upper bound / sanity)
set -e
cd "$HOME/cognitiveMapsPlusPlus"; mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
export PYTORCH_ALLOC_CONF=expandable_segments:True
COMMON="--nagents 4 --G 24 --steps 20000 --eval_every 4000 --K 12 --R 12 --cap 60000 --reps 3 \
        --encoder factored --d_model 64 --D 64 --emb_W 6 --emb_M 60000 --emb_k 2000 --seed 1 --device cuda"
echo "===== E5-learned (factored, gate=learned + collapse loss) ====="
$PY -u probe/stratified_gated_probe.py $COMMON --gate learned --lam_collapse 1.0 \
    --save_model factored_vis/gated4_fact_learned.pt
echo "===== E5-hard (factored, gate=hardwired) ====="
$PY -u probe/stratified_gated_probe.py $COMMON --gate hardwired \
    --save_model factored_vis/gated4_fact_hard.pt
echo "ALL DONE"
