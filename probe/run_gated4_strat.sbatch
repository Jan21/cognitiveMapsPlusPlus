#!/bin/bash
#SBATCH --job-name=strat_g4s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=probe/results/gated4strat_%j.out

# 4-agent gated probe WITH illegal-repulsion: make the gate geometric so the EMBEDDING is
# stratified (illegal 1-step neighbours pushed far). Report BFS ladder, embedding-NN ladder,
# and the legal/illegal distance gap. Saves the model.
set -e
cd "$HOME/cognitiveMapsPlusPlus"
mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
export PYTORCH_ALLOC_CONF=expandable_segments:True
$PY -c "import torch;print('cuda',torch.cuda.is_available())"
$PY -u probe/stratified_gated_probe.py --nagents 4 --G 24 --steps 30000 --eval_every 3000 \
    --K 12 --R 12 --cap 60000 --reps 3 \
    --margin 6 --lam_ill 1.0 \
    --emb_W 6 --emb_M 60000 --emb_k 3000 \
    --d_model 96 --D 64 --slots 16 --seed 1 --device cuda \
    --save_model factored_vis/gated4_strat_seed1.pt
echo "ALL DONE"
