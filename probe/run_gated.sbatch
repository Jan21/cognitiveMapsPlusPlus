#!/bin/bash
#SBATCH --job-name=strat_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=probe/results/gated_%j.out

# Gated-knob image probe: local dimension at representative points (no UMAP).
set -e
cd "$HOME/cognitiveMapsPlusPlus"
mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -c "import torch;print('cuda',torch.cuda.is_available())"
for SEED in 1 2; do
  echo "========================= seed $SEED ========================="
  $PY -u probe/stratified_gated_probe.py --steps 20000 --eval_every 5000 \
      --cap 6000 --reps 4 --K 2 --R 7 --seed "$SEED" --device cuda
done
echo "ALL DONE"
