#!/bin/bash
#SBATCH --job-name=strat_gate4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:30:00
#SBATCH --output=probe/results/gated4_%j.out

# Gated-knob probe, 4 agents (DOF 0..8), G=24 image input. Local dimension vs true DOF at
# representative points (DOF ladder), measured by exact BFS ball + correlation-dim (no UMAP).
set -e
cd "$HOME/cognitiveMapsPlusPlus"
mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
export PYTORCH_ALLOC_CONF=expandable_segments:True
$PY -c "import torch;print('cuda',torch.cuda.is_available())"
for SEED in 1 2; do
  echo "========================= seed $SEED ========================="
  $PY -u probe/stratified_gated_probe.py --nagents 4 --G 24 --steps 30000 --eval_every 6000 \
      --K 12 --R 12 --cap 60000 --reps 4 --batch 512 \
      --d_model 96 --D 64 --slots 16 --seed "$SEED" --device cuda
done
echo "ALL DONE"
