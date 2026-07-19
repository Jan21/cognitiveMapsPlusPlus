#!/bin/bash
#SBATCH --job-name=strat_img4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=02:30:00
#SBATCH --output=probe/results/strat_img4_%j.out

# 4 agents (DOF 0..8), bigger grid (24x24), image input, unit-cost knob changes.
set -e
cd "$HOME/cognitiveMapsPlusPlus"
mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
$PY -c "import torch;print('cuda',torch.cuda.is_available())"
for CFG in "a4 1" "a4b 2"; do
  set -- $CFG; TAG=$1; SEED=$2
  echo "===== $TAG (4 agents, G=24, knob_dist=1, seed=$SEED) ====="
  $PY -u probe/stratified_image_probe.py --nagents 4 --G 24 --steps 20000 --eval_every 10000 \
      --knob_dist 1 --K 2 --R 8 --N 30000 --per_dof 25 --n_vis 3600 \
      --d_model 96 --D 64 --slots 16 --seed "$SEED" --tag "$TAG" --device cuda
done
echo "ALL DONE"
