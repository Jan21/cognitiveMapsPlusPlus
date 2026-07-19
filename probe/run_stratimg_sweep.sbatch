#!/bin/bash
#SBATCH --job-name=strat_img
#SBATCH --partition=gpu
# torch 2.9 (cu128) dropped Pascal sm_61 -> avoid GTX1080Ti; Volta100 (sm_70) works.
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=probe/results/strat_img_%j.out

# Sweep to maximize the chance of CLEAN strata from image input: vary the strata-separation
# strength (knob_dist) and seed, bigger encoder, more steps. Each writes a tagged figure.
set -e
cd "$HOME/cognitiveMapsPlusPlus"
mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
$PY -c "import torch;print('cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# tag  knob_dist  seed
for CFG in "kd6 6 1" "kd10 10 1" "kd16 16 1" "kd10b 10 2" "kd22 22 1"; do
  set -- $CFG; TAG=$1; KD=$2; SEED=$3
  echo "========================= $TAG (knob_dist=$KD seed=$SEED) ========================="
  $PY -u probe/stratified_image_probe.py --steps 16000 --eval_every 8000 --knob_dist "$KD" \
      --K 2 --R 6 --N 6000 --per_dof 30 --vis_per_config 200 \
      --d_model 64 --D 48 --slots 8 --seed "$SEED" --tag "$TAG" --device cuda
done
echo "ALL DONE"
