#!/bin/bash
#SBATCH --job-name=strat_unit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=probe/results/strat_unit_%j.out

# Unit-cost knob changes (a knob flip costs the same as a position move). Strata stay glued,
# so we show stratification as a DIMENSION FIELD (embedding colored by measured local dim).
set -e
cd "$HOME/cognitiveMapsPlusPlus"
mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"
$PY -c "import torch;print('cuda',torch.cuda.is_available())"
for CFG in "unit 1" "unitb 2"; do
  set -- $CFG; TAG=$1; SEED=${2}
  echo "===== $TAG (knob_dist=1 seed=$SEED) ====="
  $PY -u probe/stratified_image_probe.py --steps 16000 --eval_every 8000 --knob_dist 1 \
      --K 2 --R 6 --N 6000 --per_dof 30 --vis_per_config 220 \
      --d_model 64 --D 48 --slots 8 --seed "$SEED" --tag "$TAG" --device cuda
done
echo "ALL DONE"
