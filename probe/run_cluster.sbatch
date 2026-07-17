#!/bin/bash
#SBATCH --job-name=bridged_tori
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=probe/results/slurm_%j.out

# Runs the bridged-tori factored-attention probe on a GPU node for all three
# distance-head variants (B3 free scalar -> fails; cross_reg + self_norm -> fixes).
set -e
cd "$HOME/cognitiveMapsPlusPlus"
mkdir -p probe/results
PY="$HOME/miniconda3/bin/python"

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for HEAD in cross_scalar cross_reg self_norm; do
  echo "========================= $HEAD ========================="
  $PY probe/bridged_tori_probe.py --head "$HEAD" --steps 4000 --eval_every 500 \
      --seed 0 --json_out "probe/results/${HEAD}.json"
done
echo "ALL DONE"
