#!/bin/bash
#SBATCH --job-name=seeWalls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-75%12
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/seeWalls_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# --seewalls test (colleague's finding): open NON-gate cross doorways (mean 2.3/map) were only implicit in the pooled
# wall token on the factored path. Does exposing them as tokens help, and does it help ONLY the integrator?
# Fair protocol: factored enc, map + wire, seewalls {0,1}. integ at its own recipe (lr 2e-3, L4 d128 T14, no clip) x 3 seeds;
# baselines iqe/mrn/sym/scalar tuned over baselayers {2,4} x lr {1e-3,2e-3} (seed 0, gradclip 1.0 as in the fair benchmark).
B="python3 switchyard.py --train --enc factored --d 128 --T 14 --nmaps 200 --poolq 2000 --steps 80000 --heads 4"
A=()
for sw in 0 1; do for sp in map wire; do
  for sd in 0 1 2; do A+=("$B --nobaseline --layers 4 --lr 2e-3 --seewalls $sw --split $sp --seed $sd --tag sw${sw}_integ_${sp}_s${sd}"); done
  for h in iqeonly mrnonly symonly scalaronly; do for L in 2 4; do for lr in 1e-3 2e-3; do
    A+=("$B --$h --baselayers $L --lr $lr --gradclip 1.0 --seewalls $sw --split $sp --seed 0 --tag sw${sw}_${h%only}_${sp}_L${L}_lr${lr}")
  done; done; done
done; done
echo "TASK $SLURM_ARRAY_TASK_ID / ${#A[@]} : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
