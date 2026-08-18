#!/bin/bash
#SBATCH --job-name=wave1b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-19%16
#SBATCH --output=/home/hulajan1/cognitiveMapsPlusPlus/distance_model/wave1b_%a.out
cd /home/hulajan1/cognitiveMapsPlusPlus/distance_model
# wave 1b: seed-1 replicas of the 20 most promising wave-1 configs (selection robustness).
A=(
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --layers 2 --tag w1_L5map_layers2_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --layers 3 --tag w1_L5map_layers3_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --slots 16 --tag w1_L5map_slots16_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --slots 24 --tag w1_L5map_slots24_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --d 256 --layers 3 --tag w1_L5map_d256_L3_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --T 8 --tag w1_L5map_T8_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --markerheads 8 --tag w1_L5map_mh8_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --cnnw 128 --tag w1_L5map_cnnw128_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --cnnmix 1 --tag w1_L5map_cnnmix_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --coordconv 1 --tag w1_L5map_coordconv_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --readout fgpix --tag w1_L5map_fgpix_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --steps 120000 --tag w1_L5map_steps120k_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --cosine 1 --tag w1_L5map_cosine_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --nmaps 400 --tag w1_L5map_nmaps400_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --attnovl 0.1 --tag w1_L5map_ovl01_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --layers 3 --slots 16 --tag w1_L5map_L3_s16_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --layers 3 --slots 12 --d 256 --tag w1_L5map_L3_s12_d256_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --layers 3 --T 8 --tag w1_L5map_L3_T8_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --slots 16 --d 256 --cnnw 128 --tag w1_L5map_s16_d256_w128_s1"
 "python3 switchyard.py --train --enc pureimage --cnnk 1 --readout xattn --slots 12 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --layers 4 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --nobaseline --split map --seed 1 --layers 3 --cosine 1 --steps 120000 --tag w1_L5map_L3_cos120k_s1"
)
echo "TASK $SLURM_ARRAY_TASK_ID : ${A[$SLURM_ARRAY_TASK_ID]}"
${A[$SLURM_ARRAY_TASK_ID]}
