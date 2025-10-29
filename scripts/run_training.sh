#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --account=OPEN-34-14
#SBATCH --partition=qgpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=train/train_%j.out
#SBATCH --error=train/train_%j.err

set -e

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

module --force purge
module load apptainer

unset SINGULARITY_BINDPATH

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
SIF_FILE="${SCRIPT_DIR}/scripts/cognitiveMapsPlusPlus.sif"

cd "$SCRIPT_DIR"

# Bind mount system CA certificates for SSL verification
BIND_OPTS="--bind /etc/pki/tls:/etc/pki/tls:ro"

# Generate graph data first
apptainer exec --nv --cleanenv --env WANDB_API_KEY="$WANDB_API_KEY" --env SSL_CERT_FILE="$SSL_CERT_FILE" "$SIF_FILE" python generate/generate_graph.py

# Generate training data
apptainer exec --nv --cleanenv --env WANDB_API_KEY="$WANDB_API_KEY" --env SSL_CERT_FILE="$SSL_CERT_FILE" "$SIF_FILE" python generate/generate_data.py

# Run training
apptainer exec --nv --cleanenv --env WANDB_API_KEY="$WANDB_API_KEY" --env SSL_CERT_FILE="$SSL_CERT_FILE" "$SIF_FILE" python train.py

# hyperparam sweep
# apptainer exec --nv --cleanenv --env WANDB_API_KEY="$WANDB_API_KEY" --env SSL_CERT_FILE="$SSL_CERT_FILE" "$SIF_FILE" python train.py --multirun