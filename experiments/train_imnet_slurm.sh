#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=172:00:00

# Usage: WANDB_API_KEY=xxx sbatch train_imnet_slurm.sh

set -euo pipefail

# ------------- Paths -------------
REPO_DIR="/mnt/data0/shared/owen/jepax"

# ------------- Environment -------------
source /mnt/data0/shared/owen/pain1/bin/activate

# ------------- W&B Setup -------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Warning: WANDB_API_KEY not set. W&B logging will be disabled."
fi

export WANDB_MODE=online
export WANDB_DIR="/mnt/data0/shared/owen/tmp/wandb_$SLURM_JOB_ID"
mkdir -p "$WANDB_DIR"

cd "$REPO_DIR"

# ------------- Run -------------
python -m jepax.train.train_ijepa --config configs/imagenet.yaml
