#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=172:00:00
#SBATCH --exclude=hyperplane-4,tinybox

# Usage: WANDB_API_KEY=xxx sbatch train_imnet_slurm.sh

set -euo pipefail

# ------------- Paths -------------
REPO_DIR="/mnt/data0/shared/owen/jepax"

# ------------- Environment -------------
echo "Node: $(hostname)"
echo "Python before activate: $(which python)"

# Detect Ubuntu version and use appropriate venv
UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || grep VERSION_ID /etc/os-release | cut -d'"' -f2)
echo "Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" == "24"* ]]; then
    VENV_PATH="/mnt/data0/shared/owen/venv24"
    echo "Using Ubuntu 24 venv: $VENV_PATH"

    # Create venv if it doesn't exist or is invalid
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "Creating new venv for Ubuntu 24..."
        rm -rf "$VENV_PATH"
        python3 -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip
        pip install -e "$REPO_DIR[experiments]"
        pip install -U "jax[cuda12]"
    else
        source "$VENV_PATH/bin/activate"
        # Ensure CUDA JAX is installed
        pip install -U "jax[cuda12]" -q
    fi
else
    VENV_PATH="/mnt/data0/shared/owen/pain1"
    echo "Using Ubuntu 22 venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

echo "Python after activate: $(which python)"
echo "Python version: $(python --version)"

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
