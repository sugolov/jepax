#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./train_imnet.sh WANDB_API_KEY"
    exit 1
fi

source .venv/bin/activate

export WANDB_API_KEY="$1"

nohup python -m jepax.train.train_ijepa --config configs/imagenet.yaml > train.log 2>&1 &

echo "Training started with PID $!"
echo "Logs: tail -f train.log"
echo "Stop: kill $!"
