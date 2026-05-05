#!/bin/bash
source .venv/bin/activate

python -m jepax.train.train_ijepa \
    --config configs/ijepa_test.yaml \
    --data_dir ~/.data \
    --save_dir .checkpoints \
    --save_interval 1 \
    --exp_name jepa-test \
    --use_wandb \
    --wandb_project ijepa \
    --profile \
    --profile_start_step 5 \
    --profile_end_step 10 \
    --profile_log_dir .logs
