#!/bin/bash

source .venv/bin/activate

python -m jepax.train.train_ijepa \
    --seed 0 \
    --num_workers 8 \
    --xla_buckets 64 128 192 256 \
    --data_name imnet \
    --data_dir ~/data/imagenet \
    --model_name ijepa-b \
    --save_dir ~/anton/checkpoints/ijepa/ijepa-b-imagenet \
    --save_interval 10 \
    --use_wandb \
    --wandb_project ijepa \
    --patch_size 14 \
    --seq_len 256 \
    --num_channels 3 \
    --batch_size 512 \
    --lr 1.5e-4 \
    --epochs 300 \
    --eval_interval 10 \
    --eval_epochs 20 \
    --weight_decay 0.05 \
    --warmup_epochs 40 \
    --ema_decay 0.996 \
    --num_pred_masks 4 \
    --pred_scale 0.15 0.2 \
    --pred_aspect 0.75 1.5 \
    --ctx_scale 0.85 1.0 \
    --ctx_aspect 1.0 \
    --profile \
    --profile_start_step 5 \
    --profile_end_step 10 \
    --profile_log_dir ~/anton/logs/ijepa/