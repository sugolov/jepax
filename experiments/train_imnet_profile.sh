#!/bin/bash
source .venv/bin/activate

# Uses imagenet config with profiling enabled via config override
python -m jepax.train.train_ijepa --config configs/imagenet_profile.yaml
