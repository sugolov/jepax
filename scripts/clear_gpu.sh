#!/bin/bash

kill -9 $(nvidia-smi --query-compute-apps=pid  --format=csv,noheader -i 0 | head -1)
sudo nvidia-smi --gpu-reset
nvidia-smi