#!/bin/bash

PROJECT_ROOT=$(pwd)

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

PYTHON_PATH="python"

$PYTHON_PATH -m torch.distributed.run \
    --nproc_per_node=4 \
    tools/test.py \
    -c configs/strategies/local_mamba/config.yaml \
    --model timm_local_vssm_small \
    --drop-path-rate 0.1 \
    --experiment lightvit_small_test \
    --resume /home/LocalMamba/weights/local_vssm_small.ckpt \
    --data-path /home/Vim-main/data/ImageNet-1K