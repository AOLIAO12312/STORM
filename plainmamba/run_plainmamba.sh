#!/bin/bash

PROJECT_ROOT=$(pwd)

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# 2. 指定 Python 解释器路径 (根据截图中的 Remote Python 路径)
PYTHON_BIN="python"

# 3. 运行分布式测试脚本
$PYTHON_BIN -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=1 \
    --master_port=29500 \
    tools/test.py \
    plain_mamba_configs/plain_mamba_l2_in1k_300e.py \
    /home/PlainMamba/weights/l2.pth \
    --metrics accuracy precision recall f1_score support \
    --launcher pytorch