#!/bin/bash

PROJECT_ROOT="/home/STORM"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHON_CMD="python"
TORCHRUN_CMD="torchrun"

echo "Project Root: $PROJECT_ROOT"

cd "$PROJECT_ROOT" || exit 1

if ! command -v $TORCHRUN_CMD &> /dev/null; then
    echo "torchrun not found, using python -m torch.distributed.run instead"
    TORCHRUN_CMD="$PYTHON_CMD -m torch.distributed.run"
fi

echo "Starting VSSM Test with torchrun..."
echo "Using devices: $CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

$TORCHRUN_CMD \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29501 \
    ./classification/main.py \
    --cfg ./classification/configs/vssm/vmambav2_small_224.yaml \
    --batch-size 64 \
    --data-path /home/Vim-main/data/ImageNet-1K \
    --output ./outputs \
    --pretrained ../VMamba-main/weights/vssm_small_0229_ckpt_epoch_222.pth \
    --throughput

echo "Finished."