#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CFG="./classification/configs/vssm/vmambav2_base_224.yaml" # path/to/config
DEFAULT_BATCH_SIZE=64
DEFAULT_DATA_PATH="/home/Vim-main/data/ImageNet-1K" # path/to/data
DEFAULT_OUTPUT="./outputs" # path/to/output
DEFAULT_PRETRAINED="../../VMamba-main/weights/vssm_base_0229_ckpt_epoch_237.pth" # path/to/ckpt
DEFAULT_THROUGHPUT=true

CFG="$DEFAULT_CFG"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
DATA_PATH="$DEFAULT_DATA_PATH"
OUTPUT="$DEFAULT_OUTPUT"
PRETRAINED="$DEFAULT_PRETRAINED"
THROUGHPUT="$DEFAULT_THROUGHPUT"

while [[ $# -gt 0 ]]; do
  case $1 in
    --cfg)
      CFG="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --pretrained)
      PRETRAINED="$2"
      shift 2
      ;;
    --throughput)
      THROUGHPUT=true
      shift 1
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--cfg <config_file>] [--batch-size <size>] [--data-path <data_dir>] [--output <output_dir>] [--pretrained <pretrained_model_path>] [--throughput]"
      exit 1
      ;;
  esac
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHON_CMD="python"
TORCHRUN_CMD="torchrun"

echo "Project Root: $PROJECT_ROOT"
echo "Config: $CFG"
echo "Batch Size: $BATCH_SIZE"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT"
echo "Pretrained Model: $PRETRAINED"
echo "Throughput Mode: $THROUGHPUT"
echo "----------------------------------------"

cd "$PROJECT_ROOT" || exit 1

if ! command -v $TORCHRUN_CMD &> /dev/null; then
    echo "torchrun not found, using python -m torch.distributed.run instead"
    TORCHRUN_CMD="$PYTHON_CMD -m torch.distributed.run"
fi

echo "Starting VSSM Test with torchrun..."
echo "Using devices: $CUDA_VISIBLE_DEVICES"

CMD="$TORCHRUN_CMD \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29501 \
    ./classification/main.py \
    --cfg '$CFG' \
    --batch-size '$BATCH_SIZE' \
    --data-path '$DATA_PATH' \
    --output '$OUTPUT' \
    --pretrained '$PRETRAINED'"

if [ "$THROUGHPUT" = true ]; then
    CMD="$CMD --throughput"
fi

echo "Executing command: $CMD"
eval $CMD

echo "Finished."