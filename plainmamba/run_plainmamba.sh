#!/bin/bash

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# --- 默认参数设置 ---
DEFAULT_CFG="plain_mamba_configs/plain_mamba_l2_in1k_300e.py"
DEFAULT_CHECKPOINT="/home/PlainMamba/weights/l2.pth"
DEFAULT_GPUS=4
DEFAULT_PORT=29503

# 初始化变量
CFG="$DEFAULT_CFG"
CHECKPOINT="$DEFAULT_CHECKPOINT"
GPUS="$DEFAULT_GPUS"
PORT="$DEFAULT_PORT"

# --- 参数解析 ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --cfg)
      CFG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--cfg <config>] [--checkpoint <path>] [--gpus <num>] [--port <port>]"
      exit 1
      ;;
  esac
done

# --- 环境配置 ---
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)))

PYTHON_CMD="python"
TORCHRUN_CMD="torchrun"

if ! command -v $TORCHRUN_CMD &> /dev/null; then
    TORCHRUN_CMD="$PYTHON_CMD -m torch.distributed.run"
fi

# --- 执行命令 ---
cd "$PROJECT_ROOT" || exit 1

CMD="$TORCHRUN_CMD \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/test.py \
    '$CFG' \
    '$CHECKPOINT' \
    --metrics accuracy precision recall f1_score support \
    --launcher pytorch"

echo "Executing: $CMD"
eval $CMD