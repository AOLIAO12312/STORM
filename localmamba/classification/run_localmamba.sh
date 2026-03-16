#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

DEFAULT_CFG="configs/strategies/local_mamba/config.yaml"
DEFAULT_MODEL="timm_local_vssm_small"
DEFAULT_DROP_PATH=0.1
DEFAULT_EXP="local_mamba_test"
DEFAULT_RESUME="/home/LocalMamba/weights/local_vssm_small.ckpt"
DEFAULT_DATA_PATH="/home/Vim-main/data/ImageNet-1K"
DEFAULT_GPUS=4

CFG="$DEFAULT_CFG"
MODEL="$DEFAULT_MODEL"
DROP_PATH="$DEFAULT_DROP_PATH"
EXP="$DEFAULT_EXP"
RESUME="$DEFAULT_RESUME"
DATA_PATH="$DEFAULT_DATA_PATH"
GPUS="$DEFAULT_GPUS"

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--cfg)
      CFG="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --drop-path)
      DROP_PATH="$2"
      shift 2
      ;;
    --exp)
      EXP="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [-c <config>] [--model <model_name>] [--drop-path <rate>] [--exp <exp_name>] [--resume <ckpt>] [--data-path <path>] [--gpus <num>]"
      exit 1
      ;;
  esac
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)))

PYTHON_CMD="python"
TORCHRUN_CMD="torchrun"

if ! command -v $TORCHRUN_CMD &> /dev/null; then
    TORCHRUN_CMD="$PYTHON_CMD -m torch.distributed.run"
fi

echo "----------------------------------------"
echo "Project Root: $PROJECT_ROOT"
echo "Config: $CFG"
echo "Model: $MODEL"
echo "Drop Path Rate: $DROP_PATH"
echo "Experiment: $EXP"
echo "Resume Checkpoint: $RESUME"
echo "Data Path: $DATA_PATH"
echo "Using GPUs: $GPUS (Devices: $CUDA_VISIBLE_DEVICES)"
echo "----------------------------------------"

cd "$PROJECT_ROOT" || exit 1

CMD="$TORCHRUN_CMD \
    --nproc_per_node=$GPUS \
    --master_port=29502 \
    tools/test.py \
    -c '$CFG' \
    --model '$MODEL' \
    --drop-path-rate '$DROP_PATH' \
    --experiment '$EXP' \
    --resume '$RESUME' \
    --data-path '$DATA_PATH'"

echo "Executing: $CMD"
eval $CMD

echo "Finished."