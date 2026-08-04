#!/usr/bin/env bash
# Baseline condition: uniform random MIM masking.
set -euo pipefail

DATA_PATH=${DATA_PATH:?set DATA_PATH to the pre-training dataset root}
OUTPUT_DIR=${OUTPUT_DIR:-./runs/random_mim}
NPROC=${NPROC:-4}

python -m torch.distributed.launch --nproc_per_node="${NPROC}" main.py \
    --data-path "${DATA_PATH}" \
    --mim-mask-mode random \
    --epochs 300 \
    --batch_size_per_gpu 2 \
    --output_dir "${OUTPUT_DIR}" \
    --name random_mim
