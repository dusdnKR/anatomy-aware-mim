#!/usr/bin/env bash
# Proposed condition: anatomy-aware (atlas-guided) MIM masking.
set -euo pipefail

DATA_PATH=${DATA_PATH:?set DATA_PATH to the pre-training dataset root}
OUTPUT_DIR=${OUTPUT_DIR:-./runs/atlas_mim}
NPROC=${NPROC:-4}

python -m torch.distributed.launch --nproc_per_node="${NPROC}" main.py \
    --data-path "${DATA_PATH}" \
    --mim-mask-mode atlas \
    --epochs 300 \
    --batch_size_per_gpu 2 \
    --output_dir "${OUTPUT_DIR}" \
    --name atlas_mim
