#!/usr/bin/env bash
# Frozen-encoder linear probing on the schizophrenia classification cohort.
set -euo pipefail

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a pre-trained .pth}
DATA_PATH=${DATA_PATH:?set DATA_PATH to the evaluation dataset root}
GPU=${GPU:-0}

python linear_probe.py \
    --checkpoint "${CHECKPOINT}" \
    --data-path "${DATA_PATH}" \
    --folds 5 \
    --epochs 200 \
    --lr 1e-3 \
    --batch-size 256 \
    --gpu "${GPU}"
