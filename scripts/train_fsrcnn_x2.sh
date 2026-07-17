#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --train-dir data/91-image \
  --val-dirs data/Set5 data/Set14 \
  --scale 2 \
  --d 56 --s 12 --m 4 \
  --channels 1 \
  --deconv-kernel 9 \
  --deconv-std 0.001 \
  --lr-patch-size 48 \
  --repeat 100 \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3 \
  --save-dir runs/fsrcnn_deconv_x2
