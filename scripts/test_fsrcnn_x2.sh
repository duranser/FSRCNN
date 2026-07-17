#!/usr/bin/env bash
set -euo pipefail

python test.py \
  --checkpoint runs/fsrcnn_deconv_x2/best_psnr.pth \
  --test-dirs data/Set5 data/Set14 \
  --save-images results/fsrcnn_deconv_x2
