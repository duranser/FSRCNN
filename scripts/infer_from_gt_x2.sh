#!/usr/bin/env bash
set -euo pipefail

GT_IMAGE="${1:?Usage: $0 path/to/ground_truth.png}"

python inference.py \
  --input "$GT_IMAGE" \
  --output results/inference \
  --checkpoint runs/fsrcnn_deconv_x2/best_psnr.pth
