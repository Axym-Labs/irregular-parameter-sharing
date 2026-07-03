#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/home/davwis/main/exploration/_env/bin/python}"

"$PYTHON" scripts/run_butterfly_lm.py \
  --out runs/butterfly_100m_d16_w768_g16_k128_search12 \
  --tokenizer hf \
  --tokenizer-path /home/davwis/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --max-tokens 100000000 \
  --val-tokens 2000000 \
  --shard-limit 32 \
  --seq-len 256 \
  --batch-size 24 \
  --eval-batch-size 16 \
  --grad-accum 2 \
  --final-steps 5000 \
  --search-steps 800 \
  --search-budget 12 \
  --eval-every 500 \
  --eval-steps 30 \
  --dim 768 \
  --heads 12 \
  --depth 16 \
  --groups 16 \
  --shared-blocks 128 \
  --butterfly-expansion 4 \
  --final-seeds 0,1,2 \
  --final-variants unshared random max_distance best_random \
  --device cuda
