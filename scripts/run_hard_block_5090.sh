#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/home/davwis/main/exploration/_env/bin/python}"

"$PYTHON" scripts/run_hard_block_lm.py \
  --out runs/hard_block_100m_d16_w1024_c16_b128_search12 \
  --tokenizer gpt2 \
  --max-tokens 100000000 \
  --val-tokens 2000000 \
  --shard-limit 32 \
  --seq-len 256 \
  --batch-size 12 \
  --eval-batch-size 4 \
  --grad-accum 4 \
  --search-steps 800 \
  --final-steps 5000 \
  --eval-every 500 \
  --eval-steps 30 \
  --dim 1024 \
  --heads 16 \
  --depth 16 \
  --chunks 16 \
  --shared-blocks 128 \
  --search-budget 12 \
  --final-variants unshared random max_distance best_random \
  --final-seeds 0,1,2 \
  --device cuda
