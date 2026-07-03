#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/home/davwis/main/exploration/_env/bin/python}"

"$PYTHON" scripts/run_basis_lm.py \
  --out runs/basis_100m_d16_w1024_r128_r512 \
  --tokenizer gpt2 \
  --max-tokens 100000000 \
  --val-tokens 2000000 \
  --shard-limit 32 \
  --seq-len 256 \
  --batch-size 12 \
  --eval-batch-size 4 \
  --grad-accum 4 \
  --steps 5000 \
  --eval-every 500 \
  --eval-steps 30 \
  --dim 1024 \
  --heads 16 \
  --depth 16 \
  --variants shared basis_r128 lowrank_r8 basis_r512 lowrank_r32 unshared \
  --seeds 0,1,2 \
  --device cuda
