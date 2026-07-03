# irregular-parameter-sharing

Experiments for irregular parameter sharing in Transformers.

The current experiment tests hard irregular sharing of exact Transformer MLP
hidden-chunk parameter blocks. A standard dense MLP can be written as a sum
over hidden-width chunks:

```text
MLP_l(x) = sum_c GELU(x W1[id(l, c)]) W2[id(l, c)]
```

The unshared baseline gives every `(layer, hidden_chunk)` position a unique
block ID, which is equivalent to an ordinary dense Transformer MLP without
biases. The hard-sharing variants reuse exact block IDs across depth and hidden
chunks using balanced random layouts, a maximum-depth-distance layout, and a
best-of-N random-layout search.

```bash
PYTHONPATH=src python scripts/run_hard_block_lm.py \
  --out runs/hard_block_smoke \
  --device cuda \
  --token-path /home/davwis/main/exploration/irregular_token_lm/cache/openwebtext_gpt2_200000.npy \
  --vocab-size 50257 \
  --val-tokens 20000 \
  --search-steps 3 \
  --final-steps 3 \
  --eval-every 3 \
  --dim 128 \
  --depth 2 \
  --chunks 4 \
  --shared-blocks 4 \
  --search-budget 2 \
  --final-variants unshared random max_distance best_random \
  --final-seeds 0,1
```

The prior butterfly experiment asked whether sharing small butterfly-style
parameter blocks across both depth and width could outperform random sharing
under the same parameter budget. That code trains decoder-only language models
on local OpenWebText GPT-2 tokens and compares:

- an unshared butterfly-block Transformer reference,
- randomly sampled depth-width sharing layouts,
- the best layout from a fixed random search budget,
- and a maximum-depth-distance sharing layout.

The current hard block-sharing task arc is in `docs/hard-block-sharing/`.
Earlier exploratory arcs are in `docs/butterfly-sharing/`,
`docs/shared-basis-field/`, and `docs/shared-basis-sharing/`.

## Quick smoke test

```bash
python scripts/run_butterfly_lm.py \
  --out runs/smoke \
  --max-tokens 200000 \
  --val-tokens 20000 \
  --steps 20 \
  --search-budget 2 \
  --final-variants unshared random max_distance \
  --dim 256 \
  --depth 4 \
  --groups 8 \
  --batch-size 8 \
  --eval-batch-size 4
```

## Data

The runner expects OpenWebText parquet shards at
`/home/davwis/main/data/openwebtext/plain_text/` by default.
