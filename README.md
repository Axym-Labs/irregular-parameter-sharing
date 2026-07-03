# irregular-parameter-sharing

Experiments for irregular parameter sharing in Transformers.

The current experiment asks whether sharing small butterfly-style parameter
blocks across both depth and width can outperform random sharing under the same
parameter budget. The code trains decoder-only language models on local
OpenWebText GPT-2 tokens and compares:

- an unshared butterfly-block Transformer reference,
- randomly sampled depth-width sharing layouts,
- the best layout from a fixed random search budget,
- and a maximum-depth-distance sharing layout.

The main task arc is in `docs/butterfly-sharing/`.

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
