# irregular-parameter-sharing

Experiments for irregular parameter sharing in Transformers.

The current experiment is a simple shared-basis architecture: split the model
width into groups, keep one small bank of basis MLP blocks, and learn mixture
coefficients for every `(layer, width_group)`. This shares parameters across
both depth and width without layout search or butterfly-specific structure.

```bash
PYTHONPATH=src python scripts/run_shared_basis_lm.py \
  --out runs/shared_basis_smoke \
  --device cpu \
  --steps 5 \
  --dim 16 \
  --depth 2 \
  --groups 4 \
  --rank 1
```

The prior butterfly experiment asked whether sharing small butterfly-style
parameter blocks across both depth and width could outperform random sharing
under the same parameter budget. That code trains decoder-only language models
on local OpenWebText GPT-2 tokens and compares:

- an unshared butterfly-block Transformer reference,
- randomly sampled depth-width sharing layouts,
- the best layout from a fixed random search budget,
- and a maximum-depth-distance sharing layout.

The shared-basis task arc is in `docs/shared-basis-sharing/`. The earlier
butterfly task arc is in `docs/butterfly-sharing/`.

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
