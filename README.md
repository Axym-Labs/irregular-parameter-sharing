# irregular-parameter-sharing

Experiments for irregular parameter sharing in Transformers.

The current experiment is a tensorized shared-basis parameter field for
Transformer MLP weights:

```text
W_l = W_shared + sum_r a[l, r] * B_r
```

The first implementation uses low-rank basis matrices `B_r = u_r v_r^T`. This
shares basis matrices across depth and width while keeping an ordinary dense MLP
component. The main controls are an ordinary unshared Transformer MLP, a fully
shared MLP, and per-layer low-rank residual MLPs with matched factorized
parameter counts.

```bash
PYTHONPATH=src python scripts/run_basis_lm.py \
  --out runs/basis_smoke \
  --device cuda \
  --token-path /home/davwis/main/exploration/irregular_token_lm/cache/openwebtext_gpt2_200000.npy \
  --vocab-size 50257 \
  --val-tokens 20000 \
  --steps 5 \
  --dim 128 \
  --depth 2 \
  --variants shared basis_r4 lowrank_r2 unshared
```

The prior butterfly experiment asked whether sharing small butterfly-style
parameter blocks across both depth and width could outperform random sharing
under the same parameter budget. That code trains decoder-only language models
on local OpenWebText GPT-2 tokens and compares:

- an unshared butterfly-block Transformer reference,
- randomly sampled depth-width sharing layouts,
- the best layout from a fixed random search budget,
- and a maximum-depth-distance sharing layout.

The tensorized basis-field task arc is in `docs/shared-basis-field/`.
The earlier grouped shared-basis toy task arc is in
`docs/shared-basis-sharing/`, and the earlier butterfly task arc is in
`docs/butterfly-sharing/`.

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
