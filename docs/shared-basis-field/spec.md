# Shared-Basis Parameter Field Spec

## Objective

Pivot from butterfly-only sharing to a tensorized/shared-basis parameter field
for Transformer weights. The experiment should directly express reuse across
depth and channel axes, with butterfly or Monarch-style structure treated as an
optional efficient basis family rather than the main contribution.

## Core Parameterization

For each Transformer MLP layer, model the layer-specific matrices as

```text
W_l = W_shared + sum_r a[l, r] * B_r
```

with low-rank basis matrices

```text
B_r = u_r v_r^T
```

as the first faithful implementation. This is a factored parameter field over
`(layer, input-channel, output-channel)`: layer coefficients share basis
matrices across depth, and the low-rank basis shares structure across width.

## Required Baselines

- Practical baseline: ordinary unshared Transformer MLP weights.
- Simple same-spirit baseline: fully shared MLP weights across depth.
- Same-family controls: per-layer low-rank residual MLPs without a shared
  residual basis. For the 16-layer, width-1024 setup, `basis_r128` is matched
  against `lowrank_r8`, and `basis_r512` is matched against `lowrank_r32`.

## Reporting Requirements

- Report total, non-embedding, and MLP/basis parameter counts.
- Report mean validation cross-entropy across seeds.
- Report the exact basis rank, coefficient initialization, and whether the
  shared dense component is present.
- Use GPT-style small-normal initialization for embeddings and linear weights
  so all variants start in a normal language-model loss range.
- Do not interpret butterfly-specific distance results as evidence for the
  shared-basis design.
- Update the blog only after scaled results are available.

## Acceptance Boundary

- Add a runnable shared-basis LM harness to the repository.
- Pass a CUDA smoke test.
- Run a scaled OpenWebText experiment on the RTX 5090.
- Produce `result.json`, `runs.csv`, `summary.md`, and `live.jsonl`.
