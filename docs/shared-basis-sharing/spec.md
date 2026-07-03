# Shared-Basis Depth-Width Sharing Spec

## Objective

Pivot from butterfly-style parameter sharing to a conceptually simpler
experiment: share a small basis of grouped MLP blocks across both depth and
width, with learned coefficients selecting mixtures for each `(layer,
width_group)`.

## Requirements

- Treat this as an exploratory architecture experiment, not a benchmark chase.
- Keep the mechanism simple enough to inspect:
  - split hidden width into equal groups,
  - define a global bank of low-rank basis MLP blocks,
  - learn coefficients indexed by layer and width group,
  - apply the resulting grouped MLP inside an otherwise ordinary causal
    Transformer block.
- Keep the previous butterfly experiment intact as prior work.
- Provide a smoke-testable implementation with a tiny model and random token
  inputs before attempting dataset training.
- Report simple structural facts first: parameter counts, tensor shapes,
  forward-pass validity, and whether the model can optimize a toy language
  modeling loss.

## Non-Goals

- Do not tune heavily or try to maximize benchmark performance in this pass.
- Do not add random layout search, max-distance heuristics, or butterfly
  permutations unless a later experiment specifically needs them.
