# Hard Irregular Block Sharing Spec

## Objective

Test real irregular parameter sharing in a Transformer language model by tying
exact MLP hidden-chunk parameter blocks across explicit `(layer, hidden_chunk)`
positions.

## Mechanism

Represent the ordinary Transformer MLP as a sum over hidden-width chunks:

```text
MLP_l(x) = sum_c GELU(x W1[id(l, c)]) W2[id(l, c)]
```

For the unshared baseline, every `(layer, chunk)` position has a unique block
ID, which is equivalent to a standard dense MLP without biases. For hard
sharing variants, the schedule `id(l, c)` reuses exact block modules across
depth and hidden-width chunk positions.

## Required Baselines

- Practical baseline: unshared dense Transformer MLP, implemented as unique
  hidden chunks.
- Simple same-spirit baseline: balanced random hard-sharing layouts.
- Structured hard-sharing candidate: maximum-depth-distance sharing, where
  repeated uses of the same block are assigned as far apart in depth as
  possible.
- Search baseline: best of a fixed random-layout budget under a short proxy
  training budget.

## Reporting Requirements

- Report search budget, proxy steps, and search method.
- Report mean validation cross-entropy across three final seeds.
- Report total, non-embedding, and MLP-bank parameter counts.
- Treat the previous tensorized shared-basis experiment as a different idea,
  not evidence for hard irregular parameter sharing.

## Acceptance Boundary

- CUDA smoke passes.
- The 16-layer, width-1024, 100M-token run writes `live.jsonl`, `result.json`,
  `runs.csv`, and `summary.md`.
- The blog is revised only after this hard-sharing run finishes.
