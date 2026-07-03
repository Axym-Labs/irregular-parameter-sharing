# Butterfly Sharing Experiment Spec

## Objective

Replace the previous depth-only sharing experiment with a faithful test of
irregular sharing over butterfly-style parameter blocks. At each depth, the
model should contain multiple separate parameter blocks across width, and the
experiment should compare sharing across both depth and width.

## Requirements

- Use a real repository named `irregular-parameter-sharing`, matching the code
  link in the published post: `https://github.com/Axym-Labs/irregular-parameter-sharing`.
- Implement a Transformer language-model experiment with butterfly-style width
  blocks, so each layer has multiple independently shareable parameter blocks.
- Compare:
  - the practical baseline: an unshared model under the same architecture,
  - a simple same-spirit baseline: randomly sampled sharing layouts,
  - a search result: best of a fixed random sharing budget,
  - a maximum-distance sharing layout that maximizes depth separation among
    repeated uses of the same shared block.
- Report the search budget and method explicitly, e.g. "best of N sampled
  sharing layouts under a short proxy training budget."
- Report average performance across three seeds for randomly sampled sharing
  setups.
- Scale beyond the previous 20M-token, 40M-parameter result. Use the RTX 5090
  effectively and record the actual configuration.
- If maximum-distance sharing beats random sharing, interpret the result in
  terms of depth-distance structure rather than generic irregularity.
- Update the blog only after the new experiment is defensible.

## Acceptance Boundary

- The repo contains runnable code, a README, and reproducible result artifacts.
- The new harness passes a smoke test.
- The large experiment writes `result.json`, `summary.md`, and CSV/markdown
  tables suitable for the blog.
- The blog table is a single extended table with random-average, best-of-N, and
  maximum-distance columns.
- Public GitHub publishing is attempted only through an authenticated path; if
  local credentials block it, record the blocker without faking publication.
