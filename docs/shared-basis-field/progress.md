# Shared-Basis Parameter Field Progress

- EXPERIMENT (2026-07-03): Ran a one-seed 500-step scale sweep for the high-rank
  pair on 98M train tokens / 2M validation tokens, depth 16, width 1024. Search
  budget was four `basis_scale` values: 0.5, 1.0, 2.0, and 4.0. At equal
  factorized parameter budget, `basis_r512` beat `lowrank_r32` at all four
  scales: 5.9980 vs 6.0254 CE at 0.5, 6.0030 vs 6.0271 CE at 1.0, 5.9996 vs
  6.0238 CE at 2.0, and 5.9964 vs 6.0181 CE at 4.0. Chose `basis_scale=4.0`
  for the multi-seed run because it produced the best absolute high-rank
  shared-basis validation CE while using the same scale for the matched
  low-rank control.
- EXPERIMENT (2026-07-03): Ran a one-seed 500-step pilot over all main variants
  on the 100M-token cache. Final validation CE: `shared` 6.2334,
  `basis_r128` 6.1089, matched `lowrank_r8` 6.0569, `basis_r512` 6.0030,
  matched `lowrank_r32` 6.0271, and `unshared` 6.0428. This is promising but
  not decisive because it is a short one-seed pilot.
- VERIFICATION (2026-07-03): CUDA smoke passed after adding GPT-style
  small-normal initialization and the `lowrank_rN` control. Command wrote
  `runs/basis_smoke_v2/` with variants `shared`, `basis_r4`, `lowrank_r2`,
  and `unshared`; validation CE values were in the normal untrained GPT range
  around 10.7 instead of the invalid 100+ range produced by the first
  PyTorch-default initialization smoke.
- VERIFICATION (2026-07-03): 16-layer, width-1024 profile passed on the RTX
  5090 with variants `shared`, `basis_r128`, `lowrank_r8`, `basis_r512`,
  `lowrank_r32`, and `unshared`. The matched controls have nearly identical
  factorized parameter counts: `basis_r128` has 1,314,816 basis parameters
  versus 1,310,720 low-rank parameters for `lowrank_r8`; `basis_r512` has
  5,259,264 basis parameters versus 5,242,880 low-rank parameters for
  `lowrank_r32`.
- IMPLEMENTATION (2026-07-03): Added `src/irregular_parameter_sharing/basis_lm.py`
  and `scripts/run_basis_lm.py`. The harness trains decoder-only Transformers
  on cached OpenWebText GPT-2 tokens, logs per-evaluation events to
  `live.jsonl`, reports total/non-embedding/MLP/basis/low-rank parameter
  counts, and supports shared dense, shared-basis, per-layer low-rank, and
  unshared dense MLP variants.
- USER (2026-07-03): Provided feedback that butterfly matrices are a plausible
  width mixer but not the clean general answer. Requested pivot toward
  tensorized/shared-basis parameterization across layer and channel axes:
  `W_layer = W_shared + sum_r a[layer, r] * B_r`, with butterfly/Monarch/low-rank
  blocks as possible basis families.
- DECISION (2026-07-03): First implementation will use low-rank basis matrices
  `B_r = u_r v_r^T` for Transformer MLP weights. This directly shares a basis
  across depth and width while staying efficient enough for a 5090-scale
  OpenWebText run.
