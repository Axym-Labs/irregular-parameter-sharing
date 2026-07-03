# Hard Irregular Block Sharing Progress

- USER (2026-07-03): Clarified that the tensorized shared-basis pivot is not
  compatible with the intended idea because it is soft sharing through learned
  coefficients, not real irregular parameter sharing. Requested changing the
  experimental setup accordingly and continuing the run.
- DECISION (2026-07-03): Use a hard MLP hidden-chunk sharing setup instead of
  the soft shared-basis parameter field. The unshared variant is a standard
  dense Transformer MLP decomposed into hidden chunks; sharing variants reuse
  exact chunk modules across explicit `(layer, hidden_chunk)` positions.
- IMPLEMENTATION (2026-07-03): Added `src/irregular_parameter_sharing/hard_block_lm.py`
  plus `scripts/run_hard_block_lm.py` and `scripts/run_hard_block_5090.sh`.
  The harness supports unshared dense chunks, balanced random hard sharing,
  maximum-depth-distance hard sharing, and best-of-N random layout search.
- VERIFICATION (2026-07-03): CUDA smoke passed at depth 2, width 128,
  4 hidden chunks, 4 shared blocks, search budget 2, and final seeds 0/1.
  The smoke wrote `runs/hard_block_smoke/{live.jsonl,result.json,runs.csv,summary.md}`.
- VERIFICATION (2026-07-03): Full-size profile passed at depth 16, width 1024,
  16 hidden chunks, and 128 shared blocks. Parameter counts verify that the
  unshared baseline is dense-MLP sized: `unshared` has 253,119,488 total
  parameters, 201,394,176 non-embedding parameters, and 134,217,728 MLP-bank
  parameters. Hard-sharing variants have 186,010,624 total parameters,
  134,285,312 non-embedding parameters, and 67,108,864 MLP-bank parameters.
- RUN CONTROL (2026-07-03): First long run at batch size 12 / grad accumulation
  4 was stopped after 6/12 search trials because active GPU memory reached
  about 31.4GB on the smaller hard-sharing model, leaving too little headroom
  for the larger unshared dense baseline. Relaunching with batch size 8 and
  grad accumulation 6 preserves the effective batch size of 48 sequences per
  optimizer step while reducing activation memory.
