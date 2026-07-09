# Hard Irregular Block Sharing Progress

- RUN CONTROL (2026-07-10): The corrected long run was externally terminated
  with exit code 143 after about four hours, during seed 2's unshared final
  run at step 3000. Evidence points to session/process management rather than
  model failure: no Python traceback, no result files because the harness writes
  final artifacts only on completion, GPU memory was free afterward, and the
  live log contains completed search plus eight completed final runs. Added a
  final-only resume path via `--best-layout-seed`; the completed search selected
  best layout seed `10005`.
- USER (2026-07-09): User confirmed the unrelated `pptrain` GPU job has
  finished and asked to continue. Verified the RTX 5090 is free and the
  corrected `batch_size=8`, `grad_accum=6` hard-sharing run has not yet
  started at `runs/hard_block_100m_d16_w1024_c16_b128_b8ga6_search12/`.
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
