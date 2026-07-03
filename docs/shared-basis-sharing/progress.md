# Shared-Basis Depth-Width Sharing Progress

- VERIFICATION (2026-07-03): Added `SharedBasisGroupedMLP`, `UnsharedGroupedMLP`,
  `SharedBasisGPT`, `UnsharedGroupedGPT`, and a random-token toy suite. Verified
  with `/home/davwis/miniconda3/envs/bnn_sim/bin/python -m pytest -q`
  (`3 passed`) and compile check
  `/home/davwis/miniconda3/envs/bnn_sim/bin/python -m compileall -q src scripts tests`.

- EXPERIMENT (2026-07-03): Ran a CPU smoke suite:
  `PYTHONPATH=src /home/davwis/miniconda3/envs/bnn_sim/bin/python scripts/run_shared_basis_lm.py --out runs/shared_basis_smoke --device cpu --vocab 16 --train-tokens 512 --val-tokens 128 --seq-len 8 --batch-size 4 --eval-batches 2 --steps 5 --dim 16 --heads 4 --depth 2 --groups 4 --rank 1 --expansion 2`.
  The shared-basis MLP used 80 parameters versus 512 for the unshared grouped
  MLP, a ratio of 0.15625. The smoke artifacts were written under
  `runs/shared_basis_smoke/` and are intentionally ignored by git.

- USER (2026-07-03): Pivoted from the butterfly-matrix idea to a simpler
  shared-basis parameterization across depth and width. Requested an
  experiment, not a benchmark-maximization push.

- DECISION (2026-07-03): Use `irregular-parameter-sharing` as the project and
  keep the existing `docs/butterfly-sharing/` arc unchanged as prior work.
  Start a separate `docs/shared-basis-sharing/` task arc for the cleaner
  mechanism.
