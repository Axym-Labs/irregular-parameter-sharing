# Shared-Basis Depth-Width Sharing Progress

- CLEANUP (2026-07-03): User clarified the shared-basis messages were meant
  for another place and that this repository is GPU-only. Removed the CPU
  device path from the shared-basis toy runner by making `device='cuda'` the
  only accepted mode and changing the smoke test/docs away from CPU.

- VERIFICATION (2026-07-03): Added `SharedBasisGroupedMLP`, `UnsharedGroupedMLP`,
  `SharedBasisGPT`, `UnsharedGroupedGPT`, and a random-token toy suite. Verified
  with `/home/davwis/miniconda3/envs/bnn_sim/bin/python -m pytest -q`
  (`3 passed`) and compile check
  `/home/davwis/miniconda3/envs/bnn_sim/bin/python -m compileall -q src scripts tests`.

- EXPERIMENT (2026-07-03): Ran a pre-clarification toy smoke suite. The
  shared-basis MLP used 80 parameters versus 512 for the unshared grouped MLP,
  a ratio of 0.15625. The ignored smoke artifacts were written under
  `runs/shared_basis_smoke/`; future runs in this repo should use CUDA only.

- USER (2026-07-03): Pivoted from the butterfly-matrix idea to a simpler
  shared-basis parameterization across depth and width. Requested an
  experiment, not a benchmark-maximization push.

- DECISION (2026-07-03): Use `irregular-parameter-sharing` as the project and
  keep the existing `docs/butterfly-sharing/` arc unchanged as prior work.
  Start a separate `docs/shared-basis-sharing/` task arc for the cleaner
  mechanism.
