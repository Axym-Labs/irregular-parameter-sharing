# Butterfly Sharing Experiment Progress

- USER (2026-07-03): Rejected the previous depth-only sharing test as too weak.
  Requested a real GitHub repo named by the published post, a butterfly-block
  sharing experiment across depth and width, random sharing averages over three
  seeds, a best-of-N search budget/method, a maximum-depth-distance sharing
  layout, a larger RTX 5090-scale run, and a revised single results table.
- INITIATION (2026-07-03): Found the published post links
  `https://github.com/Axym-Labs/irregular-parameter-sharing`. GitHub connector
  returns 404 for that repo. Local `gh` exists but has an invalid token for
  `DavideWiest`, so public repo creation is blocked unless another authenticated
  path is available. Proceeding with local repo scaffold first.
- INITIATION (2026-07-03): Initialized local git repo at
  `/home/davwis/main/workspace/irregular-parameter-sharing` with remote
  `https://github.com/Axym-Labs/irregular-parameter-sharing.git`. Attempted
  `gh repo create Axym-Labs/irregular-parameter-sharing --public`; it failed
  with `error connecting to api.github.com` because shell network access is
  restricted in this session.
- IMPLEMENTATION (2026-07-03): Added a vectorized butterfly-block Transformer
  harness. Each layer has multiple shareable width-pair MLP blocks from a
  global bank, with schedules mapping `(depth, butterfly stage, width pair)` to
  block IDs. Implemented balanced random sharing, best-of-N random search,
  depth-tied sharing, unshared butterfly reference, and max-depth-distance
  sharing.
- DEBUGGING (2026-07-03): First smoke failed because `tiktoken` tried to fetch
  GPT-2 assets from the network. Root cause: new repo cache path did not contain
  tokenizer assets and shell network is restricted. Fixed by adding `--token-path`,
  `--vocab-size`, and offline Hugging Face tokenizer support.
- VERIFICATION (2026-07-03): Smoke test passed with the existing 200k GPT-2
  token cache: `runs/smoke/summary.md`, `runs/smoke/result.json`, and
  `runs/smoke/runs.csv` were written successfully. The toy max-distance layout
  had higher mean depth span than the selected random layout.
- BLOCKER (2026-07-03): Large RTX 5090 run is blocked in this session because
  PyTorch cannot see CUDA: `torch.cuda.is_available() == False`,
  `torch.cuda.device_count() == 0`, and `/dev/nvidia*` is not exposed inside the
  sandbox, although `nvidia-smi` can query the host GPU.
