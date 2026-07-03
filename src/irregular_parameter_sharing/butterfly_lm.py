from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path("/home/davwis/main")
DATA = ROOT / "data"
ROBERTA_TOKENIZER = Path(
    "/home/davwis/.cache/huggingface/hub/models--roberta-base/"
    "snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(base: int, name: str) -> int:
    value = base * 1_000_003
    for i, ch in enumerate(name):
        value += (i + 1) * ord(ch)
    return value % (2**31 - 1)


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_params(model: nn.Module, include_embedding: bool = True) -> int:
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not include_embedding and ("tok_emb" in name or "pos_emb" in name):
            continue
        total += param.numel()
    return total


def count_named_params(model: nn.Module, token: str) -> int:
    return sum(p.numel() for name, p in model.named_parameters() if token in name and p.requires_grad)


def prepare_openwebtext_tokens(
    cache_dir: Path,
    max_tokens: int,
    shard_limit: int,
    tokenizer_name: str,
    tokenizer_path: Path | None,
    data_dir: Path = DATA / "openwebtext" / "plain_text",
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_slug = tokenizer_name if tokenizer_name != "hf" else f"hf_{(tokenizer_path or ROBERTA_TOKENIZER).name}"
    token_path = cache_dir / f"openwebtext_{tokenizer_slug}_{max_tokens}.npy"
    meta_path = cache_dir / f"openwebtext_{tokenizer_slug}_{max_tokens}.json"
    if token_path.exists() and meta_path.exists():
        return token_path

    if tokenizer_name == "gpt2":
        enc = tiktoken.get_encoding("gpt2")

        def encode_text(value: str) -> list[int]:
            return enc.encode_ordinary(value)

        eot = enc.eot_token
        vocab_size = enc.n_vocab
    elif tokenizer_name == "hf":
        local_path = tokenizer_path or ROBERTA_TOKENIZER
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

        def encode_text(value: str) -> list[int]:
            return tokenizer.encode(value, add_special_tokens=False)

        eot = int(tokenizer.eos_token_id or tokenizer.sep_token_id or 2)
        vocab_size = int(tokenizer.vocab_size)
    else:
        raise ValueError(f"unknown tokenizer {tokenizer_name}")

    shards = sorted(data_dir.glob("*.parquet"))[:shard_limit]
    if not shards:
        raise FileNotFoundError(f"no OpenWebText parquet shards found under {data_dir}")

    tokens: list[int] = []
    docs = 0
    for shard in shards:
        frame = pd.read_parquet(shard, columns=["text"])
        for text in frame["text"].tolist():
            tokens.extend(encode_text(text))
            tokens.append(eot)
            docs += 1
            if len(tokens) >= max_tokens:
                arr = np.asarray(tokens[:max_tokens], dtype=np.uint16)
                np.save(token_path, arr)
                meta_path.write_text(
                    json.dumps(
                        {
                            "tokens": int(arr.size),
                            "docs": docs,
                            "shards": [str(s) for s in shards],
                            "tokenizer": tokenizer_name,
                            "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
                            "vocab_size": vocab_size,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return token_path

    arr = np.asarray(tokens, dtype=np.uint16)
    np.save(token_path, arr)
    meta_path.write_text(
        json.dumps(
            {
                "tokens": int(arr.size),
                "docs": docs,
                "shards": [str(s) for s in shards],
                "tokenizer": tokenizer_name,
                "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
                "vocab_size": vocab_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return token_path


def infer_vocab_size(args: argparse.Namespace, token_path: Path) -> int:
    if args.vocab_size:
        return int(args.vocab_size)
    meta_path = token_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("vocab_size"):
            return int(meta["vocab_size"])
    if args.tokenizer == "hf":
        tokenizer = AutoTokenizer.from_pretrained(Path(args.tokenizer_path), local_files_only=True)
        return int(tokenizer.vocab_size)
    return 50_257


def load_split(token_path: Path, val_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.load(token_path).astype(np.int64)
    val_tokens = min(val_tokens, max(1, len(arr) // 10))
    return torch.from_numpy(arr[:-val_tokens]), torch.from_numpy(arr[-val_tokens:])


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device, non_blocking=True)
    return x, y


def butterfly_pairs(groups: int, stage: int) -> list[tuple[int, int]]:
    stride = 2**stage
    pairs: list[tuple[int, int]] = []
    for base in range(0, groups, 2 * stride):
        for offset in range(stride):
            pairs.append((base + offset, base + stride + offset))
    return pairs


def all_positions(depth: int, groups: int) -> list[tuple[int, int, int]]:
    stages = int(math.log2(groups))
    return [(layer, stage, pair) for layer in range(depth) for stage in range(stages) for pair in range(groups // 2)]


def balanced_random_schedule(depth: int, groups: int, shared_blocks: int, seed: int) -> torch.Tensor:
    positions = all_positions(depth, groups)
    if shared_blocks > len(positions):
        raise ValueError("shared_blocks cannot exceed number of shareable positions")
    ids = list(range(shared_blocks))
    while len(ids) < len(positions):
        ids.extend(range(shared_blocks))
    ids = ids[: len(positions)]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return torch.tensor(ids, dtype=torch.long).reshape(depth, int(math.log2(groups)), groups // 2)


def unshared_schedule(depth: int, groups: int) -> torch.Tensor:
    positions = all_positions(depth, groups)
    return torch.arange(len(positions), dtype=torch.long).reshape(depth, int(math.log2(groups)), groups // 2)


def depth_tied_schedule(depth: int, groups: int) -> torch.Tensor:
    stages = int(math.log2(groups))
    width_positions = stages * (groups // 2)
    ids = []
    for _layer in range(depth):
        ids.extend(range(width_positions))
    return torch.tensor(ids, dtype=torch.long).reshape(depth, stages, groups // 2)


def max_depth_distance_schedule(depth: int, groups: int, shared_blocks: int) -> torch.Tensor:
    positions = all_positions(depth, groups)
    if shared_blocks > len(positions):
        raise ValueError("shared_blocks cannot exceed number of shareable positions")
    stages = int(math.log2(groups))

    def bit_reverse(value: int, bits: int) -> int:
        out = 0
        for _ in range(bits):
            out = (out << 1) | (value & 1)
            value >>= 1
        return out

    layer_order: list[int] = []
    for i in range((depth + 1) // 2):
        layer_order.append(i)
        if depth - 1 - i != i:
            layer_order.append(depth - 1 - i)
    slot_count = stages * (groups // 2)
    slot_bits = max(1, math.ceil(math.log2(slot_count)))
    slot_order = sorted(range(slot_count), key=lambda x: bit_reverse(x, slot_bits))
    ordered_positions = [
        (layer, slot // (groups // 2), slot % (groups // 2))
        for layer in layer_order
        for slot in slot_order
    ]

    assigned: list[list[tuple[int, int]]] = [[] for _ in range(shared_blocks)]
    counts = [0 for _ in range(shared_blocks)]
    mapping: dict[tuple[int, int, int], int] = {}
    for layer, stage, pair in ordered_positions:
        slot = stage * (groups // 2) + pair
        best_id = 0
        best_score = (-1.0, 0.0, 0.0)
        for block_id in range(shared_blocks):
            if not assigned[block_id]:
                min_depth = depth
                slot_diversity = slot_count
            else:
                min_depth = min(abs(layer - prev_layer) for prev_layer, _ in assigned[block_id])
                slot_diversity = min(abs(slot - prev_slot) for _, prev_slot in assigned[block_id])
            score = (float(min_depth), float(slot_diversity) / slot_count, -float(counts[block_id]))
            if score > best_score:
                best_score = score
                best_id = block_id
        assigned[best_id].append((layer, slot))
        counts[best_id] += 1
        mapping[(layer, stage, pair)] = best_id

    ids = [mapping[pos] for pos in positions]
    return torch.tensor(ids, dtype=torch.long).reshape(depth, stages, groups // 2)


def schedule_stats(schedule: torch.Tensor) -> dict:
    flat = schedule.flatten().tolist()
    depths: dict[int, list[int]] = {}
    for layer in range(schedule.shape[0]):
        for block_id in schedule[layer].flatten().tolist():
            depths.setdefault(int(block_id), []).append(layer)
    min_depth_gaps = []
    spans = []
    for layers in depths.values():
        unique = sorted(set(layers))
        if len(unique) > 1:
            spans.append(max(unique) - min(unique))
            min_depth_gaps.append(min(b - a for a, b in zip(unique, unique[1:])))
        else:
            spans.append(0)
            min_depth_gaps.append(0)
    return {
        "positions": len(flat),
        "shared_blocks": len(set(flat)),
        "mean_uses_per_block": float(np.mean([flat.count(i) for i in sorted(set(flat))])),
        "mean_depth_span": float(np.mean(spans)),
        "mean_min_depth_gap": float(np.mean(min_depth_gaps)),
    }


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must divide heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().reshape(batch, tokens, channels)
        return self.proj(y)


class ButterflyBank(nn.Module):
    def __init__(self, blocks: int, pair_dim: int, expansion: int) -> None:
        super().__init__()
        hidden = pair_dim * expansion
        scale1 = pair_dim**-0.5
        scale2 = hidden**-0.5
        self.w1 = nn.Parameter(torch.randn(blocks, pair_dim, hidden) * scale1)
        self.b1 = nn.Parameter(torch.zeros(blocks, hidden))
        self.w2 = nn.Parameter(torch.randn(blocks, hidden, pair_dim) * scale2)
        self.b2 = nn.Parameter(torch.zeros(blocks, pair_dim))

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # x: [batch, tokens, pairs, pair_dim], ids: [pairs]
        w1 = self.w1[ids]
        b1 = self.b1[ids]
        w2 = self.w2[ids]
        b2 = self.b2[ids]
        x_norm = F.layer_norm(x, (x.shape[-1],))
        h = torch.einsum("btpd,pdh->btph", x_norm, w1) + b1.view(1, 1, ids.numel(), -1)
        h = F.gelu(h)
        y = torch.einsum("btph,phd->btpd", h, w2) + b2.view(1, 1, ids.numel(), -1)
        return x + y


class ButterflyMix(nn.Module):
    def __init__(self, dim: int, groups: int, schedule: torch.Tensor, shared_blocks: int, expansion: int) -> None:
        super().__init__()
        if groups <= 1 or groups & (groups - 1):
            raise ValueError("groups must be a power of two greater than one")
        if dim % groups:
            raise ValueError("dim must divide groups")
        self.dim = dim
        self.groups = groups
        self.group_dim = dim // groups
        self.pair_dim = 2 * self.group_dim
        self.stages = int(math.log2(groups))
        self.register_buffer("schedule", schedule.clone().long(), persistent=False)
        self.stage_indices = [
            torch.tensor([idx for pair in butterfly_pairs(groups, stage) for idx in pair], dtype=torch.long)
            for stage in range(self.stages)
        ]
        self.bank = ButterflyBank(shared_blocks, self.pair_dim, expansion)

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch, tokens, _ = x.shape
        grouped = x.reshape(batch, tokens, self.groups, self.group_dim)
        for stage in range(self.stages):
            idx = self.stage_indices[stage].to(x.device)
            pair_count = self.groups // 2
            pairs = grouped.index_select(2, idx).reshape(batch, tokens, pair_count, self.pair_dim)
            ids = self.schedule[layer_idx, stage].to(x.device)
            pairs = self.bank(pairs, ids)
            grouped = grouped.clone()
            grouped[:, :, idx, :] = pairs.reshape(batch, tokens, self.groups, self.group_dim)
        return grouped.reshape(batch, tokens, self.dim)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, butterfly: ButterflyMix, layer_idx: int) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(butterfly(self.ln2(x), layer_idx))
        return x


class ButterflyGPT(nn.Module):
    def __init__(
        self,
        vocab: int,
        seq_len: int,
        dim: int,
        heads: int,
        depth: int,
        groups: int,
        schedule: torch.Tensor,
        shared_blocks: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.layers = nn.ModuleList([TransformerLayer(dim, heads, dropout) for _ in range(depth)])
        self.butterfly = ButterflyMix(dim, groups, schedule, shared_blocks, expansion)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, tokens = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :tokens]
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, self.butterfly, layer_idx)
        return self.head(self.ln_f(x))


@dataclass
class TrainResult:
    final_val_loss: float
    final_train_loss: float
    history: list[dict]
    seconds: float
    params_total: int
    params_nonembedding: int
    params_butterfly: int


@torch.no_grad()
def eval_lm(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    steps: int,
    seed: int,
) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        x, y = get_batch(data, batch_size, seq_len, device, gen)
        logits = model(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
    return float(np.mean(losses))


def train_lm(
    model: ButterflyGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    steps: int,
    seed: int,
    run_name: str,
) -> TrainResult:
    start = time.time()
    device = device_from_arg(args.device)
    model.to(device)
    if args.compile:
        model = torch.compile(model)  # type: ignore[assignment]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    gen = torch.Generator().manual_seed(seed + 17)
    history: list[dict] = []
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.seq_len, device, gen)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) / args.grad_accum
            scaler.scale(loss).backward()
            loss_accum += float(loss.detach())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        if step % args.eval_every == 0 or step == steps:
            val_loss = eval_lm(
                model,
                val_data,
                args.eval_batch_size,
                args.seq_len,
                device,
                args.eval_steps,
                seed + 99,
            )
            history.append(
                {
                    "run": run_name,
                    "step": step,
                    "train_loss": loss_accum,
                    "val_loss": val_loss,
                }
            )
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    return TrainResult(
        final_val_loss=history[-1]["val_loss"],
        final_train_loss=history[-1]["train_loss"],
        history=history,
        seconds=time.time() - start,
        params_total=count_params(raw_model, include_embedding=True),
        params_nonembedding=count_params(raw_model, include_embedding=False),
        params_butterfly=count_named_params(raw_model, "butterfly.bank"),
    )


def make_model(
    args: argparse.Namespace,
    vocab_size: int,
    schedule: torch.Tensor,
    shared_blocks: int,
    init_seed: int,
) -> ButterflyGPT:
    set_seed(init_seed)
    return ButterflyGPT(
        vocab=vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        heads=args.heads,
        depth=args.depth,
        groups=args.groups,
        schedule=schedule,
        shared_blocks=shared_blocks,
        expansion=args.butterfly_expansion,
        dropout=args.dropout,
    )


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def summarize_runs(runs: list[dict]) -> list[dict]:
    variants = sorted({run["variant"] for run in runs})
    rows = []
    for variant in variants:
        selected = [run for run in runs if run["variant"] == variant]
        mean, sd = mean_sd(run["val_loss"] for run in selected)
        rows.append(
            {
                "variant": variant,
                "runs": len(selected),
                "val_loss_mean": mean,
                "val_loss_sd": sd,
                "params_total": selected[0]["params_total"],
                "params_nonembedding": selected[0]["params_nonembedding"],
                "params_butterfly": selected[0]["params_butterfly"],
            }
        )
    return rows


def write_outputs(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "runs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "seed",
                "layout_seed",
                "val_loss",
                "train_loss",
                "seconds",
                "params_total",
                "params_nonembedding",
                "params_butterfly",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(result["runs"])
    lines = [
        "# butterfly depth-width sharing",
        "",
        f"Status: {result['status']}",
        f"Device: {result['device']}",
        f"Seconds: {result['seconds']:.2f}",
        "",
        "## Configuration",
        "",
        f"- depth: {result['config']['depth']}",
        f"- dim: {result['config']['dim']}",
        f"- groups: {result['config']['groups']}",
        f"- shareable positions: {result['position_count']}",
        f"- shared blocks: {result['config']['shared_blocks']}",
        f"- search method: best of {result['config']['search_budget']} random balanced layouts after {result['config']['search_steps']} proxy steps",
        "",
        "## Results",
        "",
        "| Variant | Runs | Val CE mean | Val CE sd | Total params | Non-embed params | Butterfly params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['val_loss_mean']:.4f} | "
            f"{row['val_loss_sd']:.4f} | {row['params_total']} | {row['params_nonembedding']} | "
            f"{row['params_butterfly']} |"
        )
    if result.get("decision"):
        lines += ["", "## Decision", "", result["decision"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_event(out: Path, event: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "live.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), **event}) + "\n")


def run_suite(args: argparse.Namespace) -> None:
    start = time.time()
    device = device_from_arg(args.device)
    torch.set_float32_matmul_precision("high")
    out = Path(args.out)
    append_event(out, {"event": "start", "config": vars(args), "device": str(device)})
    if args.token_path:
        token_path = Path(args.token_path)
    else:
        token_path = prepare_openwebtext_tokens(
            Path(args.cache_dir),
            args.max_tokens,
            args.shard_limit,
            args.tokenizer,
            Path(args.tokenizer_path) if args.tokenizer_path else None,
        )
    train_data, val_data = load_split(token_path, args.val_tokens)
    vocab_size = infer_vocab_size(args, token_path)
    append_event(
        out,
        {
            "event": "data_ready",
            "token_path": str(token_path),
            "train_tokens": int(train_data.numel()),
            "val_tokens": int(val_data.numel()),
            "vocab_size": vocab_size,
        },
    )

    position_count = args.depth * int(math.log2(args.groups)) * (args.groups // 2)
    if args.shared_blocks >= position_count:
        raise ValueError(f"shared_blocks={args.shared_blocks} must be smaller than positions={position_count}")

    search_records = []
    best_schedule = None
    best_layout_seed = None
    if args.search_budget:
        for trial in range(args.search_budget):
            layout_seed = args.search_seed + trial
            schedule = balanced_random_schedule(args.depth, args.groups, args.shared_blocks, layout_seed)
            model = make_model(args, vocab_size, schedule, args.shared_blocks, stable_seed(args.seed, f"search-{trial}"))
            result = train_lm(
                model,
                train_data,
                val_data,
                args,
                args.search_steps,
                stable_seed(args.seed, f"search-train-{trial}"),
                f"search_random_{trial}",
            )
            search_records.append(
                {
                    "trial": trial,
                    "layout_seed": layout_seed,
                    "val_loss": result.final_val_loss,
                    "train_loss": result.final_train_loss,
                    "seconds": result.seconds,
                    "schedule_stats": schedule_stats(schedule),
                    "schedule": schedule.tolist(),
                }
            )
            append_event(out, {"event": "search_trial_complete", **search_records[-1]})
        best = min(search_records, key=lambda r: r["val_loss"])
        best_schedule = torch.tensor(best["schedule"], dtype=torch.long)
        best_layout_seed = int(best["layout_seed"])
        append_event(
            out,
            {
                "event": "search_complete",
                "best_trial": best["trial"],
                "best_layout_seed": best_layout_seed,
                "best_val_loss": best["val_loss"],
            },
        )

    max_schedule = max_depth_distance_schedule(args.depth, args.groups, args.shared_blocks)
    unshared = unshared_schedule(args.depth, args.groups)
    depth_tied = depth_tied_schedule(args.depth, args.groups)
    final_seeds = [int(s) for s in args.final_seeds.split(",") if s.strip()]

    runs: list[dict] = []

    def run_variant(variant: str, seed: int, schedule: torch.Tensor, blocks: int, layout_seed: int | None) -> None:
        init_seed = stable_seed(seed, f"{variant}-init-{blocks}")
        train_seed = stable_seed(seed, f"{variant}-train")
        model = make_model(args, vocab_size, schedule, blocks, init_seed)
        result = train_lm(model, train_data, val_data, args, args.final_steps, train_seed, variant)
        runs.append(
            {
                "variant": variant,
                "seed": seed,
                "layout_seed": layout_seed,
                "val_loss": result.final_val_loss,
                "train_loss": result.final_train_loss,
                "seconds": result.seconds,
                "params_total": result.params_total,
                "params_nonembedding": result.params_nonembedding,
                "params_butterfly": result.params_butterfly,
                "schedule_stats": schedule_stats(schedule),
            }
        )
        append_event(out, {"event": "final_run_complete", **runs[-1]})

    requested = set(args.final_variants)
    for seed in final_seeds:
        if "unshared" in requested:
            run_variant("unshared", seed, unshared, position_count, None)
        if "depth_tied" in requested:
            run_variant("depth_tied", seed, depth_tied, int(depth_tied.max().item()) + 1, None)
        if "random" in requested:
            layout_seed = args.random_final_seed + seed
            schedule = balanced_random_schedule(args.depth, args.groups, args.shared_blocks, layout_seed)
            run_variant("random_avg", seed, schedule, args.shared_blocks, layout_seed)
        if "max_distance" in requested:
            run_variant("max_distance", seed, max_schedule, args.shared_blocks, None)
        if "best_random" in requested and best_schedule is not None:
            run_variant("best_of_random", seed, best_schedule, args.shared_blocks, best_layout_seed)

    summary = summarize_runs(runs)
    by_variant = {row["variant"]: row for row in summary}
    decision = ""
    if "max_distance" in by_variant and "random_avg" in by_variant:
        delta = by_variant["random_avg"]["val_loss_mean"] - by_variant["max_distance"]["val_loss_mean"]
        decision = f"max_distance mean CE gain over random_avg is {delta:.4f} if positive."
    result = {
        "status": "completed",
        "device": str(device),
        "seconds": time.time() - start,
        "token_path": str(token_path),
        "train_tokens": int(train_data.numel()),
        "val_tokens": int(val_data.numel()),
        "vocab_size": vocab_size,
        "position_count": position_count,
        "config": vars(args),
        "search_records": search_records,
        "best_layout_seed": best_layout_seed,
        "schedules": {
            "max_distance": max_schedule.tolist(),
            "unshared": unshared.tolist(),
            "depth_tied": depth_tied.tolist(),
            "best_random": best_schedule.tolist() if best_schedule is not None else None,
        },
        "schedule_stats": {
            "max_distance": schedule_stats(max_schedule),
            "unshared": schedule_stats(unshared),
            "depth_tied": schedule_stats(depth_tied),
            "best_random": schedule_stats(best_schedule) if best_schedule is not None else None,
        },
        "runs": runs,
        "summary": summary,
        "decision": decision,
    }
    write_outputs(out, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default="/home/davwis/main/workspace/irregular-parameter-sharing/runs/cache")
    parser.add_argument("--token-path")
    parser.add_argument("--tokenizer", choices=["gpt2", "hf"], default="hf")
    parser.add_argument("--tokenizer-path", default=str(ROBERTA_TOKENIZER))
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--shard-limit", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--final-steps", type=int, default=5000)
    parser.add_argument("--search-steps", type=int, default=800)
    parser.add_argument("--search-budget", type=int, default=12)
    parser.add_argument("--search-seed", type=int, default=10_000)
    parser.add_argument("--random-final-seed", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--shared-blocks", type=int, default=128)
    parser.add_argument("--butterfly-expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--final-seeds", default="0,1,2")
    parser.add_argument(
        "--final-variants",
        nargs="+",
        default=["unshared", "random", "max_distance", "best_random"],
        choices=["unshared", "depth_tied", "random", "max_distance", "best_random"],
    )
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
