from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from irregular_parameter_sharing.butterfly_lm import (
    ROBERTA_TOKENIZER,
    CausalSelfAttention,
    append_event,
    count_named_params,
    count_params,
    device_from_arg,
    eval_lm,
    get_batch,
    infer_vocab_size,
    load_split,
    prepare_openwebtext_tokens,
    set_seed,
    stable_seed,
)


def all_positions(depth: int, chunks: int) -> list[tuple[int, int]]:
    return [(layer, chunk) for layer in range(depth) for chunk in range(chunks)]


def balanced_random_schedule(depth: int, chunks: int, shared_blocks: int, seed: int) -> torch.Tensor:
    positions = all_positions(depth, chunks)
    if shared_blocks > len(positions):
        raise ValueError("shared_blocks cannot exceed the number of shareable positions")
    ids = list(range(shared_blocks))
    while len(ids) < len(positions):
        ids.extend(range(shared_blocks))
    ids = ids[: len(positions)]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return torch.tensor(ids, dtype=torch.long).reshape(depth, chunks)


def unshared_schedule(depth: int, chunks: int) -> torch.Tensor:
    return torch.arange(depth * chunks, dtype=torch.long).reshape(depth, chunks)


def depth_tied_schedule(depth: int, chunks: int) -> torch.Tensor:
    ids = [chunk for _layer in range(depth) for chunk in range(chunks)]
    return torch.tensor(ids, dtype=torch.long).reshape(depth, chunks)


def max_depth_distance_schedule(depth: int, chunks: int, shared_blocks: int) -> torch.Tensor:
    positions = all_positions(depth, chunks)
    if shared_blocks > len(positions):
        raise ValueError("shared_blocks cannot exceed the number of shareable positions")

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
    chunk_bits = max(1, math.ceil(math.log2(chunks)))
    chunk_order = sorted(range(chunks), key=lambda x: bit_reverse(x, chunk_bits))
    ordered_positions = [(layer, chunk) for layer in layer_order for chunk in chunk_order]

    assigned: list[list[tuple[int, int]]] = [[] for _ in range(shared_blocks)]
    counts = [0 for _ in range(shared_blocks)]
    mapping: dict[tuple[int, int], int] = {}
    for layer, chunk in ordered_positions:
        best_id = 0
        best_score = (-1.0, 0.0, 0.0)
        for block_id in range(shared_blocks):
            if not assigned[block_id]:
                min_depth_gap = depth
                min_chunk_gap = chunks
            else:
                min_depth_gap = min(abs(layer - prev_layer) for prev_layer, _ in assigned[block_id])
                min_chunk_gap = min(abs(chunk - prev_chunk) for _, prev_chunk in assigned[block_id])
            score = (float(min_depth_gap), float(min_chunk_gap) / chunks, -float(counts[block_id]))
            if score > best_score:
                best_score = score
                best_id = block_id
        assigned[best_id].append((layer, chunk))
        counts[best_id] += 1
        mapping[(layer, chunk)] = best_id

    ids = [mapping[position] for position in positions]
    return torch.tensor(ids, dtype=torch.long).reshape(depth, chunks)


def schedule_stats(schedule: torch.Tensor) -> dict:
    flat = schedule.flatten().tolist()
    block_positions: dict[int, list[tuple[int, int]]] = {}
    for layer in range(schedule.shape[0]):
        for chunk in range(schedule.shape[1]):
            block_positions.setdefault(int(schedule[layer, chunk]), []).append((layer, chunk))
    depth_spans = []
    min_depth_gaps = []
    chunk_spans = []
    for positions in block_positions.values():
        layers = sorted({layer for layer, _ in positions})
        chunks = sorted({chunk for _, chunk in positions})
        depth_spans.append(max(layers) - min(layers) if len(layers) > 1 else 0)
        min_depth_gaps.append(min((b - a for a, b in zip(layers, layers[1:])), default=0))
        chunk_spans.append(max(chunks) - min(chunks) if len(chunks) > 1 else 0)
    return {
        "positions": len(flat),
        "shared_blocks": len(set(flat)),
        "mean_uses_per_block": float(np.mean([flat.count(i) for i in sorted(set(flat))])),
        "mean_depth_span": float(np.mean(depth_spans)),
        "mean_min_depth_gap": float(np.mean(min_depth_gaps)),
        "mean_chunk_span": float(np.mean(chunk_spans)),
    }


class HardBlockMLPBank(nn.Module):
    """Bank of exact MLP hidden-chunk modules selected by a hard schedule."""

    def __init__(self, blocks: int, dim: int, chunk_hidden: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(blocks, dim, chunk_hidden))
        self.w2 = nn.Parameter(torch.empty(blocks, chunk_hidden, dim))
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        w1 = self.w1[ids]
        w2 = self.w2[ids]
        h = torch.einsum("btd,cdh->btch", x, w1)
        h = self.drop(F.gelu(h))
        return torch.einsum("btch,chd->btd", h, w2)


class HardBlockTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mlp_bank: HardBlockMLPBank, ids: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(mlp_bank(self.ln2(x), ids))
        return x


class HardBlockGPT(nn.Module):
    def __init__(
        self,
        vocab: int,
        seq_len: int,
        dim: int,
        heads: int,
        depth: int,
        chunks: int,
        schedule: torch.Tensor,
        shared_blocks: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if expansion * dim % chunks:
            raise ValueError("expansion * dim must be divisible by chunks")
        self.seq_len = seq_len
        self.depth = depth
        self.chunks = chunks
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.layers = nn.ModuleList([HardBlockTransformerLayer(dim, heads, dropout) for _ in range(depth)])
        self.register_buffer("schedule", schedule.clone().long(), persistent=False)
        self.mlp_bank = HardBlockMLPBank(shared_blocks, dim, expansion * dim // chunks, dropout)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        self.mlp_bank.reset_parameters()
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, tokens = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :tokens]
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, self.mlp_bank, self.schedule[layer_idx].to(x.device))
        return self.head(self.ln_f(x))


@dataclass
class TrainResult:
    final_val_loss: float
    final_train_loss: float
    history: list[dict]
    seconds: float
    params_total: int
    params_nonembedding: int
    params_mlp_bank: int


def train_lm(
    model: HardBlockGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    steps: int,
    seed: int,
    run_name: str,
    out: Path | None = None,
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
            event = {
                "run": run_name,
                "step": step,
                "train_loss": loss_accum,
                "val_loss": val_loss,
                "seconds": time.time() - start,
            }
            history.append(event)
            if out is not None:
                append_event(out, {"event": "eval", **event})
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    return TrainResult(
        final_val_loss=history[-1]["val_loss"],
        final_train_loss=history[-1]["train_loss"],
        history=history,
        seconds=time.time() - start,
        params_total=count_params(raw_model, include_embedding=True),
        params_nonembedding=count_params(raw_model, include_embedding=False),
        params_mlp_bank=count_named_params(raw_model, "mlp_bank"),
    )


def make_model(
    args: argparse.Namespace,
    vocab_size: int,
    schedule: torch.Tensor,
    shared_blocks: int,
    init_seed: int,
) -> HardBlockGPT:
    set_seed(init_seed)
    return HardBlockGPT(
        vocab=vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        heads=args.heads,
        depth=args.depth,
        chunks=args.chunks,
        schedule=schedule,
        shared_blocks=shared_blocks,
        expansion=args.expansion,
        dropout=args.dropout,
    )


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def summarize_runs(runs: list[dict], variant_order: list[str]) -> list[dict]:
    rows = []
    for variant in variant_order:
        selected = [run for run in runs if run["variant"] == variant or run["variant_base"] == variant]
        if not selected:
            continue
        mean, sd = mean_sd([run["val_loss"] for run in selected])
        rows.append(
            {
                "variant": variant,
                "runs": len(selected),
                "val_loss_mean": mean,
                "val_loss_sd": sd,
                "params_total": selected[0]["params_total"],
                "params_nonembedding": selected[0]["params_nonembedding"],
                "params_mlp_bank": selected[0]["params_mlp_bank"],
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
                "variant_base",
                "seed",
                "layout_seed",
                "val_loss",
                "train_loss",
                "seconds",
                "params_total",
                "params_nonembedding",
                "params_mlp_bank",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(result["runs"])
    lines = [
        "# hard irregular MLP chunk sharing",
        "",
        f"Status: {result['status']}",
        f"Device: {result['device']}",
        f"Seconds: {result['seconds']:.2f}",
        "",
        "## Configuration",
        "",
        f"- depth: {result['config']['depth']}",
        f"- dim: {result['config']['dim']}",
        f"- hidden chunks per layer: {result['config']['chunks']}",
        f"- shareable positions: {result['position_count']}",
        f"- shared blocks: {result['config']['shared_blocks']}",
        f"- search method: best of {result['config']['search_budget']} balanced random hard-sharing layouts after {result['config']['search_steps']} proxy steps",
        f"- final steps: {result['config']['final_steps']}",
        "",
        "## Results",
        "",
        "| Variant | Runs | Val CE mean | Val CE sd | Total params | Non-embed params | MLP-bank params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['val_loss_mean']:.4f} | "
            f"{row['val_loss_sd']:.4f} | {row['params_total']} | {row['params_nonembedding']} | "
            f"{row['params_mlp_bank']} |"
        )
    if result.get("decision"):
        lines += ["", "## Decision", "", result["decision"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    position_count = args.depth * args.chunks
    if args.shared_blocks >= position_count:
        raise ValueError(f"shared_blocks={args.shared_blocks} must be smaller than positions={position_count}")

    search_records = []
    best_schedule = None
    best_layout_seed = None
    if args.search_budget:
        for trial in range(args.search_budget):
            layout_seed = args.search_seed + trial
            schedule = balanced_random_schedule(args.depth, args.chunks, args.shared_blocks, layout_seed)
            model = make_model(args, vocab_size, schedule, args.shared_blocks, stable_seed(args.seed, f"search-{trial}"))
            result = train_lm(
                model,
                train_data,
                val_data,
                args,
                args.search_steps,
                stable_seed(args.seed, f"search-train-{trial}"),
                f"search_random_{trial}",
                out,
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
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
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

    schedules = {
        "unshared": unshared_schedule(args.depth, args.chunks),
        "depth_tied": depth_tied_schedule(args.depth, args.chunks),
        "max_distance": max_depth_distance_schedule(args.depth, args.chunks, args.shared_blocks),
    }
    final_seeds = [int(s) for s in args.final_seeds.split(",") if s.strip()]
    runs: list[dict] = []

    def run_variant(
        variant: str,
        variant_base: str,
        seed: int,
        schedule: torch.Tensor,
        blocks: int,
        layout_seed: int | None,
    ) -> None:
        model = make_model(args, vocab_size, schedule, blocks, stable_seed(seed, f"{variant}-init-{blocks}"))
        result = train_lm(
            model,
            train_data,
            val_data,
            args,
            args.final_steps,
            stable_seed(seed, f"{variant}-train"),
            variant,
            out,
        )
        runs.append(
            {
                "variant": variant,
                "variant_base": variant_base,
                "seed": seed,
                "layout_seed": layout_seed,
                "val_loss": result.final_val_loss,
                "train_loss": result.final_train_loss,
                "seconds": result.seconds,
                "params_total": result.params_total,
                "params_nonembedding": result.params_nonembedding,
                "params_mlp_bank": result.params_mlp_bank,
                "schedule_stats": schedule_stats(schedule),
            }
        )
        append_event(out, {"event": "final_run_complete", **runs[-1]})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    requested = set(args.final_variants)
    for seed in final_seeds:
        if "unshared" in requested:
            run_variant("unshared", "unshared", seed, schedules["unshared"], position_count, None)
        if "depth_tied" in requested:
            run_variant("depth_tied", "depth_tied", seed, schedules["depth_tied"], args.chunks, None)
        if "random" in requested:
            layout_seed = args.random_final_seed + seed
            schedule = balanced_random_schedule(args.depth, args.chunks, args.shared_blocks, layout_seed)
            run_variant("random_avg", "random", seed, schedule, args.shared_blocks, layout_seed)
        if "max_distance" in requested:
            run_variant("max_distance", "max_distance", seed, schedules["max_distance"], args.shared_blocks, None)
        if "best_random" in requested and best_schedule is not None:
            run_variant("best_of_random", "best_random", seed, best_schedule, args.shared_blocks, best_layout_seed)

    summary = summarize_runs(runs, args.final_variants)
    by_variant = {row["variant"]: row for row in summary}
    decision = ""
    if "max_distance" in by_variant and "random" in by_variant:
        delta = by_variant["random"]["val_loss_mean"] - by_variant["max_distance"]["val_loss_mean"]
        decision = f"max_distance mean CE gain over random_avg is {delta:.4f}."
    if "best_random" in by_variant and "random" in by_variant:
        delta = by_variant["random"]["val_loss_mean"] - by_variant["best_random"]["val_loss_mean"]
        decision = (decision + " " if decision else "") + f"best_of_random mean CE gain over random_avg is {delta:.4f}."
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
        "runs": runs,
        "summary": summary,
        "schedules": {
            "unshared": schedules["unshared"].tolist(),
            "depth_tied": schedules["depth_tied"].tolist(),
            "max_distance": schedules["max_distance"].tolist(),
            "best_random": best_schedule.tolist() if best_schedule is not None else None,
        },
        "schedule_stats": {
            "unshared": schedule_stats(schedules["unshared"]),
            "depth_tied": schedule_stats(schedules["depth_tied"]),
            "max_distance": schedule_stats(schedules["max_distance"]),
            "best_random": schedule_stats(best_schedule) if best_schedule is not None else None,
        },
        "decision": decision,
    }
    write_outputs(out, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default="/home/davwis/main/workspace/irregular-parameter-sharing/runs/cache")
    parser.add_argument("--token-path")
    parser.add_argument("--tokenizer", choices=["gpt2", "hf"], default="gpt2")
    parser.add_argument("--tokenizer-path", default=str(ROBERTA_TOKENIZER))
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--shard-limit", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--search-steps", type=int, default=800)
    parser.add_argument("--final-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--shared-blocks", type=int, default=128)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--search-budget", type=int, default=12)
    parser.add_argument("--search-seed", type=int, default=10_000)
    parser.add_argument("--random-final-seed", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-seeds", default="0,1,2")
    parser.add_argument(
        "--final-variants",
        nargs="+",
        default=["unshared", "random", "max_distance", "best_random"],
    )
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
