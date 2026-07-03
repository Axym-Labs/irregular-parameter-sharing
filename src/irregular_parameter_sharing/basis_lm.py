from __future__ import annotations

import argparse
import csv
import json
import math
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


class DenseMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
        del layer_idx
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class BasisFieldMLP(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        hidden: int,
        rank: int,
        dropout: float,
        coeff_init: float,
        basis_scale: float,
        shared_component: bool,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.rank = rank
        self.shared_component = shared_component
        self.drop = nn.Dropout(dropout)
        if shared_component:
            self.fc1_shared = nn.Linear(dim, hidden, bias=False)
            self.fc2_shared = nn.Linear(hidden, dim, bias=False)
        else:
            self.fc1_shared = None
            self.fc2_shared = None

        self.u1 = nn.Parameter(torch.randn(rank, dim) / math.sqrt(dim))
        self.v1 = nn.Parameter(torch.randn(rank, hidden) / math.sqrt(rank))
        self.u2 = nn.Parameter(torch.randn(rank, hidden) / math.sqrt(hidden))
        self.v2 = nn.Parameter(torch.randn(rank, dim) / math.sqrt(rank))
        self.coeff1 = nn.Parameter(torch.randn(depth, rank) * coeff_init)
        self.coeff2 = nn.Parameter(torch.randn(depth, rank) * coeff_init)
        self.basis_scale = basis_scale

    def basis_linear(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        coeff: torch.Tensor,
    ) -> torch.Tensor:
        projected = torch.einsum("btd,rd->btr", x, u)
        weighted = projected * coeff.view(1, 1, -1)
        return torch.einsum("btr,rh->bth", weighted, v) * self.basis_scale

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.shared_component:
            h = self.fc1_shared(x)
        else:
            h = torch.zeros(*x.shape[:-1], self.v1.shape[1], device=x.device, dtype=x.dtype)
        h = h + self.basis_linear(x, self.u1, self.v1, self.coeff1[layer_idx])
        h = self.drop(F.gelu(h))
        if self.shared_component:
            y = self.fc2_shared(h)
        else:
            y = torch.zeros(*h.shape[:-1], self.v2.shape[1], device=h.device, dtype=h.dtype)
        y = y + self.basis_linear(h, self.u2, self.v2, self.coeff2[layer_idx])
        return y


class LowRankResidualMLP(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        hidden: int,
        rank: int,
        dropout: float,
        coeff_init: float,
        basis_scale: float,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.rank = rank
        self.fc1_shared = nn.Linear(dim, hidden, bias=False)
        self.fc2_shared = nn.Linear(hidden, dim, bias=False)
        self.u1 = nn.Parameter(torch.randn(depth, rank, dim) / math.sqrt(dim))
        self.v1 = nn.Parameter(torch.randn(depth, rank, hidden) / math.sqrt(rank))
        self.u2 = nn.Parameter(torch.randn(depth, rank, hidden) / math.sqrt(hidden))
        self.v2 = nn.Parameter(torch.randn(depth, rank, dim) / math.sqrt(rank))
        self.residual_scale = coeff_init * basis_scale
        self.drop = nn.Dropout(dropout)

    def lowrank_linear(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("btd,rd->btr", x, u)
        return torch.einsum("btr,rh->bth", projected, v) * self.residual_scale

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        h = self.fc1_shared(x) + self.lowrank_linear(x, self.u1[layer_idx], self.v1[layer_idx])
        h = self.drop(F.gelu(h))
        return self.fc2_shared(h) + self.lowrank_linear(h, self.u2[layer_idx], self.v2[layer_idx])


class BasisTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, mlp: nn.Module | None = None) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = mlp
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, layer_idx: int, shared_mlp: nn.Module | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        mlp = self.mlp if self.mlp is not None else shared_mlp
        if mlp is None:
            raise RuntimeError("MLP module missing")
        x = x + self.drop(mlp(self.ln2(x), layer_idx))
        return x


class BasisGPT(nn.Module):
    def __init__(
        self,
        vocab: int,
        seq_len: int,
        dim: int,
        heads: int,
        depth: int,
        mlp_kind: str,
        basis_rank: int,
        dropout: float,
        coeff_init: float,
        basis_scale: float,
        shared_component: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.depth = depth
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        hidden = 4 * dim
        if mlp_kind == "unshared":
            self.shared_mlp = None
            self.layers = nn.ModuleList(
                [BasisTransformerLayer(dim, heads, dropout, DenseMLP(dim, hidden, dropout)) for _ in range(depth)]
            )
        elif mlp_kind == "shared":
            self.shared_mlp = DenseMLP(dim, hidden, dropout)
            self.layers = nn.ModuleList([BasisTransformerLayer(dim, heads, dropout) for _ in range(depth)])
        elif mlp_kind == "basis":
            self.shared_mlp = BasisFieldMLP(
                depth,
                dim,
                hidden,
                basis_rank,
                dropout,
                coeff_init,
                basis_scale,
                shared_component,
            )
            self.layers = nn.ModuleList([BasisTransformerLayer(dim, heads, dropout) for _ in range(depth)])
        elif mlp_kind == "lowrank":
            self.shared_mlp = LowRankResidualMLP(depth, dim, hidden, basis_rank, dropout, coeff_init, basis_scale)
            self.layers = nn.ModuleList([BasisTransformerLayer(dim, heads, dropout) for _ in range(depth)])
        else:
            raise ValueError(f"unknown mlp_kind {mlp_kind}")
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
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
            x = layer(x, layer_idx, self.shared_mlp)
        return self.head(self.ln_f(x))


@dataclass
class TrainResult:
    final_val_loss: float
    final_train_loss: float
    history: list[dict]
    seconds: float
    params_total: int
    params_nonembedding: int
    params_mlp: int
    params_basis: int
    params_lowrank: int
    params_factorized: int


def train_lm(
    model: BasisGPT,
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
    factorized_params = (
        count_named_params(raw_model, "shared_mlp.u")
        + count_named_params(raw_model, "shared_mlp.v")
        + count_named_params(raw_model, "shared_mlp.coeff")
    )
    shared_mlp = getattr(raw_model, "shared_mlp", None)
    basis_params = factorized_params if isinstance(shared_mlp, BasisFieldMLP) else 0
    lowrank_params = factorized_params if isinstance(shared_mlp, LowRankResidualMLP) else 0
    return TrainResult(
        final_val_loss=history[-1]["val_loss"],
        final_train_loss=history[-1]["train_loss"],
        history=history,
        seconds=time.time() - start,
        params_total=count_params(raw_model, include_embedding=True),
        params_nonembedding=count_params(raw_model, include_embedding=False),
        params_mlp=count_named_params(raw_model, "mlp"),
        params_basis=basis_params,
        params_lowrank=lowrank_params,
        params_factorized=factorized_params,
    )


def make_model(args: argparse.Namespace, vocab_size: int, variant: str, seed: int) -> tuple[BasisGPT, dict]:
    set_seed(seed)
    if variant == "unshared":
        kind = "unshared"
        rank = 0
        shared_component = True
    elif variant == "shared":
        kind = "shared"
        rank = 0
        shared_component = True
    elif variant.startswith("basis_r"):
        kind = "basis"
        rank = int(variant.split("basis_r", 1)[1])
        shared_component = True
    elif variant.startswith("basis_noshared_r"):
        kind = "basis"
        rank = int(variant.split("basis_noshared_r", 1)[1])
        shared_component = False
    elif variant.startswith("lowrank_r"):
        kind = "lowrank"
        rank = int(variant.split("lowrank_r", 1)[1])
        shared_component = True
    else:
        raise ValueError(f"unknown variant {variant}")
    model = BasisGPT(
        vocab=vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        heads=args.heads,
        depth=args.depth,
        mlp_kind=kind,
        basis_rank=rank,
        dropout=args.dropout,
        coeff_init=args.coeff_init,
        basis_scale=args.basis_scale,
        shared_component=shared_component,
    )
    return model, {
        "mlp_kind": kind,
        "basis_rank": rank,
        "shared_component": shared_component,
        "coeff_init": args.coeff_init,
        "basis_scale": args.basis_scale,
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def summarize(runs: list[dict], variant_order: list[str]) -> list[dict]:
    rows = []
    for variant in variant_order:
        selected = [run for run in runs if run["variant"] == variant]
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
                "params_mlp": selected[0]["params_mlp"],
                "params_basis": selected[0]["params_basis"],
                "params_lowrank": selected[0]["params_lowrank"],
                "params_factorized": selected[0]["params_factorized"],
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
                "val_loss",
                "train_loss",
                "seconds",
                "params_total",
                "params_nonembedding",
                "params_mlp",
                "params_basis",
                "params_lowrank",
                "params_factorized",
                "mlp_kind",
                "basis_rank",
                "shared_component",
                "coeff_init",
                "basis_scale",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(result["runs"])
    lines = [
        "# shared-basis parameter field",
        "",
        f"Status: {result['status']}",
        f"Device: {result['device']}",
        f"Seconds: {result['seconds']:.2f}",
        "",
        "## Configuration",
        "",
        f"- depth: {result['config']['depth']}",
        f"- dim: {result['config']['dim']}",
        f"- seq_len: {result['config']['seq_len']}",
        f"- train_tokens: {result['train_tokens']}",
        f"- val_tokens: {result['val_tokens']}",
        f"- variants: {', '.join(result['config']['variants'])}",
        "",
        "## Results",
        "",
        "| Variant | Runs | Val CE mean | Val CE sd | Total params | Non-embed params | MLP params | Basis params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['val_loss_mean']:.4f} | {row['val_loss_sd']:.4f} | "
            f"{row['params_total']} | {row['params_nonembedding']} | {row['params_mlp']} | {row['params_basis']} |"
        )
    lines += [
        "",
        "## Factorized Parameter Counts",
        "",
        "| Variant | Low-rank params | All factorized params |",
        "|---|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(f"| {row['variant']} | {row['params_lowrank']} | {row['params_factorized']} |")
    if result.get("decision"):
        lines += ["", "## Decision", "", result["decision"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> None:
    start = time.time()
    out = Path(args.out)
    device = device_from_arg(args.device)
    torch.set_float32_matmul_precision("high")
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
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    runs = []
    for seed in seeds:
        for variant in args.variants:
            model, meta = make_model(args, vocab_size, variant, stable_seed(seed, f"{variant}-init"))
            result = train_lm(
                model,
                train_data,
                val_data,
                args,
                args.steps,
                stable_seed(seed, f"{variant}-train"),
                variant,
                out,
            )
            run = {
                "variant": variant,
                "seed": seed,
                "val_loss": result.final_val_loss,
                "train_loss": result.final_train_loss,
                "seconds": result.seconds,
                "params_total": result.params_total,
                "params_nonembedding": result.params_nonembedding,
                "params_mlp": result.params_mlp,
                "params_basis": result.params_basis,
                "params_lowrank": result.params_lowrank,
                "params_factorized": result.params_factorized,
                **meta,
            }
            runs.append(run)
            append_event(out, {"event": "run_complete", **run})
    summary = summarize(runs, args.variants)
    by_variant = {row["variant"]: row for row in summary}
    decision = ""
    if "shared" in by_variant:
        gains = []
        for variant, row in by_variant.items():
            if variant.startswith("basis_r"):
                gains.append((variant, by_variant["shared"]["val_loss_mean"] - row["val_loss_mean"]))
        if gains:
            best = max(gains, key=lambda x: x[1])
            decision = f"best basis gain over shared MLP is {best[1]:.4f} CE for {best[0]}."
            if "unshared" in by_variant:
                gap = by_variant[best[0]]["val_loss_mean"] - by_variant["unshared"]["val_loss_mean"]
                decision += f" gap from unshared MLP is {gap:.4f} CE."
    result = {
        "status": "completed",
        "device": str(device),
        "seconds": time.time() - start,
        "token_path": str(token_path),
        "train_tokens": int(train_data.numel()),
        "val_tokens": int(val_data.numel()),
        "vocab_size": vocab_size,
        "config": vars(args),
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
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coeff-init", type=float, default=0.02)
    parser.add_argument("--basis-scale", type=float, default=1.0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", nargs="+", default=["shared", "basis_r128", "basis_r512", "unshared"])
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
