from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
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


class SharedBasisGroupedMLP(nn.Module):
    """Grouped MLP whose block weights are mixtures from one global basis bank."""

    def __init__(self, dim: int, depth: int, groups: int, rank: int, expansion: int) -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be positive.")
        if dim % groups:
            raise ValueError("dim must be divisible by groups.")
        if depth < 1 or rank < 1 or expansion < 1:
            raise ValueError("depth, rank, and expansion must be positive.")
        self.dim = dim
        self.depth = depth
        self.groups = groups
        self.rank = rank
        self.group_dim = dim // groups
        self.hidden_dim = self.group_dim * expansion

        up_scale = self.group_dim**-0.5
        down_scale = self.hidden_dim**-0.5
        coeff_scale = rank**-0.5
        self.basis_up = nn.Parameter(torch.randn(rank, self.group_dim, self.hidden_dim) * up_scale)
        self.basis_down = nn.Parameter(
            torch.randn(rank, self.hidden_dim, self.group_dim) * down_scale
        )
        self.coeff_up = nn.Parameter(torch.randn(depth, groups, rank) * coeff_scale)
        self.coeff_down = nn.Parameter(torch.randn(depth, groups, rank) * coeff_scale)

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not 0 <= layer_idx < self.depth:
            raise IndexError(f"layer_idx={layer_idx} is outside depth={self.depth}.")
        batch, tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {channels}.")
        grouped = x.reshape(batch, tokens, self.groups, self.group_dim)
        w_up = torch.einsum("gr,rch->gch", self.coeff_up[layer_idx], self.basis_up)
        w_down = torch.einsum("gr,rhc->ghc", self.coeff_down[layer_idx], self.basis_down)
        hidden = torch.einsum("btgc,gch->btgh", grouped, w_up)
        hidden = F.gelu(hidden)
        out = torch.einsum("btgh,ghc->btgc", hidden, w_down)
        return out.reshape(batch, tokens, channels)


class UnsharedGroupedMLP(nn.Module):
    def __init__(self, dim: int, depth: int, groups: int, expansion: int) -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be positive.")
        if dim % groups:
            raise ValueError("dim must be divisible by groups.")
        if depth < 1 or expansion < 1:
            raise ValueError("depth and expansion must be positive.")
        self.dim = dim
        self.depth = depth
        self.groups = groups
        self.group_dim = dim // groups
        self.hidden_dim = self.group_dim * expansion
        up_scale = self.group_dim**-0.5
        down_scale = self.hidden_dim**-0.5
        self.up = nn.Parameter(
            torch.randn(depth, groups, self.group_dim, self.hidden_dim) * up_scale
        )
        self.down = nn.Parameter(
            torch.randn(depth, groups, self.hidden_dim, self.group_dim) * down_scale
        )

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not 0 <= layer_idx < self.depth:
            raise IndexError(f"layer_idx={layer_idx} is outside depth={self.depth}.")
        batch, tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {channels}.")
        grouped = x.reshape(batch, tokens, self.groups, self.group_dim)
        hidden = torch.einsum("btgc,gch->btgh", grouped, self.up[layer_idx])
        hidden = F.gelu(hidden)
        out = torch.einsum("btgh,ghc->btgc", hidden, self.down[layer_idx])
        return out.reshape(batch, tokens, channels)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mlp: nn.Module, layer_idx: int) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(mlp(self.ln2(x), layer_idx))
        return x


class _GroupedGPTBase(nn.Module):
    def __init__(
        self,
        *,
        vocab: int,
        seq_len: int,
        dim: int,
        heads: int,
        depth: int,
        dropout: float,
        mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.layers = nn.ModuleList([TransformerLayer(dim, heads, dropout) for _ in range(depth)])
        self.mlp = mlp
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, tokens = idx.shape
        if tokens > self.seq_len:
            raise ValueError(f"sequence length {tokens} exceeds configured seq_len={self.seq_len}.")
        x = self.tok_emb(idx) + self.pos_emb[:, :tokens]
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, self.mlp, layer_idx)
        return self.head(self.ln_f(x))


class SharedBasisGPT(_GroupedGPTBase):
    def __init__(
        self,
        *,
        vocab: int,
        seq_len: int,
        dim: int,
        heads: int,
        depth: int,
        groups: int,
        rank: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__(
            vocab=vocab,
            seq_len=seq_len,
            dim=dim,
            heads=heads,
            depth=depth,
            dropout=dropout,
            mlp=SharedBasisGroupedMLP(
                dim=dim,
                depth=depth,
                groups=groups,
                rank=rank,
                expansion=expansion,
            ),
        )


class UnsharedGroupedGPT(_GroupedGPTBase):
    def __init__(
        self,
        *,
        vocab: int,
        seq_len: int,
        dim: int,
        heads: int,
        depth: int,
        groups: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__(
            vocab=vocab,
            seq_len=seq_len,
            dim=dim,
            heads=heads,
            depth=depth,
            dropout=dropout,
            mlp=UnsharedGroupedMLP(dim=dim, depth=depth, groups=groups, expansion=expansion),
        )


def shared_basis_compression_ratio(
    *,
    depth: int,
    groups: int,
    group_dim: int,
    rank: int,
    expansion: int,
) -> float:
    hidden_dim = group_dim * expansion
    unshared = depth * groups * ((group_dim * hidden_dim) + (hidden_dim * group_dim))
    shared = (
        rank * ((group_dim * hidden_dim) + (hidden_dim * group_dim))
        + 2 * depth * groups * rank
    )
    return shared / unshared


@dataclass(slots=True)
class ToyRunConfig:
    vocab: int = 64
    train_tokens: int = 4096
    val_tokens: int = 1024
    seq_len: int = 32
    batch_size: int = 8
    eval_batches: int = 8
    steps: int = 50
    lr: float = 3e-3
    weight_decay: float = 0.0
    dim: int = 64
    heads: int = 4
    depth: int = 4
    groups: int = 4
    rank: int = 2
    expansion: int = 4
    dropout: float = 0.0
    seed: int = 0
    device: str = "cuda"


def device_from_name(name: str) -> torch.device:
    if name != "cuda":
        raise ValueError("shared-basis toy runs are CUDA-only; pass device='cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("shared-basis toy runs require a visible CUDA device.")
    return torch.device("cuda")


def _random_tokens(*, vocab: int, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (count,), generator=gen, dtype=torch.long)


def _batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    data: torch.Tensor,
    config: ToyRunConfig,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(config.eval_batches):
        x, y = _batch(
            data,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            gen=gen,
        )
        logits = model(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
    return float(np.mean(losses))


def _train_toy_variant(
    *,
    variant: str,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: ToyRunConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    start = time.time()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    history: list[dict[str, float | int]] = []
    initial_val_loss = _eval_loss(model, val_data, config, device, seed + 101)
    for step in range(1, config.steps + 1):
        model.train()
        x, y = _batch(
            train_data,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            gen=gen,
        )
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append({"step": step, "train_loss": float(loss.detach())})
    final_val_loss = _eval_loss(model, val_data, config, device, seed + 202)
    return {
        "variant": variant,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "train_loss": history[-1]["train_loss"] if history else None,
        "history": history,
        "seconds": time.time() - start,
        "total_params": count_parameters(model),
        "mlp_params": count_parameters(model.mlp),  # type: ignore[attr-defined]
    }


def run_toy_suite(config: ToyRunConfig, *, out: str | Path | None = None) -> dict[str, Any]:
    if config.train_tokens <= config.seq_len + 1 or config.val_tokens <= config.seq_len + 1:
        raise ValueError("train_tokens and val_tokens must exceed seq_len + 1.")
    device = device_from_name(config.device)
    train_data = _random_tokens(vocab=config.vocab, count=config.train_tokens, seed=config.seed + 1)
    val_data = _random_tokens(vocab=config.vocab, count=config.val_tokens, seed=config.seed + 2)
    set_seed(config.seed)
    shared = SharedBasisGPT(
        vocab=config.vocab,
        seq_len=config.seq_len,
        dim=config.dim,
        heads=config.heads,
        depth=config.depth,
        groups=config.groups,
        rank=config.rank,
        expansion=config.expansion,
        dropout=config.dropout,
    )
    set_seed(config.seed)
    unshared = UnsharedGroupedGPT(
        vocab=config.vocab,
        seq_len=config.seq_len,
        dim=config.dim,
        heads=config.heads,
        depth=config.depth,
        groups=config.groups,
        expansion=config.expansion,
        dropout=config.dropout,
    )
    runs = [
        _train_toy_variant(
            variant="shared_basis",
            model=shared,
            train_data=train_data,
            val_data=val_data,
            config=config,
            device=device,
            seed=config.seed + 10,
        ),
        _train_toy_variant(
            variant="unshared_grouped",
            model=unshared,
            train_data=train_data,
            val_data=val_data,
            config=config,
            device=device,
            seed=config.seed + 20,
        ),
    ]
    result = {
        "status": "completed",
        "device": str(device),
        "config": asdict(config),
        "shared_mlp_to_unshared_mlp_ratio": runs[0]["mlp_params"] / runs[1]["mlp_params"],
        "theory_shared_mlp_to_unshared_mlp_ratio": shared_basis_compression_ratio(
            depth=config.depth,
            groups=config.groups,
            group_dim=config.dim // config.groups,
            rank=config.rank,
            expansion=config.expansion,
        ),
        "runs": runs,
    }
    if out is not None:
        write_toy_outputs(Path(out), result)
    return result


def write_toy_outputs(out: Path, result: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# shared-basis depth-width sharing",
        "",
        f"Status: {result['status']}",
        f"Device: {result['device']}",
        "",
        "## Structure",
        "",
        f"- depth: {result['config']['depth']}",
        f"- dim: {result['config']['dim']}",
        f"- groups: {result['config']['groups']}",
        f"- rank: {result['config']['rank']}",
        f"- MLP parameter ratio: {result['shared_mlp_to_unshared_mlp_ratio']:.4f}",
        "",
        "## Toy Losses",
        "",
        "| Variant | Initial val CE | Final val CE | Final train CE | MLP params | "
        "Total params | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {run['variant']} | {run['initial_val_loss']:.4f} | {run['final_val_loss']:.4f} | "
            f"{run['train_loss']:.4f} | {run['mlp_params']} | {run['total_params']} | "
            f"{run['seconds']:.2f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/shared_basis_toy")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--train-tokens", type=int, default=4096)
    parser.add_argument("--val-tokens", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ToyRunConfig(
        vocab=args.vocab,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dim=args.dim,
        heads=args.heads,
        depth=args.depth,
        groups=args.groups,
        rank=args.rank,
        expansion=args.expansion,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
    )
    result = run_toy_suite(config, out=args.out)
    print(json.dumps({"status": result["status"], "out": args.out}, indent=2))
