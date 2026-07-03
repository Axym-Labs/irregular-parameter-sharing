import torch

from irregular_parameter_sharing.shared_basis_lm import (
    SharedBasisGPT,
    SharedBasisGroupedMLP,
    ToyRunConfig,
    UnsharedGroupedGPT,
    count_parameters,
    run_toy_suite,
)


def test_shared_basis_mlp_mixes_one_basis_bank_by_layer_and_width_group() -> None:
    mlp = SharedBasisGroupedMLP(dim=16, depth=3, groups=4, rank=2, expansion=2)
    x = torch.randn(2, 5, 16)

    y0 = mlp(x, layer_idx=0)
    y1 = mlp(x, layer_idx=1)

    assert y0.shape == x.shape
    assert y1.shape == x.shape
    assert mlp.basis_up.shape == (2, 4, 8)
    assert mlp.basis_down.shape == (2, 8, 4)
    assert mlp.coeff_up.shape == (3, 4, 2)
    assert mlp.coeff_down.shape == (3, 4, 2)
    assert not torch.allclose(y0, y1)


def test_shared_basis_gpt_is_forward_compatible_and_reduces_mlp_parameters() -> None:
    shared = SharedBasisGPT(
        vocab=32,
        seq_len=8,
        dim=16,
        heads=4,
        depth=3,
        groups=4,
        rank=2,
        expansion=2,
        dropout=0.0,
    )
    unshared = UnsharedGroupedGPT(
        vocab=32,
        seq_len=8,
        dim=16,
        heads=4,
        depth=3,
        groups=4,
        expansion=2,
        dropout=0.0,
    )
    idx = torch.randint(0, 32, (2, 8))

    logits = shared(idx)

    assert logits.shape == (2, 8, 32)
    assert count_parameters(shared.mlp) < count_parameters(unshared.mlp)


def test_toy_suite_writes_structural_artifacts(tmp_path) -> None:
    result = run_toy_suite(
        ToyRunConfig(
            vocab=16,
            train_tokens=128,
            val_tokens=32,
            seq_len=4,
            batch_size=2,
            eval_batches=2,
            steps=2,
            dim=8,
            heads=2,
            depth=2,
            groups=2,
            rank=1,
            expansion=2,
            seed=5,
            device="cpu",
        ),
        out=tmp_path,
    )

    assert result["status"] == "completed"
    assert [run["variant"] for run in result["runs"]] == ["shared_basis", "unshared_grouped"]
    assert result["runs"][0]["mlp_params"] < result["runs"][1]["mlp_params"]
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "summary.md").exists()
