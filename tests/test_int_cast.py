"""Unit tests for full-int IntLinear."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

from difr_expt.int_cast import (
    IntEmbedding,
    IntLinear,
    patch_model_int_cast,
    patch_model_int_embedding,
    quantize_per_row,
    quantize_per_token,
    set_true_int_matmul,
)


def test_quantize_per_row_roundtrip_16bit():
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.float32) * 0.1
    w_int, scale = quantize_per_row(w, bits=16)
    assert scale.shape == (64, 1)
    assert (w_int.abs().max() <= (1 << 15) - 1).item()
    w_recon = w_int.to(torch.float32) * scale
    # 16-bit symmetric quant: max error ~ 0.5 * scale, so relative error ~ 1/qmax = 3e-5.
    rel_err = (w_recon - w).abs() / w.abs().clamp_min(1e-12)
    # Use a robust statistic; sub-min values can blow per-element rel error.
    assert rel_err.median().item() < 1e-4
    abs_err = (w_recon - w).abs() / w.abs().max()
    assert abs_err.max().item() < 1e-4, f"max abs/scale {abs_err.max().item()}"


def test_quantize_per_token_basic():
    x = torch.randn(8, 64, dtype=torch.float32)
    x_int, scale = quantize_per_token(x, bits=16)
    assert scale.shape == (8, 1)
    assert (x_int.abs().max() <= (1 << 15) - 1).item()
    x_recon = x_int.to(torch.float32) * scale
    abs_err = (x_recon - x).abs() / x.abs().max()
    assert abs_err.max().item() < 1e-4


def test_int_linear_float_path_matches_fp32_linear_approximately():
    """16-bit weight + 16-bit activation should still produce outputs within
    ~1e-3 relative of the float reference for typical inputs."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 64).to(torch.float32)
    int_lin = IntLinear.from_linear(lin, weight_bits=16, activation_bits=16)
    x = torch.randn(4, 128, dtype=torch.float32)
    y_ref = lin(x)
    y_int = int_lin(x)
    rel = (y_int - y_ref).norm() / y_ref.norm()
    # Expected ~1e-5 from per-weight rel error of ~3e-5 averaged across dim;
    # accept up to 1e-3 with safety margin for the per-token activation quant.
    assert rel.item() < 1e-3, f"relative L2 {rel.item()}"


def test_int_linear_no_bias():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=False).to(torch.float32)
    int_lin = IntLinear.from_linear(lin, weight_bits=16, activation_bits=16)
    assert int_lin.bias is None
    x = torch.randn(2, 64, dtype=torch.float32)
    y_ref = lin(x)
    y_int = int_lin(x)
    rel = (y_int - y_ref).norm() / y_ref.norm()
    assert rel.item() < 1e-3


def test_int_linear_bf16_compute():
    torch.manual_seed(0)
    lin = nn.Linear(128, 64).to(torch.bfloat16)
    int_lin = IntLinear.from_linear(lin, weight_bits=16, activation_bits=16)
    assert int_lin.compute_dtype == torch.bfloat16
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    y_ref = lin(x)
    y_int = int_lin(x)
    # bf16 base noise dominates; just check we're within an order of magnitude.
    rel = ((y_int - y_ref).float()).norm() / y_ref.float().norm()
    assert rel.item() < 5e-2, f"relative L2 {rel.item()}"


def test_true_int_path_matches_float_path():
    """The two compute paths should agree to within float roundoff."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 64).to(torch.float32)
    int_lin = IntLinear.from_linear(lin, weight_bits=16, activation_bits=16)
    x = torch.randn(4, 128, dtype=torch.float32)

    int_lin.true_int_matmul = False
    y_float_path = int_lin(x)
    int_lin.true_int_matmul = True
    y_int_path = int_lin(x)

    # They should be very close; difference comes only from float reduction order.
    rel = (y_float_path - y_int_path).norm() / y_int_path.norm()
    assert rel.item() < 1e-5, f"path divergence {rel.item()}"


def test_patch_model_replaces_linears():
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 32)
            self.lm_head = nn.Linear(32, 100)

        def forward(self, ids):
            h = self.embed(ids)
            h = torch.relu(self.fc1(h))
            h = self.fc2(h)
            return self.lm_head(h)

    model = Toy()
    model_int = Toy()
    model_int.load_state_dict(model.state_dict())
    replaced = patch_model_int_cast(model_int, weight_bits=16, activation_bits=16)
    assert set(replaced.keys()) == {"fc1", "fc2", "lm_head"}, replaced.keys()

    ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        y_ref = model(ids)
        y_int = model_int(ids)
    rel = (y_ref - y_int).norm() / y_ref.norm()
    # Three stacked quant matmuls; expect a small but visible error.
    assert rel.item() < 5e-3, f"end-to-end rel error: {rel.item()}"


def test_patch_model_skip_lm_head():
    model = nn.Sequential()
    model.add_module("fc", nn.Linear(8, 8))
    model.add_module("lm_head", nn.Linear(8, 100))
    replaced = patch_model_int_cast(model, weight_bits=16, activation_bits=16, include_lm_head=False)
    assert "fc" in replaced
    assert "lm_head" not in replaced
    assert isinstance(model.lm_head, nn.Linear)


def test_set_true_int_matmul_toggle():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    patch_model_int_cast(model, weight_bits=16, activation_bits=16)
    set_true_int_matmul(model, True)
    for m in model:
        assert m.true_int_matmul is True
    set_true_int_matmul(model, False)
    for m in model:
        assert m.true_int_matmul is False


def test_argmax_match_rate_toy():
    """Top-1 argmax match between float and int-cast on a toy model. With full
    int conversion at 16 bits we expect very high but not necessarily perfect
    match, even on random-weight nets."""
    torch.manual_seed(0)
    n_vocab = 1000

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(n_vocab, 64)
            self.fc = nn.Linear(64, 64)
            self.lm_head = nn.Linear(64, n_vocab)

        def forward(self, ids):
            return self.lm_head(torch.tanh(self.fc(self.embed(ids))))

    model = Toy()
    model_int = Toy()
    model_int.load_state_dict(model.state_dict())
    patch_model_int_cast(model_int, weight_bits=16, activation_bits=16)

    ids = torch.randint(0, n_vocab, (8, 32))
    with torch.no_grad():
        logits_ref = model(ids)
        logits_int = model_int(ids)
    match = (logits_ref.argmax(-1) == logits_int.argmax(-1)).float().mean().item()
    print(f"toy top-1 match: {match:.6f}")
    # 16-bit quant on a tiny untrained model: ≥99%.
    assert match >= 0.99, f"toy match rate too low: {match}"


def test_int_embedding_close_to_float_at_b24():
    torch.manual_seed(0)
    vocab, d = 257, 64
    emb = nn.Embedding(vocab, d)
    emb.weight.data = torch.randn(vocab, d) * 0.05
    int_emb = IntEmbedding.from_embedding(emb, bits=24)
    ids = torch.randint(0, vocab, (4, 16))
    out_ref = emb(ids)
    out_int = int_emb(ids)
    assert out_int.shape == out_ref.shape
    assert out_int.dtype == out_ref.dtype
    delta = (out_int.float() - out_ref.float()).abs().max().item()
    # b=24 symmetric quant: per-element relative noise ~ 1/(2^23-1) ≈ 1.2e-7
    assert delta < 1e-5, f"int_embedding delta too large: {delta}"


def test_int_embedding_padding_idx_zeroed():
    torch.manual_seed(0)
    vocab, d = 32, 16
    emb = nn.Embedding(vocab, d, padding_idx=0)
    emb.weight.data = torch.randn(vocab, d) * 0.1
    int_emb = IntEmbedding.from_embedding(emb, bits=24)
    ids = torch.tensor([[0, 5, 0, 7]])
    out = int_emb(ids)
    assert torch.all(out[0, 0] == 0)
    assert torch.all(out[0, 2] == 0)
    assert not torch.all(out[0, 1] == 0)


def test_patch_model_int_embedding_replaces():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Embedding(64, 32),
        nn.Linear(32, 32),
    )
    replaced = patch_model_int_embedding(model, bits=24)
    assert len(replaced) == 1
    assert isinstance(model[0], IntEmbedding)
    ids = torch.randint(0, 64, (2, 8))
    out = model(ids)
    assert out.shape == (2, 8, 32)


if __name__ == "__main__":
    g = dict(globals())
    fails = []
    for name, fn in g.items():
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                fails.append(name)
    print(f"failures: {fails}")
    sys.exit(1 if fails else 0)
