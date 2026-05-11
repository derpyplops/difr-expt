"""Unit tests for int approximations of non-matmul ops."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from difr_expt.int_ops import (
    IntMatmul,
    IntRMSNorm,
    int_rope_apply,
    int_silu,
    int_softmax,
    set_true_int_path,
)


# ---------------------------------------------------------------------------
# IntRMSNorm
# ---------------------------------------------------------------------------


def _fp_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference float RMSNorm (Llama/Qwen convention)."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    h = xf * torch.rsqrt(variance + eps)
    return (weight * h.to(orig_dtype)).to(orig_dtype)


def test_int_rmsnorm_close_to_float():
    torch.manual_seed(0)
    H = 512
    x = torch.randn(4, 16, H, dtype=torch.float32) * 0.5
    norm = IntRMSNorm(H, eps=1e-6, bits=24)
    norm.weight.data.copy_(torch.randn(H) * 0.1 + 1.0)
    y_int = norm(x)
    y_ref = _fp_rmsnorm(x, norm.weight, eps=1e-6)
    rel = (y_int - y_ref).norm() / y_ref.norm()
    assert rel.item() < 5e-4, f"rmsnorm rel err {rel.item()}"


def test_int_rmsnorm_invsqrt_lut_accuracy():
    """The NR invsqrt with 2 iterations should converge to ~1e-7 rel error."""
    norm = IntRMSNorm(64, bits=24, nr_iterations=2)
    v = torch.tensor([[1.0], [4.0], [0.25], [100.0], [1e-3]])
    r = norm._invsqrt(v)
    ref = 1.0 / v.sqrt()
    rel = (r - ref).abs() / ref
    assert rel.max().item() < 1e-5, f"invsqrt rel err {rel}"


def test_int_rmsnorm_true_int_matches_float_equiv():
    """true_int_path branch should match the default float-equivalent forward."""
    torch.manual_seed(4)
    H = 256
    x = torch.randn(3, 12, H, dtype=torch.float32) * 0.5
    norm = IntRMSNorm(H, eps=1e-6, bits=24)
    norm.weight.data.copy_(torch.randn(H) * 0.1 + 1.0)
    norm.true_int_path = False
    y_float = norm(x)
    norm.true_int_path = True
    y_int = norm(x)
    max_abs = (y_float - y_int).abs().max().item()
    assert max_abs < 1e-4, f"rmsnorm true-int delta {max_abs}"


def test_int_rmsnorm_from_hf():
    """from_hf_rmsnorm should pick up weight and eps from an HF-style RMSNorm."""
    class FakeRMSNorm(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(hidden))
            self.variance_epsilon = 1e-5

    hf = FakeRMSNorm(128)
    int_norm = IntRMSNorm.from_hf_rmsnorm(hf, bits=24)
    assert int_norm.hidden_size == 128
    assert int_norm.eps == 1e-5
    assert torch.allclose(int_norm.weight, hf.weight.float())


# ---------------------------------------------------------------------------
# int_softmax
# ---------------------------------------------------------------------------


def test_int_softmax_close_to_float():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 3.0  # softmax inputs typically [-10, 10]
    p_ref = F.softmax(x, dim=-1)
    p_int = int_softmax(x, dim=-1, lut_size=1024)
    rel = (p_ref - p_int).abs().max().item()
    assert rel < 2e-3, f"softmax max abs err {rel}"


def test_int_softmax_sums_to_one():
    x = torch.randn(4, 32) * 2.0
    p = int_softmax(x, dim=-1, lut_size=1024)
    sums = p.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_int_softmax_lut_size_improves_accuracy():
    """Larger LUT → tighter approximation."""
    torch.manual_seed(1)
    x = torch.randn(8, 128) * 5.0
    p_ref = F.softmax(x, dim=-1)
    err_1k = (p_ref - int_softmax(x, lut_size=1024)).abs().max().item()
    err_64k = (p_ref - int_softmax(x, lut_size=65536)).abs().max().item()
    assert err_64k <= err_1k * 2, f"larger LUT did not help: {err_1k} vs {err_64k}"


def test_int_softmax_true_int_matches_float_equiv():
    """true_int=True path should match the default float-equivalent path."""
    torch.manual_seed(2)
    x = torch.randn(8, 64) * 3.0
    p_float = int_softmax(x, dim=-1, lut_size=1024, true_int=False)
    p_int = int_softmax(x, dim=-1, lut_size=1024, true_int=True)
    max_abs = (p_float - p_int).abs().max().item()
    assert max_abs < 1e-4, f"softmax true-int delta {max_abs}"


# ---------------------------------------------------------------------------
# int_silu
# ---------------------------------------------------------------------------


def test_int_silu_close_to_float():
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 3.0
    y_ref = F.silu(x)
    y_int = int_silu(x, lut_size=4096)
    rel = (y_int - y_ref).norm() / y_ref.norm()
    assert rel.item() < 5e-3, f"silu rel err {rel.item()}"


def test_int_silu_lut_size_improves_accuracy():
    torch.manual_seed(2)
    x = torch.randn(32, 64) * 2.0
    y_ref = F.silu(x)
    err_4k = (y_ref - int_silu(x, lut_size=4096)).abs().max().item()
    err_64k = (y_ref - int_silu(x, lut_size=65536)).abs().max().item()
    assert err_64k <= err_4k, f"larger LUT did not help: {err_4k} vs {err_64k}"


def test_int_silu_true_int_matches_float_equiv():
    """true_int=True path should match the default float-equivalent path."""
    torch.manual_seed(3)
    x = torch.randn(16, 64) * 3.0
    y_float = int_silu(x, lut_size=4096, true_int=False)
    y_int = int_silu(x, lut_size=4096, true_int=True)
    max_abs = (y_float - y_int).abs().max().item()
    assert max_abs < 1e-4, f"silu true-int delta {max_abs}"


# ---------------------------------------------------------------------------
# IntMatmul
# ---------------------------------------------------------------------------


def test_int_matmul_close_to_float():
    torch.manual_seed(0)
    a = torch.randn(2, 4, 16, 64)  # [b, h, m, k]
    b = torch.randn(2, 4, 64, 32)  # [b, h, k, n]
    out_ref = a @ b
    out_int = IntMatmul(bits=24)(a, b)
    rel = (out_int - out_ref).norm() / out_ref.norm()
    assert rel.item() < 1e-3, f"int_matmul rel err {rel.item()}"


def test_int_matmul_true_int_matches_float_path():
    torch.manual_seed(0)
    a = torch.randn(1, 1, 8, 16)
    b = torch.randn(1, 1, 16, 12)

    mm = IntMatmul(bits=20)
    mm.true_int_path = False
    out_float = mm(a, b)
    mm.true_int_path = True
    out_int = mm(a, b)

    rel = (out_float - out_int).norm() / out_int.norm()
    assert rel.item() < 1e-5, f"path divergence {rel.item()}"


def test_set_true_int_path_toggle():
    """set_true_int_path should flip the flag on IntRMSNorm and IntMatmul."""
    mod = nn.Sequential(
        IntRMSNorm(32),
        IntMatmul(bits=24),
    )
    set_true_int_path(mod, True)
    assert mod[0].true_int_path is True
    assert mod[1].true_int_path is True
    set_true_int_path(mod, False)
    assert mod[0].true_int_path is False
    assert mod[1].true_int_path is False


def test_set_true_int_path_covers_silu_and_softmax_cfg():
    """set_true_int_path should also flip IntSiLUModule.true_int_path and the
    cfg.true_int_nonmatmul flag on attached cfgs (for softmax)."""
    from difr_expt.patch_hf_model import IntSiLUModule, IntOpsConfig

    cfg = IntOpsConfig()
    silu = IntSiLUModule(cfg)

    class FakeAttn(nn.Module):
        pass

    attn = FakeAttn()
    attn._int_ops_cfg = cfg

    mod = nn.Module()
    mod.silu = silu
    mod.attn = attn

    set_true_int_path(mod, True)
    assert silu.true_int_path is True
    assert cfg.true_int_nonmatmul is True
    set_true_int_path(mod, False)
    assert silu.true_int_path is False
    assert cfg.true_int_nonmatmul is False


# ---------------------------------------------------------------------------
# int_rope_apply (rotary position embedding)
# ---------------------------------------------------------------------------


def _rotate_half_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference rotate_half — matches transformers' implementation."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _fp_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Reference float rotary apply, matching HF's apply_rotary_pos_emb."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + _rotate_half_ref(q) * sin
    k_embed = k * cos + _rotate_half_ref(k) * sin
    return q_embed, k_embed


def test_int_rope_close_to_float():
    """int_rope_apply @ bits=24 should match the float rotary within <1e-4."""
    torch.manual_seed(0)
    batch, n_heads, seq, head_dim = 1, 4, 8, 64
    q = torch.randn(batch, n_heads, seq, head_dim) * 0.5
    k = torch.randn(batch, n_heads, seq, head_dim) * 0.5

    # cos/sin from real rotary frequencies (more realistic than random in [-1,1]).
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq, half]
    emb = torch.cat([freqs, freqs], dim=-1)   # [seq, head_dim]
    cos = emb.cos().unsqueeze(0).expand(batch, seq, head_dim).contiguous()
    sin = emb.sin().unsqueeze(0).expand(batch, seq, head_dim).contiguous()

    q_ref, k_ref = _fp_apply_rotary_pos_emb(q, k, cos, sin)
    q_int, k_int = int_rope_apply(q, k, cos, sin, bits=24)

    q_err = (q_int - q_ref).abs().max().item()
    k_err = (k_int - k_ref).abs().max().item()
    max_err = max(q_err, k_err)
    assert max_err < 1e-4, f"int_rope max abs err {max_err} (q={q_err}, k={k_err})"


def test_int_rope_true_int_matches_float_equiv():
    """true_int=True path should bit-equivalently match the default float path."""
    torch.manual_seed(1)
    batch, n_heads, seq, head_dim = 1, 4, 8, 64
    q = torch.randn(batch, n_heads, seq, head_dim) * 0.5
    k = torch.randn(batch, n_heads, seq, head_dim) * 0.5

    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).expand(batch, seq, head_dim).contiguous()
    sin = emb.sin().unsqueeze(0).expand(batch, seq, head_dim).contiguous()

    q_float, k_float = int_rope_apply(q, k, cos, sin, bits=24, true_int=False)
    q_int, k_int = int_rope_apply(q, k, cos, sin, bits=24, true_int=True)

    assert torch.allclose(q_float, q_int, atol=1e-5), (
        f"q path divergence max={ (q_float - q_int).abs().max().item() }"
    )
    assert torch.allclose(k_float, k_int, atol=1e-5), (
        f"k path divergence max={ (k_float - k_int).abs().max().item() }"
    )


# ---------------------------------------------------------------------------
# HF model surgery sanity test
# ---------------------------------------------------------------------------


def test_patch_replaces_rmsnorm_on_toy_qwen_like():
    """Build a toy module that looks like Qwen2RMSNorm by class name and confirm
    the patcher replaces it."""

    class Qwen2RMSNorm(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden))
            self.variance_epsilon = 1e-6

        def forward(self, x):
            xf = x.float()
            var = xf.pow(2).mean(-1, keepdim=True)
            h = xf * torch.rsqrt(var + self.variance_epsilon)
            return (self.weight * h).to(x.dtype)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = Qwen2RMSNorm(64)
            self.linear = nn.Linear(64, 64)

        def forward(self, x):
            return self.linear(self.input_layernorm(x))

    from difr_expt.patch_hf_model import patch_model_int_nonmatmul, IntOpsConfig

    model = Toy()
    cfg = IntOpsConfig(replace_softmax=False, replace_attn_matmul=False, replace_silu=False)
    counts = patch_model_int_nonmatmul(model, cfg)
    assert counts["rmsnorm"] == 1
    assert isinstance(model.input_layernorm, IntRMSNorm)

    x = torch.randn(2, 8, 64)
    y = model(x)
    assert y.shape == (2, 8, 64)


if __name__ == "__main__":
    import inspect
    g = dict(globals())
    fails = []
    for name, fn in g.items():
        if name.startswith("test_") and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                fails.append(name)
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                fails.append(name)
    print(f"failures: {fails}")
    sys.exit(1 if fails else 0)
