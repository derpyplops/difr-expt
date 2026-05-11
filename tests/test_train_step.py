"""Sanity tests for STE training: gradients flow, parameters move, loss drops."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

from difr_expt.int_cast import (
    IntLinear,
    fake_quantize_per_row_ste,
    fake_quantize_per_token_ste,
    freeze_model_int,
    patch_model_int_cast,
)


def test_fake_quantize_per_row_ste_passes_gradient():
    w = (torch.randn(8, 16, dtype=torch.float32) * 0.1).requires_grad_(True)
    out = fake_quantize_per_row_ste(w, bits=16)
    # STE: gradient through the quant op should equal the upstream gradient
    # (since the dequant-difference is detached).
    g_upstream = torch.randn_like(out)
    out.backward(g_upstream)
    assert w.grad is not None
    # STE means dw = g_upstream exactly, not g_upstream * d(quantize)/dw.
    assert torch.allclose(w.grad, g_upstream, atol=1e-6), \
        f"STE gradient diverged: max abs diff {(w.grad - g_upstream).abs().max().item()}"


def test_fake_quantize_per_token_ste_passes_gradient():
    x = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    out = fake_quantize_per_token_ste(x, bits=16)
    g_upstream = torch.randn_like(out)
    out.backward(g_upstream)
    assert torch.allclose(x.grad, g_upstream, atol=1e-6)


def test_intlinear_trainable_has_parameter():
    lin = nn.Linear(32, 16).to(torch.float32)
    ilin = IntLinear.from_linear(lin, weight_bits=16, activation_bits=16, trainable=True)
    assert ilin.trainable is True
    assert isinstance(ilin.weight_fp, nn.Parameter)
    assert ilin.weight_fp.requires_grad
    # frozen-mode buffers should be None placeholders
    assert ilin.weight_int is None
    assert ilin.weight_scale is None


def test_intlinear_trainable_forward_matches_quantize_then_matmul():
    """In eval, the trainable forward should equal a manual quant+matmul."""
    torch.manual_seed(0)
    lin = nn.Linear(32, 16).to(torch.float32)
    ilin = IntLinear.from_linear(lin, weight_bits=16, activation_bits=16, trainable=True)
    x = torch.randn(4, 32, dtype=torch.float32)
    y = ilin(x)
    # Reference: same STE applied manually.
    w_ste = fake_quantize_per_row_ste(ilin.weight_fp, 16)
    x_ste = fake_quantize_per_token_ste(x.float(), 16)
    expected = x_ste @ w_ste.t() + ilin.bias
    assert torch.allclose(y, expected.to(y.dtype), atol=1e-6)


def test_trainable_intlinear_backward_updates_weight_fp():
    torch.manual_seed(0)
    lin = nn.Linear(16, 8).to(torch.float32)
    ilin = IntLinear.from_linear(lin, weight_bits=16, activation_bits=16, trainable=True)
    w_before = ilin.weight_fp.detach().clone()
    x = torch.randn(4, 16, dtype=torch.float32)
    target = torch.randn(4, 8, dtype=torch.float32)
    opt = torch.optim.SGD([ilin.weight_fp], lr=1e-1)
    y = ilin(x)
    loss = (y - target).pow(2).mean()
    loss.backward()
    assert ilin.weight_fp.grad is not None
    assert ilin.weight_fp.grad.abs().sum().item() > 0
    opt.step()
    assert not torch.allclose(ilin.weight_fp, w_before)


def test_patch_model_trainable_creates_parameters():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = Toy()
    replaced = patch_model_int_cast(model, weight_bits=16, activation_bits=16, trainable=True)
    assert len(replaced) == 2
    params = [p for p in model.parameters() if p.requires_grad]
    # weight_fp for fc1 (16*8=128) and fc2 (4*16=64) + biases (16 + 4).
    fp_param_numels = sorted(
        m.weight_fp.numel() for m in model.modules() if isinstance(m, IntLinear)
    )
    assert fp_param_numels == [64, 128]


def test_loss_decreases_on_toy_with_ste():
    """End-to-end: train a misaligned student toward a teacher with logit-L2."""
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self, vocab=64, hid=32):
            super().__init__()
            self.fc1 = nn.Linear(vocab, hid)
            self.fc2 = nn.Linear(hid, vocab)

        def forward(self, x):
            return self.fc2(torch.tanh(self.fc1(x)))

    teacher = Toy()
    for p in teacher.parameters():
        p.requires_grad = False
    # Student has its own (different) init — gives the optimizer a real gap
    # to close, not just quant noise around a perfect-alignment start.
    student = Toy()
    patch_model_int_cast(student, weight_bits=16, activation_bits=16, trainable=True)
    for n, p in student.named_parameters():
        if not n.endswith(".weight_fp"):
            p.requires_grad = False

    x = torch.randn(8, 64)
    opt = torch.optim.AdamW(
        [m.weight_fp for m in student.modules() if isinstance(m, IntLinear)],
        lr=1e-2,
    )
    initial = (student(x) - teacher(x)).pow(2).mean().item()
    for _ in range(100):
        opt.zero_grad()
        loss = (student(x) - teacher(x)).pow(2).mean()
        loss.backward()
        opt.step()
    final = (student(x) - teacher(x)).pow(2).mean().item()
    assert final < initial * 0.5, f"loss did not decrease enough: {initial} -> {final}"


def test_freeze_model_int_bakes_weights():
    """After freeze_to_int, weight_fp is gone and forward uses fixed buffers."""
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc(x)

    model = Toy()
    patch_model_int_cast(model, weight_bits=16, activation_bits=16, trainable=True)
    x = torch.randn(4, 16)
    y_before = model(x).detach().clone()

    freeze_model_int(model)
    assert isinstance(model.fc, IntLinear)
    assert model.fc.weight_fp is None
    assert model.fc.weight_int is not None
    assert model.fc.weight_scale is not None

    y_after = model(x)
    # Frozen forward should match trainable forward in eval mode (no STE noise
    # in the value, only in the gradient path).
    assert torch.allclose(y_before, y_after, atol=1e-5), \
        f"freeze changed numerics: max diff {(y_before - y_after).abs().max().item()}"


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
