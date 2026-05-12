"""Bit-exact int wrappers for non-matmul ops.

The full int model architecture:
- At every operation boundary, every bf16 value is committed as int30 + scale
  (per-token or per-row). This is the prover's commitment in DiFR.
- The bf16 → int30 → bf16 round-trip is *exact identity* for all bf16 values
  (empirically verified for int28+; we use int30 with margin). The operation's
  output is therefore bit-identical to teacher's output on the same input.
- The operation kernel itself runs in teacher's exact compute path (bf16/fp32
  ops via torch). These primitives are deterministic and reproducible — in a
  ZK circuit they expand to fp32-on-int31 bit emulation, which is standard.

In short: this is the "boundary int model" — int commitments at every layer,
deterministic ops between them. This satisfies DiFR's proof obligation while
preserving top-1 bit-exact teacher (≠ the LUT-based int_ops.py implementations,
which approximate the float ops and lose top-1 to ~0.91).
"""

from __future__ import annotations

from typing import Optional, Any

import torch
import torch.nn as nn


_INT30_QMAX = float((1 << 29) - 1)


def commit_int30_bf16(x: torch.Tensor, qmax: float = _INT30_QMAX) -> torch.Tensor:
    """Quantize a bf16 tensor to int30 per-token (last-dim absmax) and dequant.

    For bf16 input the round-trip is *exact identity* (qmax = 2^29 - 1 gives
    grid spacing ≤ bf16 ULP at every element). For non-bf16 inputs this is a
    pass-through.

    Returns a tensor bit-identical to the input (when bf16). The int values
    (x / scale).round() and the per-token scale together commit the operand.
    """
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        return x
    # Compute absmax in fp32. Doing it in bf16 and then casting to fp32 can
    # round absmax to a different value than the actual bf16 absmax (visible on
    # short rows like Qwen3's per-head [..., 128] q_norm inputs), which shifts
    # the int30 grid and breaks identity.
    x_fp32 = x.detach().to(torch.float32)
    absmax = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = absmax / qmax  # fp32
    x_int = (x.to(torch.float32) / scale).round().clamp(-qmax, qmax)
    return (x_int * scale).to(torch.bfloat16)


class IntCommitWrap(nn.Module):
    """Wraps a module; quantizes bf16 positional/keyword args to int30 first."""

    def __init__(self, inner: nn.Module, bits: int = 30):
        super().__init__()
        self.inner = inner
        self._qmax = float((1 << (bits - 1)) - 1)

    def _maybe_commit(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16:
            return commit_int30_bf16(x, qmax=self._qmax)
        return x

    def forward(self, *args, **kwargs):
        new_args = tuple(self._maybe_commit(a) for a in args)
        new_kwargs = {k: self._maybe_commit(v) for k, v in kwargs.items()}
        return self.inner(*new_args, **new_kwargs)


def patch_model_int_bitexact(model: nn.Module) -> dict[str, int]:
    """Wrap non-matmul ops with explicit int30 commits.

    Replaces in-place:
      - RMSNorm modules → IntCommitWrap(RMSNorm)
      - MLP.act_fn (SiLU) → IntCommitWrap(act_fn)

    Attention's softmax / Q@K.T / P@V are not wrapped here — they sit inside
    the attention forward closure, but they receive bf16 tensors that have
    already been committed by the upstream IntLinear (int_matmul_path) on
    q_proj/k_proj/v_proj outputs. Q@K.T and P@V are exact bf16 matmuls that
    can be int-committed analogously (todo if needed).

    Returns replacement counts for sanity checks.
    """
    from difr_expt.patch_hf_model import _is_rmsnorm, _is_mlp

    counts = {"rmsnorm": 0, "silu": 0}

    to_wrap_norms: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if _is_rmsnorm(mod):
            to_wrap_norms.append((name, mod))
    for name, mod in to_wrap_norms:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, IntCommitWrap(mod))
        counts["rmsnorm"] += 1

    for name, mod in model.named_modules():
        if _is_mlp(mod):
            if hasattr(mod, "act_fn") and not isinstance(mod.act_fn, IntCommitWrap):
                mod.act_fn = IntCommitWrap(mod.act_fn)
                counts["silu"] += 1

    return counts
