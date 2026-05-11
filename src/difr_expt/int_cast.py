"""Cast nn.Linear to full int conversion (weights + activations).

For ZK-proof-compatible inference, both weights AND activations of every matmul
must be integers, with a public float scale that multiplies the int matmul
output. This module wraps `nn.Linear` to implement that pipeline.

Storage: int32 tensors, but effective bit width is capped (default 16) to keep
the matmul accumulator inside int64. With dim 14336 (Llama 3.1 8B MLP):
  - 16-bit * 16-bit per product = 30 bits
  - sum of 14336 such products ≈ 2^44, well inside int64
  - 32-bit * 32-bit would overflow.

Compute: by default we run `(W_int * sw) @ (x_int * sa).T` as a fp32 matmul.
This is mathematically equal to true int matmul plus dequant in exact
arithmetic, and differs only by float reduction order — which is the same
noise the float reference accumulates. It runs at normal GPU speed. The
`true_int_matmul=True` flag switches to literal `int32@int32 -> int64` (CPU
fallback when on GPU, à la Luke's repo) for verification runs.

RMSNorm, RoPE, softmax stay float. Those are a separate research thread.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _qmax(bits: int) -> int:
    if bits < 2 or bits > 31:
        raise ValueError(f"effective bits must be in [2, 31], got {bits}")
    return (1 << (bits - 1)) - 1


def quantize_per_row(
    weight: torch.Tensor, bits: int = 16, store_dtype: torch.dtype = torch.int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-row quantization.

    weight: [out_features, in_features] (Linear convention).

    Returns (weight_int, scale). scale has shape [out_features, 1] in fp32.
    weight_int is stored in `store_dtype` (default int32) but its values are
    clamped to ±qmax for the requested effective bit width.

    Math is done in fp64 to avoid float32 precision loss at large magnitudes.
    """
    assert weight.dim() == 2, f"expected 2D weight, got {weight.shape}"
    w64 = weight.detach().to(torch.float64)
    qmax = _qmax(bits)
    row_absmax = w64.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = row_absmax / qmax  # fp64
    weight_q = torch.round(w64 / scale).clamp(-qmax, qmax)
    return weight_q.to(store_dtype), scale.to(torch.float32)


def quantize_per_token(
    x: torch.Tensor, bits: int = 16, store_dtype: torch.dtype = torch.int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (per-token) symmetric quantization of a 2D activation tensor.

    x: [n_tokens, in_features]. Returns (x_int, scale) where scale is fp32 of
    shape [n_tokens, 1]. Math is done in the dtype the input arrived in
    (typically bf16/fp16); we don't promote to fp64 in the hot path.
    """
    qmax = _qmax(bits)
    xf = x.float()  # fp32 is enough for activations; fp64 would crater speed
    row_absmax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = row_absmax / qmax  # fp32
    x_q = torch.round(xf / scale).clamp(-qmax, qmax)
    return x_q.to(store_dtype), scale


class IntLinear(nn.Module):
    """Full int conversion: weights AND activations quantized.

    Forward does:
        x_int, sa = quantize_per_token(x, bits=activation_bits)
        out = (W_int.float() * sw) @ (x_int.float() * sa).T + bias  (default)
        # or with true_int_matmul=True:
        out = (W_int.int64 @ x_int.int64.T).float() * sw * sa + bias

    Both produce the same result in exact arithmetic; the float path runs at
    GPU speed, the true-int path uses CPU fallback (torch lacks int matmul on
    CUDA) and is meant for verification.
    """

    true_int_matmul: bool = False

    def __init__(
        self,
        weight_int: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        weight_bits: int,
        activation_bits: int,
    ):
        super().__init__()
        self.register_buffer("weight_int", weight_int, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias.to(compute_dtype), persistent=True)
        else:
            self.bias = None
        self.compute_dtype = compute_dtype
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.out_features, self.in_features = weight_int.shape

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_bits: int = 16,
        activation_bits: int = 16,
    ) -> "IntLinear":
        weight = linear.weight.detach()
        compute_dtype = weight.dtype
        weight_int, weight_scale = quantize_per_row(weight, bits=weight_bits)
        return cls(
            weight_int=weight_int,
            weight_scale=weight_scale,
            bias=linear.bias.detach() if linear.bias is not None else None,
            compute_dtype=compute_dtype,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
        )

    def _float_path(self, x_flat: torch.Tensor) -> torch.Tensor:
        # Quantize activations, dequant both sides, do an fp32 matmul. The
        # dequantized values are bit-identical to what a real int circuit would
        # multiply by its public scales, so this is faithful to the int
        # representation; only the matmul reduction order differs (and that's
        # the same kind of noise the float reference carries).
        x_int, x_scale = quantize_per_token(x_flat, bits=self.activation_bits)
        w_deq = self.weight_int.to(torch.float32) * self.weight_scale     # [O, I] fp32
        x_deq = x_int.to(torch.float32) * x_scale                          # [N, I] fp32
        return x_deq @ w_deq.t()

    def _true_int_path(self, x_flat: torch.Tensor) -> torch.Tensor:
        # int32 -> int64 accumulate. torch.matmul on int dtypes works only on
        # CPU, so kick a CPU roundtrip when we're on CUDA. Slow; intended for
        # verification runs, not the default measurement pass.
        x_int, x_scale = quantize_per_token(x_flat, bits=self.activation_bits)
        w_i64 = self.weight_int.to(torch.int64)
        x_i64 = x_int.to(torch.int64)
        if x_flat.device.type == "cuda":
            accum = torch.matmul(x_i64.cpu(), w_i64.cpu().t()).to(x_flat.device)
        else:
            accum = torch.matmul(x_i64, w_i64.t())
        return accum.to(torch.float32) * x_scale * self.weight_scale.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        if self.true_int_matmul:
            out_flat = self._true_int_path(x_flat)
        else:
            out_flat = self._float_path(x_flat)
        out = out_flat.to(self.compute_dtype).reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"w_bits={self.weight_bits}, a_bits={self.activation_bits}, "
            f"compute_dtype={self.compute_dtype}"
        )


def set_true_int_matmul(model: nn.Module, value: bool) -> None:
    """Toggle the int-matmul path across every IntLinear in the model."""
    for m in model.modules():
        if isinstance(m, IntLinear):
            m.true_int_matmul = value


def patch_model_int_cast(
    model: nn.Module,
    weight_bits: int = 16,
    activation_bits: int = 16,
    skip_names: tuple[str, ...] = (),
    include_lm_head: bool = True,
) -> dict[str, IntLinear]:
    """Walk `model` and replace every nn.Linear with an IntLinear.

    Default is full int conversion at 16 effective bits, matching Luke's
    `int-model-approximation` setup (and the constraint imposed by int64
    accumulator overflow at Llama-3.1 8B's hidden dims).
    """
    replaced: dict[str, IntLinear] = {}

    def should_skip(name: str) -> bool:
        if not include_lm_head and "lm_head" in name:
            return True
        return any(s in name for s in skip_names)

    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            to_replace.append((name, module))

    for name, linear in to_replace:
        int_lin = IntLinear.from_linear(linear, weight_bits=weight_bits, activation_bits=activation_bits)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, int_lin)
        replaced[name] = int_lin

    return replaced
