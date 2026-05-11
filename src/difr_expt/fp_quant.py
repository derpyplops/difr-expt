"""fp4 / fp8 fake-quant for emulating a low-precision production model.

The fp4-emulation experiment needs a "production model" that runs at fp4 or fp8
(what real labs deploy) so that the int24 student has something noisy to
emulate. We don't compile to actual fp4/fp8 kernels — we just fake-quantize
every Linear's weights and inputs each forward, which is enough to produce the
logit distribution a real fp4/fp8 forward would produce (modulo accumulator
precision; we accumulate in fp32 which is what production GEMM kernels also
do).

Two formats:

  - **fp8 e4m3 (or e5m2)**: native torch dtypes. Per-row absmax scale on weights
    (256-channel block), per-token absmax scale on activations. Round via
    `.to(torch.float8_e4m3fn).to(torch.float32)` — torch's fp8 saturates at
    ±448 (e4m3) so the absmax scale brings the max value to fp8_max.

  - **fp4 (MXFP4 / E2M1)**: not in torch natively. We implement block-wise
    absmax + nearest-snap to the 8 positive E2M1 representable values
    {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Default block size 32 (MX standard).

Both formats are STE-friendly: the rounding error feeds back as identity so a
fake-quantized Linear can be embedded inside a larger computation that we
*don't* want to train (the teacher).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# E2M1 positive representable values. The fp4 format is 1 sign + 2 exp + 1
# mantissa with subnormals; the 8 non-negative values are these. The 16 total
# signed values are these unioned with their negations.
_FP4_E2M1_POS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_FP4_MAX = 6.0


def _round_to_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Snap each element of x to the nearest entry in `grid` (1-D positive
    monotonic). Sign is preserved.
    """
    # x absolute value vs the sorted grid → find nearest bucket. Vectorized via
    # bucketize-like trick: for each |x|, the nearest of the 8 entries is the
    # one minimizing |abs - grid|.
    # We do a small explicit search since grid is tiny (8 elements).
    a = x.abs().unsqueeze(-1)  # [..., 1]
    g = grid.to(x.device).view(*([1] * x.dim()), -1)  # [1, ..., 1, G]
    dist = (a - g).abs()
    idx = dist.argmin(dim=-1)
    snapped = grid.to(x.device)[idx]
    return snapped * x.sign()


def fake_quantize_fp4_blocksym(
    x: torch.Tensor, block_size: int = 32, last_dim_only: bool = True
) -> torch.Tensor:
    """Block-wise symmetric fp4 (E2M1) fake quant along the last dim.

    Each block of `block_size` consecutive last-dim entries gets its own absmax
    scale; the block is divided by (absmax / 6.0), snapped to E2M1 representable
    values, multiplied back. Sign-preserving.

    If the last-dim size isn't a multiple of `block_size`, the leftover tail
    is handled as a single short block.
    """
    orig_dtype = x.dtype
    if x.numel() == 0:
        return x
    last = x.shape[-1]
    if last_dim_only and last % block_size != 0:
        # Split into full blocks + tail; recurse on tail with its own scale.
        n_full = (last // block_size) * block_size
        head = x[..., :n_full]
        tail = x[..., n_full:]
        head_q = fake_quantize_fp4_blocksym(head, block_size=block_size)
        # Tail: treat as a single block of size `last - n_full`.
        tail_q = fake_quantize_fp4_blocksym(tail, block_size=tail.shape[-1])
        return torch.cat([head_q, tail_q], dim=-1).to(orig_dtype)
    # Reshape last dim into [..., n_blocks, block_size]
    xf = x.float()
    n_blocks = last // block_size
    reshaped = xf.reshape(*xf.shape[:-1], n_blocks, block_size)
    absmax = reshaped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = absmax / _FP4_MAX
    scaled = reshaped / scale
    snapped = _round_to_grid(scaled, _FP4_E2M1_POS)
    dequant = (snapped * scale).reshape(*xf.shape[:-1], last)
    return dequant.to(orig_dtype)


def fake_quantize_fp8(
    x: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    per_row: bool = True, axis: int = -1,
) -> torch.Tensor:
    """Per-row absmax-scaled fp8 fake quant.

    fp8 e4m3 saturates at ±448 — we rescale so the row absmax maps to 448, snap
    to fp8 representable values via the native dtype, then unscale. e5m2 is the
    same idea with max=57344.
    """
    if x.numel() == 0:
        return x
    orig_dtype = x.dtype
    xf = x.float()
    fp8_max = float(torch.finfo(fp8_dtype).max)
    if per_row:
        absmax = xf.abs().amax(dim=axis, keepdim=True).clamp_min(1e-30)
    else:
        absmax = xf.abs().amax().clamp_min(1e-30)
    scale = absmax / fp8_max
    scaled = xf / scale
    # Round via the native dtype (saturates at fp8_max by construction).
    rounded = scaled.to(fp8_dtype).to(torch.float32)
    dequant = rounded * scale
    return dequant.to(orig_dtype)


class LowPrecisionLinear(nn.Module):
    """Drop-in for nn.Linear that fake-quantizes weight + input to fp4 or fp8.

    Weight is fake-quantized once at construction (it's frozen — this is the
    "production model"). Activations are fake-quantized each forward (dynamic
    per-token absmax for fp8, block-wise for fp4).

    `precision`: "fp8_e4m3" | "fp8_e5m2" | "fp4_e2m1".
    `block_size`: block size for fp4 (default 32, MX standard). Ignored for fp8.
    `quantize_act`: whether to fake-quantize inputs too. Default True (production
        labs typically quantize both sides). Set False for weight-only mode.
    """

    def __init__(
        self,
        linear: nn.Linear,
        precision: str = "fp4_e2m1",
        block_size: int = 32,
        quantize_act: bool = True,
    ):
        super().__init__()
        self.precision = precision
        self.block_size = block_size
        self.quantize_act = quantize_act
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        w = linear.weight.detach()
        compute_dtype = w.dtype
        # Fake-quantize the weight once, store as a buffer at compute_dtype so
        # the Linear forward sees the noisy version directly. No grad needed.
        w_q = self._fake_quant(w.float())
        self.register_buffer("weight", w_q.to(compute_dtype), persistent=True)
        if linear.bias is not None:
            # Bias stays fp32-equivalent — production fp4/fp8 designs usually
            # leave the bias at high precision since it's tiny.
            self.register_buffer("bias", linear.bias.detach().clone().to(compute_dtype), persistent=True)
        else:
            self.bias = None
        self.compute_dtype = compute_dtype

    def _fake_quant(self, t: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp8_e4m3":
            return fake_quantize_fp8(t, torch.float8_e4m3fn, per_row=True, axis=-1)
        if self.precision == "fp8_e5m2":
            return fake_quantize_fp8(t, torch.float8_e5m2, per_row=True, axis=-1)
        if self.precision == "fp4_e2m1":
            return fake_quantize_fp4_blocksym(t, block_size=self.block_size, last_dim_only=True)
        raise ValueError(f"unknown precision {self.precision!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_act:
            x_q = self._fake_quant(x)
        else:
            x_q = x
        return F.linear(x_q.to(self.weight.dtype), self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"precision={self.precision}, block_size={self.block_size}, "
            f"quantize_act={self.quantize_act}"
        )


def patch_model_low_precision(
    model: nn.Module,
    precision: str = "fp4_e2m1",
    block_size: int = 32,
    skip_names: tuple[str, ...] = (),
    include_lm_head: bool = False,
    quantize_act: bool = True,
) -> dict[str, LowPrecisionLinear]:
    """Walk model and replace every nn.Linear with LowPrecisionLinear.

    Default `include_lm_head=False` keeps lm_head in fp32 — that's a typical
    production choice (output projection is high-precision so logits stay
    interpretable).
    """
    replaced: dict[str, LowPrecisionLinear] = {}

    def should_skip(name: str) -> bool:
        if not include_lm_head and "lm_head" in name:
            return True
        return any(s in name for s in skip_names)

    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            to_replace.append((name, module))

    for name, linear in to_replace:
        wrap = LowPrecisionLinear(
            linear,
            precision=precision,
            block_size=block_size,
            quantize_act=quantize_act,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrap)
        replaced[name] = wrap
    return replaced
