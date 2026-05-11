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

Two operating modes:
  - **frozen** (default, used by the baseline experiment): weight_int and
    weight_scale are buffers fixed at construction time.
  - **trainable** (used by the training experiment): weight_fp is a
    nn.Parameter (fp32 shadow of the original weight); weight_int and
    weight_scale are recomputed every forward pass via the STE trick so
    gradients flow through weight_fp.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def quantize_per_row_asym(
    weight: torch.Tensor, bits: int = 16, store_dtype: torch.dtype = torch.int32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-output-row quantization with zero-point.

    Range is [qmin=-(qmax+1), qmax] (full signed int).
    scale = (max - min) / (qmax - qmin); zp = round(qmin - min/scale).
    q = clamp(round(x/scale + zp), qmin, qmax). Dequant: x = (q - zp) * scale.

    Returns (weight_int, scale, zp) with scale [out, 1] fp32 and zp [out, 1] int32.
    """
    assert weight.dim() == 2, f"expected 2D weight, got {weight.shape}"
    w64 = weight.detach().to(torch.float64)
    qmax = _qmax(bits)
    qmin = -(qmax + 1)
    row_max = w64.amax(dim=1, keepdim=True)
    row_min = w64.amin(dim=1, keepdim=True)
    rng = (row_max - row_min).clamp_min(1e-30)
    scale = rng / float(qmax - qmin)  # fp64
    zp = torch.round(qmin - row_min / scale)
    weight_q = torch.round(w64 / scale + zp).clamp(qmin, qmax)
    return weight_q.to(store_dtype), scale.to(torch.float32), zp.to(store_dtype)


def quantize_per_token_asym(
    x: torch.Tensor, bits: int = 16, store_dtype: torch.dtype = torch.int32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-token quantization. x: [n_tokens, features]."""
    qmax = _qmax(bits)
    qmin = -(qmax + 1)
    xf = x.float()
    row_max = xf.amax(dim=1, keepdim=True)
    row_min = xf.amin(dim=1, keepdim=True)
    rng = (row_max - row_min).clamp_min(1e-30)
    scale = rng / float(qmax - qmin)
    zp = torch.round(qmin - row_min / scale)
    x_q = torch.round(xf / scale + zp).clamp(qmin, qmax)
    return x_q.to(store_dtype), scale, zp.to(store_dtype)


def quantize_per_row_groupsym(
    weight: torch.Tensor,
    bits: int = 16,
    group_size: int = 128,
    store_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric quantization along the in-features dim.

    weight: [out, in], group_size divides in (or pads with last-group).
    Returns (weight_int [out, in], scale [out, n_groups, 1]) — scale is fp32.
    Math in fp64. For fine groups (g=8) on large weights, the fp64 [O, ng, G]
    intermediate can exceed GPU memory — compute on CPU when called with a
    cuda tensor and shift result back to original device.
    """
    assert weight.dim() == 2, f"expected 2D weight, got {weight.shape}"
    O, I = weight.shape
    if I % group_size != 0:
        raise ValueError(
            f"in_features={I} not divisible by group_size={group_size}; "
            "padding not implemented"
        )
    n_groups = I // group_size
    qmax = _qmax(bits)
    orig_device = weight.device
    # CPU compute when the fp64 buffer would be too big to be safe on GPU.
    # 4.6 GiB at O=4096, ng=1792 (=14336/8), G=8 — pushes the cuda alloc over
    # the H100 ceiling when stacked with the rest. Compute on CPU instead.
    w64 = weight.detach().to("cpu", torch.float64).view(O, n_groups, group_size)
    row_absmax = w64.abs().amax(dim=2, keepdim=True).clamp_min(1e-30)
    scale = row_absmax / qmax
    weight_q = torch.round(w64 / scale).clamp(-qmax, qmax)
    return (
        weight_q.view(O, I).to(store_dtype).to(orig_device),
        scale.to(torch.float32).to(orig_device),
    )


def quantize_per_token_groupsym(
    x: torch.Tensor, bits: int = 16, group_size: int = 128, store_dtype: torch.dtype = torch.int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric per-token quant. x: [n_tokens, features]."""
    N, I = x.shape
    if I % group_size != 0:
        raise ValueError(f"in_features={I} not divisible by group_size={group_size}")
    n_groups = I // group_size
    xf = x.float().view(N, n_groups, group_size)
    qmax = _qmax(bits)
    row_absmax = xf.abs().amax(dim=2, keepdim=True).clamp_min(1e-30)
    scale = row_absmax / qmax
    x_q = torch.round(xf / scale).clamp(-qmax, qmax)
    return x_q.view(N, I).to(store_dtype), scale


def quantize_per_token(
    x: torch.Tensor, bits: int = 16, store_dtype: torch.dtype = torch.int32,
    clip_quantile: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (per-token) symmetric quantization of a 2D activation tensor.

    x: [n_tokens, in_features]. Returns (x_int, scale) where scale is fp32 of
    shape [n_tokens, 1]. Math is done in the dtype the input arrived in
    (typically bf16/fp16); we don't promote to fp64 in the hot path.

    If `clip_quantile` is provided (e.g. 0.999), the per-token scale is set to
    the `clip_quantile` of the absolute values instead of the absmax — so the
    top 0.1% of channels saturate to ±qmax. This trades a tiny fraction of
    outlier-channel precision for finer quantization of the bulk.
    """
    qmax = _qmax(bits)
    xf = x.float()  # fp32 is enough for activations; fp64 would crater speed
    abs_x = xf.abs()
    if clip_quantile is not None and clip_quantile < 1.0:
        row_ref = abs_x.quantile(clip_quantile, dim=1, keepdim=True).clamp_min(1e-30)
    else:
        row_ref = abs_x.amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = row_ref / qmax  # fp32
    x_q = torch.round(xf / scale).clamp(-qmax, qmax)
    return x_q.to(store_dtype), scale


def fake_quantize_per_row_ste(weight_fp: torch.Tensor, bits: int) -> torch.Tensor:
    """Fake-quantize a fp32 weight with straight-through gradient.

    Forward: returns the dequantized round-clamp-rescale value. Backward:
    identity in weight_fp (so the gradient sees the quant op as a no-op). The
    scale is computed from weight_fp.abs().amax() and is `detach()`-ed inside
    the difference so the optimizer doesn't backprop through it — this is
    the standard QAT convention.
    """
    assert weight_fp.dim() == 2, f"expected 2D weight, got {weight_fp.shape}"
    qmax = _qmax(bits)
    # Stay in fp32 here — fp64 would be slow on GPU, and fp32 is precise
    # enough for the per-row scale at qmax = 2^15-1.
    row_absmax = weight_fp.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = row_absmax / qmax  # fp32, no grad
    w_q = torch.round(weight_fp / scale).clamp(-qmax, qmax)
    w_dq = w_q * scale
    # STE: replace the forward with w_dq, but route gradient as identity.
    return weight_fp + (w_dq - weight_fp).detach()


def fake_quantize_per_token_ste(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Same STE trick for per-token (dynamic) activation quant.

    x is 2D [n_tokens, features]. Scale is detached so only the rounding
    error feeds back through the identity path. Activation quant is not
    "trainable" — there's nothing to learn — but the STE path lets the
    upstream Linear see roughly the same gradient signal it would without
    quant.
    """
    qmax = _qmax(bits)
    row_absmax = x.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = row_absmax / qmax
    x_q = torch.round(x / scale).clamp(-qmax, qmax)
    x_dq = x_q * scale
    return x + (x_dq - x).detach()


class IntLinear(nn.Module):
    """Full int conversion: weights AND activations quantized.

    Forward (frozen mode, ``trainable=False``):
        x_int, sa = quantize_per_token(x, bits=activation_bits)
        out = (W_int.float() * sw) @ (x_int.float() * sa).T + bias   # default
        # or with true_int_matmul=True:
        out = (W_int.int64 @ x_int.int64.T).float() * sw * sa + bias

    Forward (trainable mode, ``trainable=True``):
        W_q = fake_quantize_per_row_ste(W_fp, weight_bits)
        x_q = fake_quantize_per_token_ste(x, activation_bits)
        out = x_q @ W_q.T + bias

    Both paths are numerically identical in exact arithmetic; trainable mode
    additionally lets a gradient flow back to the fp32 W_fp parameter.
    """

    true_int_matmul: bool = False

    def __init__(
        self,
        weight_int: torch.Tensor | None,
        weight_scale: torch.Tensor | None,
        weight_fp: torch.Tensor | None,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        weight_bits: int,
        activation_bits: int,
        quant_scheme: str = "symmetric",
        group_size: int = 128,
        weight_zp: torch.Tensor | None = None,
        matmul_dtype: torch.dtype | None = None,
        smooth_scale: torch.Tensor | None = None,
        act_clip_quantile: float | None = None,
        cached_dequant_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        # Exactly one of (weight_int+weight_scale) or weight_fp is present —
        # the other gets registered as None for state-dict cleanliness.
        if weight_fp is not None:
            self.weight_fp = nn.Parameter(weight_fp.detach().clone().float())
            self.register_buffer("weight_int", None, persistent=False)
            self.register_buffer("weight_scale", None, persistent=False)
            self.register_buffer("weight_zp", None, persistent=False)
            self.out_features, self.in_features = weight_fp.shape
        else:
            assert weight_int is not None and weight_scale is not None
            self.weight_fp = None
            self.register_buffer("weight_int", weight_int, persistent=True)
            self.register_buffer("weight_scale", weight_scale, persistent=True)
            if weight_zp is not None:
                self.register_buffer("weight_zp", weight_zp, persistent=True)
            else:
                self.register_buffer("weight_zp", None, persistent=False)
            self.out_features, self.in_features = weight_int.shape
        if bias is not None:
            self.register_buffer("bias", bias.to(compute_dtype), persistent=True)
        else:
            self.bias = None
        # SmoothQuant per-channel rescaler: shape [in_features], fp32. If not
        # None, activations are divided by smooth_scale before quant; weights
        # were pre-multiplied by smooth_scale at construction time. The result
        # is mathematically identical, but redistributes outliers from the
        # activation axis to the weight axis (where per-row absmax handles them
        # well).
        if smooth_scale is not None:
            self.register_buffer("smooth_scale", smooth_scale, persistent=True)
        else:
            self.register_buffer("smooth_scale", None, persistent=False)
        self.compute_dtype = compute_dtype
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quant_scheme = quant_scheme
        self.group_size = group_size
        # matmul_dtype=None means "auto" — current default behaviour (fp32 matmul).
        # Set explicitly to torch.bfloat16/float16 to cast dequant outputs to that
        # dtype before the matmul, so it matches the reference path's reduction
        # order. Only affects _float_path; _true_int_path is unchanged.
        self.matmul_dtype = matmul_dtype
        self.act_clip_quantile = act_clip_quantile
        # Optional: pre-dequantized weight stored as an nn.Parameter at
        # `compute_dtype` (matches original Linear.weight shape/stride/contig).
        # When set, `_float_path` skips the weight-side dequant chain and uses
        # this Parameter directly — guarantees identical cuBLAS dispatch with
        # the reference's nn.Linear because torch sees the same tensor shape.
        # Activation quant still runs as configured.
        if cached_dequant_weight is not None:
            self.weight_dequant = nn.Parameter(
                cached_dequant_weight.detach().clone(), requires_grad=False
            )
        else:
            self.weight_dequant = None

    @property
    def trainable(self) -> bool:
        return self.weight_fp is not None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_bits: int = 16,
        activation_bits: int = 16,
        trainable: bool = False,
        quant_scheme: str = "symmetric",
        group_size: int = 128,
        matmul_dtype: torch.dtype | None = None,
        smooth_scale: torch.Tensor | None = None,
        act_clip_quantile: float | None = None,
        cached_bf16: bool = False,
    ) -> "IntLinear":
        weight = linear.weight.detach()
        compute_dtype = weight.dtype
        if trainable:
            # Float shadow in fp32 — fp64 would double memory; fp32 is fine.
            return cls(
                weight_int=None,
                weight_scale=None,
                weight_fp=weight.to(torch.float32),
                bias=linear.bias.detach() if linear.bias is not None else None,
                compute_dtype=compute_dtype,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                quant_scheme=quant_scheme,
                group_size=group_size,
                matmul_dtype=matmul_dtype,
                smooth_scale=smooth_scale,
                act_clip_quantile=act_clip_quantile,
            )
        # If SmoothQuant rescaler is provided, pre-multiply weights by it along
        # the in_features axis before quantizing. The forward path will divide
        # activations by the same scale, preserving the matmul output exactly.
        if smooth_scale is not None:
            assert smooth_scale.dim() == 1 and smooth_scale.shape[0] == weight.shape[1], (
                f"smooth_scale shape {smooth_scale.shape} must match in_features={weight.shape[1]}"
            )
            weight = weight * smooth_scale.to(weight.dtype).unsqueeze(0)
        weight_zp = None
        if quant_scheme == "symmetric":
            weight_int, weight_scale = quantize_per_row(weight, bits=weight_bits)
        elif quant_scheme == "asymmetric":
            weight_int, weight_scale, weight_zp = quantize_per_row_asym(weight, bits=weight_bits)
        elif quant_scheme == "per_group_sym":
            weight_int, weight_scale = quantize_per_row_groupsym(
                weight, bits=weight_bits, group_size=group_size
            )
        else:
            raise ValueError(f"unknown quant_scheme={quant_scheme!r}")
        # Pre-dequantize the weight back to compute_dtype and stash as an
        # nn.Parameter when cached_bf16 is set. This guarantees the IntLinear's
        # F.linear call dispatches the same cuBLAS kernel as the reference
        # nn.Linear (identical shape/stride/contiguity/dtype of the weight
        # operand). The "quantization error" is then strictly the int -> bf16
        # rounding from quant+dequant, which is what we want to characterize.
        cached_dequant_weight = None
        if cached_bf16:
            O, I = weight.shape
            if quant_scheme == "symmetric":
                w_deq = weight_int.to(torch.float32) * weight_scale  # [O, I] fp32
            elif quant_scheme == "asymmetric":
                w_deq = (
                    weight_int.to(torch.float32) - weight_zp.to(torch.float32)
                ) * weight_scale
            elif quant_scheme == "per_group_sym":
                n_groups = I // group_size
                w_deq = (
                    weight_int.to(torch.float32).view(O, n_groups, group_size)
                    * weight_scale
                ).view(O, I)
            else:
                raise ValueError(f"unknown quant_scheme={quant_scheme!r}")
            cached_dequant_weight = w_deq.to(compute_dtype)
        return cls(
            weight_int=weight_int,
            weight_scale=weight_scale,
            weight_fp=None,
            bias=linear.bias.detach() if linear.bias is not None else None,
            compute_dtype=compute_dtype,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            quant_scheme=quant_scheme,
            group_size=group_size,
            weight_zp=weight_zp,
            matmul_dtype=matmul_dtype,
            smooth_scale=smooth_scale,
            act_clip_quantile=act_clip_quantile,
            cached_dequant_weight=cached_dequant_weight,
        )

    def freeze_to_int(self) -> None:
        """Bake the current fp32 shadow into fixed int+scale buffers.

        Used at the end of training to produce a self-contained inference
        artifact. After this, the module is in frozen mode and the float
        shadow is dropped.
        """
        if not self.trainable:
            return
        with torch.no_grad():
            w_int, w_scale = quantize_per_row(self.weight_fp.data, bits=self.weight_bits)
        # Drop the Parameter so it can't accidentally receive a grad.
        del self._parameters["weight_fp"]
        self.weight_fp = None
        # Replace the placeholder None buffers with real tensors.
        self.weight_int = w_int
        self.weight_scale = w_scale

    def _trainable_forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # STE on both sides.
        w_ste = fake_quantize_per_row_ste(self.weight_fp, self.weight_bits)
        x_ste = fake_quantize_per_token_ste(x_flat.float(), self.activation_bits)
        return x_ste @ w_ste.t()

    def _float_path(self, x_flat: torch.Tensor) -> torch.Tensor:
        # If matmul_dtype is set (e.g. bf16/fp16), cast the dequantized operands
        # to that dtype before the matmul AND route through F.linear so the bias
        # is added inside cuBLAS' addmm at higher precision — exactly matching
        # nn.Linear's reference path. Default (None) preserves the original
        # fp32 matmul behaviour and uses the plain matmul (bias applied later
        # in forward()).
        md = self.matmul_dtype
        # Pre-cast bias to matmul_dtype once if needed; bias is already bf16/fp16
        # in compute_dtype storage so this is normally a no-op.
        bias_md = None
        if md is not None and self.bias is not None:
            bias_md = self.bias.to(md)
        # activation_bits=0 => sentinel for "skip activation quantization"; pass
        # the float activations straight through. Useful for weight-only PTQ
        # ablations.
        skip_act = self.activation_bits == 0

        # SmoothQuant: divide activations by per-channel smooth_scale before
        # quant (the weights have already been pre-multiplied by it).
        if self.smooth_scale is not None:
            x_flat = x_flat / self.smooth_scale.to(x_flat.dtype)

        # Cached-bf16 weight fast path: use the pre-dequantized weight stored
        # as nn.Parameter (matches reference nn.Linear's weight tensor shape
        # exactly). Still runs activation quant if activation_bits > 0.
        if self.weight_dequant is not None:
            # Pick matmul dtype: explicit matmul_dtype overrides; else the
            # cached weight's storage dtype (compute_dtype).
            tgt = md if md is not None else self.weight_dequant.dtype
            if skip_act:
                x_in = x_flat.to(tgt)
            else:
                if self.quant_scheme == "symmetric":
                    x_int, x_scale = quantize_per_token(
                        x_flat, bits=self.activation_bits,
                        clip_quantile=self.act_clip_quantile,
                    )
                    x_deq = x_int.to(torch.float32) * x_scale
                elif self.quant_scheme == "asymmetric":
                    x_int, x_scale, x_zp = quantize_per_token_asym(
                        x_flat, bits=self.activation_bits
                    )
                    x_deq = (x_int.to(torch.float32) - x_zp.to(torch.float32)) * x_scale
                elif self.quant_scheme == "per_group_sym":
                    G = self.group_size
                    n_groups = self.in_features // G
                    x_int, x_scale = quantize_per_token_groupsym(
                        x_flat, bits=self.activation_bits, group_size=G
                    )
                    x_deq = (x_int.to(torch.float32).view(-1, n_groups, G) * x_scale).view(-1, self.in_features)
                else:
                    raise ValueError(f"unknown quant_scheme={self.quant_scheme!r}")
                x_in = x_deq.to(tgt)
            w_in = self.weight_dequant.to(tgt) if self.weight_dequant.dtype != tgt else self.weight_dequant
            b_in = None
            if self.bias is not None:
                b_in = self.bias.to(tgt) if self.bias.dtype != tgt else self.bias
            return F.linear(x_in, w_in, b_in)

        if self.quant_scheme == "symmetric":
            w_deq = self.weight_int.to(torch.float32) * self.weight_scale     # [O, I] fp32
            if skip_act:
                x_deq = x_flat.float()
            else:
                x_int, x_scale = quantize_per_token(
                    x_flat, bits=self.activation_bits,
                    clip_quantile=self.act_clip_quantile,
                )
                x_deq = x_int.to(torch.float32) * x_scale                      # [N, I] fp32
            if md is not None:
                w_deq = w_deq.to(md)
                x_deq = x_deq.to(md)
                return F.linear(x_deq, w_deq, bias_md)
            return x_deq @ w_deq.t()
        if self.quant_scheme == "asymmetric":
            w_deq = (self.weight_int.to(torch.float32) - self.weight_zp.to(torch.float32)) * self.weight_scale
            if skip_act:
                x_deq = x_flat.float()
            else:
                x_int, x_scale, x_zp = quantize_per_token_asym(x_flat, bits=self.activation_bits)
                x_deq = (x_int.to(torch.float32) - x_zp.to(torch.float32)) * x_scale
            if md is not None:
                w_deq = w_deq.to(md)
                x_deq = x_deq.to(md)
                return F.linear(x_deq, w_deq, bias_md)
            return x_deq @ w_deq.t()
        if self.quant_scheme == "per_group_sym":
            O = self.out_features
            I = self.in_features
            G = self.group_size
            # Dequant W: weight_int [O, I], weight_scale [O, n_groups, 1]
            n_groups = I // G
            w_deq = (self.weight_int.to(torch.float32).view(O, n_groups, G) * self.weight_scale).view(O, I)
            if skip_act:
                x_deq = x_flat.float()
            else:
                x_int, x_scale = quantize_per_token_groupsym(
                    x_flat, bits=self.activation_bits, group_size=G
                )  # x_int [N, I], x_scale [N, n_groups, 1]
                x_deq = (x_int.to(torch.float32).view(-1, n_groups, G) * x_scale).view(-1, I)
            if md is not None:
                w_deq = w_deq.to(md)
                x_deq = x_deq.to(md)
                return F.linear(x_deq, w_deq, bias_md)
            return x_deq @ w_deq.t()
        raise ValueError(f"unknown quant_scheme={self.quant_scheme!r}")

    def _true_int_path(self, x_flat: torch.Tensor) -> torch.Tensor:
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
        bias_already_applied = False
        if self.trainable:
            out_flat = self._trainable_forward(x_flat)
        elif self.true_int_matmul:
            out_flat = self._true_int_path(x_flat)
        else:
            out_flat = self._float_path(x_flat)
            # When matmul_dtype is set OR the cached-bf16 weight path is in
            # use, _float_path uses F.linear with bias fused inside cuBLAS
            # addmm. Don't add bias again.
            if (
                self.matmul_dtype is not None or self.weight_dequant is not None
            ) and self.bias is not None:
                bias_already_applied = True
        out = out_flat.to(self.compute_dtype).reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None and not bias_already_applied:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        mode = "trainable" if self.trainable else "frozen"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"w_bits={self.weight_bits}, a_bits={self.activation_bits}, "
            f"compute_dtype={self.compute_dtype}, mode={mode}"
        )


def set_true_int_matmul(model: nn.Module, value: bool) -> None:
    """Toggle the int-matmul path across every IntLinear in the model."""
    for m in model.modules():
        if isinstance(m, IntLinear):
            m.true_int_matmul = value


def calibrate_smooth_scales(
    model: nn.Module,
    calibration_inputs: list[torch.Tensor],
    device: str,
    alpha: float = 0.5,
    skip_names: tuple[str, ...] = (),
    include_lm_head: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute SmoothQuant per-channel rescalers from a small calibration set.

    For each nn.Linear (matching the same skip rules `patch_model_int_cast`
    uses), runs the model on the provided inputs to collect per-channel max
    absolute input activation. Combines with per-channel max absolute weight
    column to compute `s_j = act_max[j]^alpha / w_col_max[j]^(1-alpha)`, the
    canonical SmoothQuant rescaler (alpha=0.5 default). Returns a dict mapping
    qualified Linear name → `s` tensor of shape [in_features], fp32 on CPU.

    Channels with `s_j == 0` (e.g. embedding-table column never seen in the
    calibration data) are bumped to 1.0 to avoid divide-by-zero downstream.
    """
    def should_skip(name: str) -> bool:
        if not include_lm_head and "lm_head" in name:
            return True
        return any(s in name for s in skip_names)

    # Collect target linears and their input absmax via forward pre-hooks.
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            targets.append((name, module))

    act_max: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(qname: str):
        def hook(_module, args):
            x = args[0].detach()
            # Reduce all batch/sequence dims, keep in_features as the channel axis.
            x_flat = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float().cpu()
            if qname in act_max:
                act_max[qname] = torch.maximum(act_max[qname], x_flat)
            else:
                act_max[qname] = x_flat
        return hook

    for name, lin in targets:
        handles.append(lin.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.inference_mode():
        for ids in calibration_inputs:
            input_ids = ids.to(device) if hasattr(ids, "to") else torch.tensor(ids, device=device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            _ = model(input_ids)

    for h in handles:
        h.remove()

    # Compute the smooth scales.
    scales: dict[str, torch.Tensor] = {}
    for name, lin in targets:
        a = act_max.get(name)
        if a is None:
            continue
        w = lin.weight.detach().float().cpu()  # [O, I]
        w_col_max = w.abs().amax(dim=0).clamp_min(1e-8)  # [I]
        a = a.clamp_min(1e-8)
        s = (a.pow(alpha) / w_col_max.pow(1 - alpha)).clamp_min(1e-5)
        scales[name] = s
    return scales


class IntEmbedding(nn.Module):
    """Drop-in for nn.Embedding with per-row symmetric int24 quant of the table.

    The lookup itself is just indexing; there is no arithmetic to integerize.
    The values stored in the table are quantized: each vocab row gets a
    public fp32 scale `s_row`, and the stored int repr `e_int[row]` satisfies
    `e_float[row] ≈ e_int[row] * s_row`. Forward returns `e_int[ids] * s_ids`
    cast to `compute_dtype` — drop-in shape and dtype.

    No `_true_int_path` toggle is needed: there is no int arithmetic — only
    indexing + a per-row public-scale multiply at output. For ZKP the table
    is a committed array of integers; the verifier knows the public scales.
    """

    def __init__(
        self,
        weight_int: torch.Tensor,
        weight_scale: torch.Tensor,
        compute_dtype: torch.dtype,
        bits: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.register_buffer("weight_int", weight_int, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        self.compute_dtype = compute_dtype
        self.bits = bits
        self.padding_idx = padding_idx
        self.num_embeddings, self.embedding_dim = weight_int.shape

    @classmethod
    def from_embedding(
        cls,
        emb: nn.Embedding,
        bits: int = 24,
        store_dtype: torch.dtype = torch.int32,
    ) -> "IntEmbedding":
        w_int, scale = quantize_per_row(emb.weight, bits=bits, store_dtype=store_dtype)
        return cls(
            weight_int=w_int.to(emb.weight.device),
            weight_scale=scale.to(emb.weight.device),
            compute_dtype=emb.weight.dtype,
            bits=bits,
            padding_idx=emb.padding_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows_int = self.weight_int[input_ids]
        rows_scale = self.weight_scale[input_ids]
        out = rows_int.to(torch.float32) * rows_scale
        if self.padding_idx is not None:
            out = out.masked_fill(
                (input_ids == self.padding_idx).unsqueeze(-1), 0.0
            )
        return out.to(self.compute_dtype)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, bits={self.bits}"
        )


def patch_model_int_embedding(
    model: nn.Module,
    bits: int = 24,
    skip_names: tuple[str, ...] = (),
) -> dict[str, IntEmbedding]:
    """Walk `model` and replace every nn.Embedding with an IntEmbedding."""
    replaced: dict[str, IntEmbedding] = {}
    to_replace: list[tuple[str, nn.Embedding]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and not any(s in name for s in skip_names):
            to_replace.append((name, module))
    for name, emb in to_replace:
        int_emb = IntEmbedding.from_embedding(emb, bits=bits)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, int_emb)
        replaced[name] = int_emb
    return replaced


def patch_model_int_cast(
    model: nn.Module,
    weight_bits: int = 16,
    activation_bits: int = 16,
    skip_names: tuple[str, ...] = (),
    include_lm_head: bool = True,
    trainable: bool = False,
    quant_scheme: str = "symmetric",
    group_size: int = 128,
    matmul_dtype: torch.dtype | None = None,
    smooth_scales: dict[str, torch.Tensor] | None = None,
    act_clip_quantile: float | None = None,
    cached_bf16: bool = False,
) -> dict[str, IntLinear]:
    """Walk `model` and replace every nn.Linear with an IntLinear.

    With `trainable=True`, the replaced IntLinears carry an fp32 weight_fp
    Parameter (for STE training). With `trainable=False` (default, baseline
    behaviour) the int+scale buffers are fixed at construction.

    `quant_scheme` controls how each Linear is quantized: "symmetric" (default,
    per-row absmax), "asymmetric" (per-row min/max with zero-point), or
    "per_group_sym" (group_size-wise symmetric along in_features). The
    `per_group_sym` scheme falls back to standard per-row symmetric on any
    layer whose in_features is not divisible by `group_size` (e.g. tiny embed
    projections in some models).
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
        scheme = quant_scheme
        if scheme == "per_group_sym" and linear.in_features % group_size != 0:
            # Fall back to symmetric for this layer
            scheme = "symmetric"
        ss = None
        if smooth_scales is not None and name in smooth_scales:
            ss = smooth_scales[name].to(linear.weight.device).to(torch.float32)
        int_lin = IntLinear.from_linear(
            linear,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            trainable=trainable,
            quant_scheme=scheme,
            group_size=group_size,
            matmul_dtype=matmul_dtype,
            smooth_scale=ss,
            act_clip_quantile=act_clip_quantile,
            cached_bf16=cached_bf16,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, int_lin)
        replaced[name] = int_lin

    return replaced


def freeze_model_int(model: nn.Module) -> None:
    """Bake every trainable IntLinear in the model to frozen int+scale form."""
    for m in model.modules():
        if isinstance(m, IntLinear) and m.trainable:
            m.freeze_to_int()
