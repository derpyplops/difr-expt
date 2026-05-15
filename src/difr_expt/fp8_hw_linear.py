"""Real hardware FP8 Linear via `torch._scaled_mm`.

Problem: HF transformers + compressed_tensors loads FP8 checkpoints, but its
default integration installs a lazy *decompress hook* that, on first forward,
restores the weights to bf16/fp16 in memory. From that point on the model
runs as a plain bf16 model — there is no real FP8 GEMM. Verified empirically
with `ScaledMmProbe` in `run_harness.py`: zero calls to `torch._scaled_mm`
under the default HF FP8 path.

To actually exercise the Hopper FP8 tensor cores, we have to:

  1. Load the checkpoint (weights end up as `torch.float8_e4m3fn`, scales
     stored as `weight_scale`).
  2. Strip the decompress hook so the weights stay FP8.
  3. Replace every FP8 `nn.Linear` with `FP8Linear`, which on forward:
        - dynamically per-token-quantizes the bf16 input to FP8E4M3,
        - calls `torch._scaled_mm(x_fp8, W_fp8.T, scale_a=x_scale,
          scale_b=w_scale.T, out_dtype=bf16)` — a real Hopper FP8 GEMM,
        - adds the bias.

This matches the recipe encoded in
`RedHatAI/Qwen2.5-0.5B-FP8-dynamic`'s `quantization_config`: per-channel
symmetric weights, per-token dynamic symmetric activations, FP8 e4m3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Max representable absolute value in fp8 e4m3 (finite-only variant).
FP8_E4M3_MAX = 448.0


def _per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token symmetric quant to FP8 e4m3.

    x: [..., K] bf16/fp16/fp32. Returns (x_fp8 [M, K], scale [M, 1] fp32)
    where M = product of the leading dims of x. The scale is the value that,
    when multiplied back, recovers the original-scale fp matmul output.
    """
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    absmax = x2.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (absmax / FP8_E4M3_MAX).clamp_min(1e-12)
    x_fp8 = (x2.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


class FP8Linear(nn.Module):
    """Linear whose forward dispatches through `torch._scaled_mm` for a real
    Hopper FP8 GEMM. Matches the FP8-dynamic recipe used by RedHatAI:

        y = (x_fp8 @ W_fp8.T) * (s_x · s_W) + b

    where the matmul is the hardware FP8 path. We let `_scaled_mm` apply
    `scale_a · scale_b` and cast the result to `out_dtype` directly.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,   # [out, in], dtype torch.float8_e4m3fn
        weight_scale: torch.Tensor,  # [out, 1], fp16 or fp32
        bias: torch.Tensor | None,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert weight_fp8.dtype == torch.float8_e4m3fn, (
            f"FP8Linear expects fp8_e4m3fn weight, got {weight_fp8.dtype}"
        )
        # Register as buffers so .to(device), .half(), state_dict all work.
        # weight stays fp8 (.to() preserves dtype); scale gets promoted to fp32
        # for numerical headroom in the scaled-mm fused multiply.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32), persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.to(out_dtype), persistent=False)
        else:
            self.bias = None
        self.in_features = weight_fp8.shape[1]
        self.out_features = weight_fp8.shape[0]
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        in_shape = x.shape
        x_fp8, x_scale = _per_token_quant_fp8(x)   # [M, K], [M, 1]
        # `torch._scaled_mm` requires `b` to be column-major in its
        # transposed orientation: stride(0)==1 in [K, N]. nn.Linear's
        # weight is row-major in [N, K] (stride [K, 1]); the `.t()` view
        # then has stride [1, K] — exactly column-major in [K, N]. Do
        # NOT call `.contiguous()` here, that forces row-major and the
        # FP8 GEMM kernel refuses it ("b.stride(0) == 1" assertion).
        w_t = self.weight.t()                      # [K, N] fp8, col-major view
        s_b = self.weight_scale.reshape(1, -1)     # [1, N] fp32
        y = torch._scaled_mm(
            x_fp8,
            w_t,
            scale_a=x_scale,
            scale_b=s_b,
            out_dtype=self.out_dtype,
        )                                          # [M, N] in out_dtype
        if self.bias is not None:
            y = y + self.bias
        return y.view(*in_shape[:-1], self.out_features).to(in_dtype)


def replace_compressed_linears_with_fp8(model: nn.Module) -> int:
    """For every `nn.Linear` whose weight is FP8 (left over from a
    compressed_tensors load, before the decompress hook fires), replace it
    with an `FP8Linear`.

    Returns the number of Linears swapped.
    """
    to_swap: list[tuple[str, nn.Linear]] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.dtype == torch.float8_e4m3fn:
            to_swap.append((name, m))
    for name, lin in to_swap:
        # weight_scale is attached by compressed_tensors; if absent we can't
        # build the FP8Linear without re-deriving the scale, which would lose
        # the checkpoint's stored values. Refuse rather than guess.
        if not hasattr(lin, "weight_scale"):
            raise RuntimeError(
                f"Linear {name!r} has FP8 weight but no `weight_scale` "
                f"buffer/attr — can't construct FP8Linear without it. The "
                f"compressed_tensors load may have already dequantized or the "
                f"checkpoint is missing the scale tensor."
            )
        fp8_lin = FP8Linear(
            weight_fp8=lin.weight.data,
            weight_scale=lin.weight_scale.data,
            bias=lin.bias.data if lin.bias is not None else None,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, fp8_lin)
    return len(to_swap)


def disable_compressed_tensors_decompress(model: nn.Module) -> None:
    """Remove the lazy decompress forward-hook(s) that compressed_tensors
    installs at load time. After this returns, the model's FP8 weights will
    NOT be silently restored to bf16 on first forward.

    The exact hook locations have moved across compressed_tensors versions;
    we try the public API first and fall back to brute-force removal of any
    forward_pre_hook on the model itself.
    """
    try:
        from compressed_tensors.compressors import ModelCompressor
        # If the model carries a compressor reference, ask it to detach.
        comp = getattr(model, "_model_compressor", None) or getattr(model, "model_compressor", None)
        if comp is not None and hasattr(comp, "remove_decompression_hook"):
            comp.remove_decompression_hook(model)
            return
    except Exception:
        pass

    # Fallback: walk forward_pre_hooks and remove ones that look like the
    # compressed_tensors decompress hook by qualified name.
    removed = 0
    for m in [model, *model.modules()]:
        if not hasattr(m, "_forward_pre_hooks"):
            continue
        for hid, fn in list(m._forward_pre_hooks.items()):
            qname = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}"
            if "compressed_tensors" in qname or "decompress" in qname.lower():
                del m._forward_pre_hooks[hid]
                removed += 1
    if removed == 0:
        # Not fatal — maybe the install version doesn't use a hook here.
        # But warn loud so the caller can investigate if the FP8 check
        # later finds the weights got dequantized anyway.
        print("[fp8_hw][warn] could not find a compressed_tensors decompress hook to remove; "
              "if the FP8 check still fails downstream, weights may be getting dequantized "
              "by a different mechanism.")
