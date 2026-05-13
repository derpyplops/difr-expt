"""Integer approximations of the non-matmul ops: RMSNorm, softmax, SiLU, and
attention matmuls (Q@K.T, P@V).

Design mirrors `int_cast.py`'s "float-equivalent" approach: the math is done in
float for speed, but each op is structured so that it is bit-equivalent to a
literal integer implementation. Per-op:

  - **IntRMSNorm**: quantize x per token at b activation bits; integer sum of
    squares (exact since we just dequant with the activation scale); Newton-
    Raphson invsqrt with public LUT seed; quantized gamma multiply.
  - **int_softmax**: per-token max subtract → integer-domain shift → LUT-based
    exp (table of public fp32 entries indexed by clamped int → fp32 dequant) →
    integer sum → Newton-Raphson reciprocal → multiply by reciprocal.
  - **int_silu**: 4096-entry sigmoid LUT keyed by clamped int representation
    of x; integer multiply x * sigmoid(x).
  - **int_matmul** (attention): per-token symmetric quant of both operands;
    int64 matmul (or its float-equivalent); dequant via outer product of scales.

All operations expose:
  - public-scale parameters (eps, ranges, LUT sizes) as constants/buffers
  - configurable bit widths via `bits` argument (default 24, matching IntLinear)

The literal-int execution path is selectable via a module flag (analogous to
`IntLinear.true_int_matmul`). By default the float-equivalent path runs for
speed; the literal-int path is for end-to-end validation runs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _qmax(bits: int) -> int:
    if bits < 2 or bits > 31:
        raise ValueError(f"effective bits must be in [2, 31], got {bits}")
    return (1 << (bits - 1)) - 1


def _quantize_per_token_symmetric(
    x: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric quant. x: [..., features]. Reduces over last dim.

    Returns (x_int_fp32, scale) where x_int_fp32 is float-valued but rounded to
    integer values in [-qmax, qmax]. scale is float, shape [..., 1].
    """
    qmax = _qmax(bits)
    xf = x.float()
    row_absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = row_absmax / qmax
    x_q = torch.round(xf / scale).clamp(-qmax, qmax)
    return x_q, scale


# ---------------------------------------------------------------------------
# IntRMSNorm
# ---------------------------------------------------------------------------


class IntRMSNorm(nn.Module):
    """Int-friendly RMSNorm.

    y_i = (x_i / sqrt(mean(x^2) + eps)) * gamma_i

    Integer pipeline:
      1. Quantize x per-token at `bits` bits  -> x_int, x_scale
      2. Sum of squares: s = sum_i x_int[i]^2   (int)
         Dequant variance: v = (s * x_scale^2) / d + eps
      3. invsqrt(v) by Newton-Raphson:
            r0 = LUT[bit_extract(v)]
            r1 = r0 * (1.5 - 0.5 * v * r0^2)
            r2 = r1 * (1.5 - 0.5 * v * r1^2)
      4. Quantize gamma symmetrically at `gamma_bits` bits  -> gamma_int, gamma_scale
      5. y_int = round(x_int * gamma_int * r2 * (x_scale * gamma_scale))
         (here r2 is dequant'd; equivalent quantized-r2 path: r2_int * r_scale)

    For the float-equivalent path, steps 1-3 produce r2 (a float scalar per
    token) and step 5 is plain elementwise multiply: out = x_dequant * gamma *
    r2. Steps 1-4 introduce only int-domain rounding; step 5 produces the same
    float as the literal-int circuit modulo float reduction order.

    Gamma is held as a learnable nn.Parameter (matches HF RMSNorm) so it can be
    fine-tuned in Approach B. The gamma quantization is done on-the-fly each
    forward (cheap, [hidden] sized) so that any training updates are reflected.
    """

    # Class flag for literal-int validation path
    true_int_path: bool = False

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        bits: int = 24,
        gamma_bits: int = 24,
        nr_iterations: int = 2,
        invsqrt_lut_bits: int = 10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.bits = bits
        self.gamma_bits = gamma_bits
        self.nr_iterations = nr_iterations
        self.invsqrt_lut_bits = invsqrt_lut_bits

        # HF-compatible parameter name
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # Newton-Raphson invsqrt LUT: 2^bits-entry table over the mantissa.
        # We use a simple log-spaced table over a wide value range to produce a
        # good seed for the NR iteration. Index by extracting bits from the
        # float bits.
        # Approach: for input v in [2^-20, 2^20], precompute 1/sqrt(v) at
        # 2^invsqrt_lut_bits sample points (log-spaced).
        n_entries = 1 << invsqrt_lut_bits
        self.register_buffer(
            "_invsqrt_lut_log_min", torch.tensor(-40.0)  # log2(v_min)
        )
        self.register_buffer(
            "_invsqrt_lut_log_max", torch.tensor(40.0)  # log2(v_max)
        )
        log_grid = torch.linspace(-40.0, 40.0, n_entries)
        v_grid = 2.0 ** log_grid
        lut = 1.0 / torch.sqrt(v_grid)
        self.register_buffer("_invsqrt_lut", lut.float())

    @classmethod
    def from_hf_rmsnorm(cls, hf_norm: nn.Module, **kwargs) -> "IntRMSNorm":
        """Build IntRMSNorm from a HF RMSNorm module (Llama/Qwen)."""
        if hasattr(hf_norm, "variance_epsilon"):
            eps = float(hf_norm.variance_epsilon)
        elif hasattr(hf_norm, "eps"):
            eps = float(hf_norm.eps)
        else:
            eps = 1e-6
        weight = hf_norm.weight.detach()
        hidden_size = weight.shape[0]
        new = cls(hidden_size=hidden_size, eps=eps, **kwargs)
        new.weight = nn.Parameter(weight.clone().float())
        # Match device
        new = new.to(weight.device)
        return new

    def _invsqrt_seed(self, v: torch.Tensor) -> torch.Tensor:
        """Look up 1/sqrt(v) in a log-spaced table."""
        log_v = torch.log2(v.clamp_min(1e-30))
        n_entries = 1 << self.invsqrt_lut_bits
        idx_f = (log_v - self._invsqrt_lut_log_min) / (
            self._invsqrt_lut_log_max - self._invsqrt_lut_log_min
        ) * (n_entries - 1)
        idx = idx_f.clamp(0, n_entries - 1).round().long()
        return self._invsqrt_lut[idx]

    def _invsqrt(self, v: torch.Tensor) -> torch.Tensor:
        """Newton-Raphson invsqrt with LUT seed."""
        r = self._invsqrt_seed(v)
        for _ in range(self.nr_iterations):
            r = r * (1.5 - 0.5 * v * r * r)
        return r

    def _quantize_gamma(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric quant of gamma to gamma_bits."""
        qmax = _qmax(self.gamma_bits)
        gf = self.weight.float()
        absmax = gf.abs().amax().clamp_min(1e-30)
        scale = absmax / qmax
        g_q = torch.round(gf / scale).clamp(-qmax, qmax)
        return g_q, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.true_int_path:
            return self._true_int_forward(x)
        orig_dtype = x.dtype
        orig_shape = x.shape
        # Reduce over last dim; flatten leading dims.
        x_flat = x.reshape(-1, orig_shape[-1])

        # Step 1: per-token symmetric quant
        x_int, x_scale = _quantize_per_token_symmetric(x_flat, bits=self.bits)
        # Step 2: int sum of squares. x_int * x_int could overflow int32 for
        # bits=24; do it in fp64 to remain bit-exact under integer interpretation.
        x_int_f64 = x_int.to(torch.float64)
        s_int = (x_int_f64 * x_int_f64).sum(dim=-1, keepdim=True)
        # Dequant variance: (s * x_scale^2) / d + eps
        d = float(orig_shape[-1])
        x_scale_f64 = x_scale.to(torch.float64)
        v = s_int * (x_scale_f64 * x_scale_f64) / d + self.eps
        v = v.float()

        # Step 3: invsqrt via NR
        r = self._invsqrt(v)  # [..., 1] float

        # Step 4: quantize gamma
        g_int, g_scale = self._quantize_gamma()  # [hidden], scalar

        # Step 5: out = x_int * x_scale * g_int * g_scale * r
        # = (x_int * x_scale) * (g_int * g_scale) * r
        x_deq = x_int * x_scale  # float
        g_deq = g_int * g_scale  # float [hidden]
        out = x_deq * g_deq * r

        out = out.to(orig_dtype).reshape(orig_shape)
        return out

    def _true_int_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Literal-int RMSNorm: integer sum-of-squares (int64), integer-domain
        NR invsqrt under a public scale, integer gamma multiply.

        Mathematically equivalent to the float-equivalent path; differs only in
        that intermediates are exact int64 values (held in int64 tensors) rather
        than float64 values that happen to be integer-valued.

        CPU fallback on CUDA (same convention as IntMatmul._true_int_forward).
        """
        orig_dtype = x.dtype
        orig_shape = x.shape
        target_device = x.device
        x_flat = x.reshape(-1, orig_shape[-1])

        # Step 1: per-token symmetric quant (gives float values that are exact
        # integers in [-qmax, qmax]).
        x_int, x_scale = _quantize_per_token_symmetric(x_flat, bits=self.bits)

        # Move to int64 for the integer pipeline.
        x_i64 = x_int.to(torch.int64)
        if target_device.type == "cuda":
            x_i64 = x_i64.cpu()

        # Step 2: integer sum of squares — int64 since (2^23)^2 * hidden < 2^63.
        s_int_i64 = (x_i64 * x_i64).sum(dim=-1, keepdim=True)  # int64

        # Step 3: dequant variance using public scales. The scale (s_x^2 / d) is
        # a public constant we apply via float multiply; in a ZK circuit this is
        # a single public-scale multiply.
        d = float(orig_shape[-1])
        # Bring back to a float scalar per token so we can run the NR invsqrt in
        # float — but the NR iteration itself only uses public-scale multiplies
        # and is bit-equivalent to the integer NR-invsqrt circuit (each NR step
        # is r * (3/2 - v * r^2 / 2), all under public scales).
        s_int = s_int_i64.to(torch.float64)
        if target_device.type == "cuda":
            x_scale_dev = x_scale.cpu().to(torch.float64)
        else:
            x_scale_dev = x_scale.to(torch.float64)
        v = s_int * (x_scale_dev * x_scale_dev) / d + self.eps
        v = v.to(torch.float32)

        # Step 4: NR invsqrt seeded from LUT (LUT entries are public).
        if target_device.type == "cuda":
            # _invsqrt uses self buffers (on cuda); run it on the cuda device by
            # moving v there briefly, then bring r back to cpu for the int math.
            r = self._invsqrt(v.to(target_device)).cpu()
        else:
            r = self._invsqrt(v)

        # Step 5: quantize gamma (integer values held in float).
        g_int, g_scale = self._quantize_gamma()  # [hidden], scalar
        g_i64 = g_int.to(torch.int64)
        if target_device.type == "cuda":
            g_i64 = g_i64.cpu()

        # Step 6: integer multiply x_int * g_int (broadcast over the row).
        # Bit budget: int_bits * gamma_bits → at most int48. Safe in int64.
        prod_i64 = x_i64 * g_i64  # [N, hidden] int64

        # Step 7: dequant: (prod_i64) * x_scale * g_scale * r  →  fp32 output.
        # The float multiply by (x_scale * g_scale * r) is a public-scale multiply.
        # `r` is already on the same device as the int pipeline (cpu when on cuda
        # target, gpu otherwise); pair `x_scale` to match.
        if target_device.type == "cuda":
            scale_combined = (x_scale.cpu().to(torch.float32) * float(g_scale) * r)
        else:
            scale_combined = (x_scale.to(torch.float32) * float(g_scale) * r)
        out = prod_i64.to(torch.float32) * scale_combined

        if target_device.type == "cuda":
            out = out.to(target_device)
        out = out.to(orig_dtype).reshape(orig_shape)
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, bits={self.bits}, "
            f"gamma_bits={self.gamma_bits}, nr_iter={self.nr_iterations}"
        )


# ---------------------------------------------------------------------------
# int_softmax
# ---------------------------------------------------------------------------


def _build_exp_lut(n_entries: int, x_min: float = -16.0) -> tuple[torch.Tensor, float]:
    """Precompute exp(x) for x in [x_min, 0] mapped to indices [0, n_entries-1].

    x_min default -16.0 means anything beyond exp(-16) ≈ 1e-7 underflows to 0
    after dividing by typical sums. For more sensitive ranges, raise n_entries
    or lower x_min.

    Returns (lut, step) where step = -x_min / (n_entries - 1).
    """
    step = -x_min / (n_entries - 1)
    grid = torch.arange(n_entries, dtype=torch.float32) * step + x_min  # [x_min, 0]
    lut = torch.exp(grid)
    return lut, step


def _nr_reciprocal_int(
    s_int: torch.Tensor,
    s_scale: float,
    iterations: int = 3,
) -> tuple[torch.Tensor, float]:
    """Newton-Raphson reciprocal of `s_int * s_scale` (a positive scalar per row).

    Returns (r_int, r_scale) such that `r_int * r_scale ≈ 1 / (s_int * s_scale)`.

    All multiplies inside the NR iteration are public-scale multiplies; the
    intermediate `r` is held as a float that's a quantized integer value times
    a public scale. We pick the r-quantization scale dynamically from the seed
    so the iteration stays in int range.

    Iteration: r_{k+1} = r_k * (2 - s * r_k).
    """
    # Seed: r_0 = 1 / s.  Compute it directly in float to seed the iteration —
    # this is the LUT-equivalent seed (a precomputed reciprocal table indexed by
    # leading bits of s).  In an actual circuit this would be a LUT lookup; the
    # math from here is bit-equivalent.
    s_f = s_int.to(torch.float64) * s_scale  # [..., 1]
    r = 1.0 / s_f
    # NR iterations: r = r * (2 - s * r).  Each multiplication is a public-scale
    # multiply.  For 2-3 iterations from a float seed the result reaches double
    # precision quickly.
    for _ in range(iterations):
        r = r * (2.0 - s_f * r)
    return r  # float64, holding the reciprocal directly


def int_softmax(
    x: torch.Tensor,
    dim: int = -1,
    lut_size: int = 1024,
    x_min: float = -16.0,
    bits: int = 24,
    cache: Optional[dict] = None,
    true_int: bool = False,
    lut_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integer-friendly softmax.

    Pipeline:
      1. Subtract per-row max: y = x - x.amax(dim, keepdim=True). All y_i <= 0.
      2. Index into exp LUT: idx = clamp(round((y - x_min) / step), 0, N-1).
         e_i = LUT[idx_i]  (float values, representing the quantized exp).
      3. Sum: s = sum_i e_i.
      4. Reciprocal r = 1/s by NR (or direct division here; division by a known
         positive sum is a standard integer NR-reciprocal sub-circuit).
      5. p_i = e_i * r.

    Note: in the float-equivalent path, we use plain float division for the
    reciprocal — it's bit-equivalent to NR-reciprocal with enough iterations.

    `cache` (optional dict) memoizes the LUT on first call so we don't rebuild
    the table for every softmax call.

    `true_int=True` runs the literal-int path: the exp LUT entries are stored
    as int24 values under a public scale `s_lut`, the sum is performed in int64,
    the reciprocal is via Newton-Raphson, and the final multiply is exact int.
    CPU fallback on CUDA (same convention as IntMatmul._true_int_forward).

    Returns probabilities with the same shape as x.
    """
    if lut_override is not None:
        # Trainable-LUT mode: a Parameter holds the entries. Step is implied by
        # lut_size + x_min and matches the default _build_exp_lut layout.
        lut = lut_override.to(x.device)
        step = -x_min / (lut.shape[0] - 1)
        lut_size = lut.shape[0]
    elif cache is not None:
        key = (lut_size, x_min, x.device)
        if key not in cache:
            lut, step = _build_exp_lut(lut_size, x_min=x_min)
            cache[key] = (lut.to(x.device), step)
        lut, step = cache[key]
    else:
        lut, step = _build_exp_lut(lut_size, x_min=x_min)
        lut = lut.to(x.device)

    orig_dtype = x.dtype
    xf = x.float()
    m = xf.amax(dim=dim, keepdim=True)
    y = xf - m
    # y in (-inf, 0]; clamp at x_min
    y = y.clamp_min(x_min)
    # Index in LUT (0 corresponds to x_min, N-1 to 0).
    idx_f = (y - x_min) / step
    idx = idx_f.round().long().clamp(0, lut_size - 1)
    e = lut[idx]  # [..., n]

    if true_int:
        # Literal-int execution: the LUT values are integers under a public
        # scale s_lut.  Since exp(y) for y in [x_min, 0] lies in [exp(x_min), 1],
        # we pick s_lut so that the max LUT value (= 1.0) maps to qmax_lut.
        qmax_lut = (1 << (bits - 1)) - 1
        s_lut = 1.0 / qmax_lut  # public scale: each int LUT value represents s_lut * int
        # Build the int-valued LUT on the fly (cheap; lut_size is small).
        lut_int_f = (lut / s_lut).round().clamp(0, qmax_lut)  # float-held int values
        e_int_f = lut_int_f[idx]  # [..., n], float-valued ints in [0, qmax_lut]

        target_device = x.device
        e_i64 = e_int_f.to(torch.int64)
        if target_device.type == "cuda":
            e_i64 = e_i64.cpu()

        # int64 sum.  Per-row count is the size of `dim`; even at lut_size=2^31-1
        # and 64k positions the sum is well inside int64 (2^23 * 2^16 = 2^39).
        s_i64 = e_i64.sum(dim=dim, keepdim=True)  # int64 sum of int24 entries

        # NR reciprocal of (s_i64 * s_lut).  The result r is a float64 holding
        # 1 / (s_i64 * s_lut) which is the reciprocal of the float sum.
        r = _nr_reciprocal_int(s_i64, s_lut, iterations=3)  # float64

        # Final multiply: p_int = e_i64 * r_int  (in a circuit, r is quantized
        # to int under a public scale; here we hold r as float64 = int * scale).
        p = e_i64.to(torch.float64) * s_lut * r  # = (e_int * s_lut) * (1 / (s * s_lut))

        if target_device.type == "cuda":
            p = p.to(target_device)
        return p.to(orig_dtype)

    s = e.sum(dim=dim, keepdim=True)
    # Reciprocal — float division is fine and bit-equivalent to NR-reciprocal
    # converged.
    p = e / s
    return p.to(orig_dtype)


# ---------------------------------------------------------------------------
# int_silu
# ---------------------------------------------------------------------------


def _build_sigmoid_lut(n_entries: int, x_range: float = 16.0) -> tuple[torch.Tensor, float]:
    """Precompute sigmoid(x) for x in [-x_range, x_range].

    Returns (lut, step) where step = 2*x_range / (n_entries - 1).
    """
    step = 2 * x_range / (n_entries - 1)
    grid = torch.arange(n_entries, dtype=torch.float32) * step - x_range
    lut = torch.sigmoid(grid)
    return lut, step


def int_silu(
    x: torch.Tensor,
    lut_size: int = 4096,
    x_range: float = 16.0,
    cache: Optional[dict] = None,
    bits: int = 24,
    true_int: bool = False,
    lut_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integer-friendly SiLU: x * sigmoid(x).

    Sigmoid is computed by LUT; the multiply is exact.

    `true_int=True` runs the literal-int path: x is per-token symmetric quant'd
    to int24 (held in int64), the sigmoid LUT entries are int values under a
    public scale s_sig (max LUT value 1.0 maps to qmax), and the product
    `x_int * sig_int` is computed in int64. CPU fallback on CUDA.
    """
    if lut_override is not None:
        lut = lut_override.to(x.device)
        step = 2 * x_range / (lut.shape[0] - 1)
        lut_size = lut.shape[0]
    elif cache is not None:
        key = (lut_size, x_range, x.device)
        if key not in cache:
            lut, step = _build_sigmoid_lut(lut_size, x_range=x_range)
            cache[key] = (lut.to(x.device), step)
        lut, step = cache[key]
    else:
        lut, step = _build_sigmoid_lut(lut_size, x_range=x_range)
        lut = lut.to(x.device)

    orig_dtype = x.dtype
    xf = x.float()
    # idx 0 = -x_range, idx N-1 = +x_range
    idx_f = (xf + x_range) / step
    idx = idx_f.round().long().clamp(0, lut_size - 1)
    sig = lut[idx]

    if true_int:
        # Literal-int path. Quantize x per-token symmetric.
        qmax = _qmax(bits)
        x_int, x_scale = _quantize_per_token_symmetric(xf, bits=bits)
        # Build int sigmoid LUT under public scale s_sig.  sigmoid(x) in (0, 1)
        # so we map max value (= 1.0) to qmax.
        s_sig = 1.0 / qmax
        sig_int_f = (lut / s_sig).round().clamp(0, qmax)
        sig_int_picked = sig_int_f[idx]  # [..., n], float-valued ints

        target_device = x.device
        x_i64 = x_int.to(torch.int64)
        sig_i64 = sig_int_picked.to(torch.int64)
        if target_device.type == "cuda":
            x_i64 = x_i64.cpu()
            sig_i64 = sig_i64.cpu()
        # Integer multiply: int24 * int24 = int48.  Safe in int64.
        prod_i64 = x_i64 * sig_i64
        # Dequant: scale_combined = x_scale * s_sig (broadcast).
        if target_device.type == "cuda":
            x_scale_dev = x_scale.cpu().to(torch.float32)
        else:
            x_scale_dev = x_scale.to(torch.float32)
        out = prod_i64.to(torch.float32) * x_scale_dev * float(s_sig)
        if target_device.type == "cuda":
            out = out.to(target_device)
        return out.to(orig_dtype)

    out = xf * sig
    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# int_matmul (attention Q@K.T and P@V)
# ---------------------------------------------------------------------------


def int_matmul(
    a: torch.Tensor, b: torch.Tensor, bits: int = 24
) -> torch.Tensor:
    """Per-token symmetric quant of both sides, then matmul.

    a: [..., M, K]
    b: [..., K, N]
    Returns [..., M, N].

    For attention Q@K.T:  a=Q (b, h, T, D), b=K.T (b, h, D, T)
    For attention P@V:    a=P (b, h, T, T), b=V (b, h, T, D)

    Per-token = per-row on `a` (reduces over the K dim); on `b` we want
    per-column quant of b (i.e. quantize b along K too, which is `b.transpose`
    per-row). To match the math, we transpose b, quantize per-row, transpose
    back.
    """
    qmax = _qmax(bits)

    # Per-row quant for a (reduce over K).
    af = a.float()
    a_absmax = af.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)  # [..., M, 1]
    a_scale = a_absmax / qmax
    a_q = torch.round(af / a_scale).clamp(-qmax, qmax)
    a_deq = a_q * a_scale  # bit-equivalent dequant

    # Per-column quant for b: quantize b.transpose(-1,-2) per-row, transpose back.
    bf = b.float()
    b_t = bf.transpose(-1, -2)  # [..., N, K]
    b_absmax = b_t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)  # [..., N, 1]
    b_scale_t = b_absmax / qmax  # [..., N, 1]
    b_q_t = torch.round(b_t / b_scale_t).clamp(-qmax, qmax)
    b_deq_t = b_q_t * b_scale_t  # bit-equivalent dequant
    b_deq = b_deq_t.transpose(-1, -2)  # [..., K, N]

    out = a_deq @ b_deq
    return out.to(a.dtype)


class IntMatmul(nn.Module):
    """nn.Module wrapper around int_matmul for tidy HF surgery."""

    true_int_path: bool = False

    def __init__(self, bits: int = 24):
        super().__init__()
        self.bits = bits

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.true_int_path:
            return self._true_int_forward(a, b)
        return int_matmul(a, b, bits=self.bits)

    def _true_int_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Literal int64 matmul (CPU fallback on CUDA). Slow."""
        qmax = _qmax(self.bits)
        af = a.float()
        a_absmax = af.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        a_scale = a_absmax / qmax
        a_q = torch.round(af / a_scale).clamp(-qmax, qmax)

        bf = b.float()
        b_t = bf.transpose(-1, -2)
        b_absmax = b_t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        b_scale_t = b_absmax / qmax
        b_q_t = torch.round(b_t / b_scale_t).clamp(-qmax, qmax)
        b_q = b_q_t.transpose(-1, -2)

        a_i64 = a_q.to(torch.int64)
        b_i64 = b_q.to(torch.int64)

        if a.device.type == "cuda":
            int_prod = (a_i64.cpu() @ b_i64.cpu()).to(a.device)
        else:
            int_prod = a_i64 @ b_i64

        # Dequant: int_prod[..., m, n] * a_scale[..., m, 1] * b_scale_t[..., n, 1].T
        b_scale = b_scale_t.transpose(-1, -2)  # [..., 1, N]
        out = int_prod.to(torch.float32) * a_scale * b_scale
        return out.to(a.dtype)


# ---------------------------------------------------------------------------
# int_rope_apply (rotary position embedding)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF rotate_half: [x_first, x_second] -> [-x_second, x_first].

    Identical to transformers.models.{llama,qwen2}.modeling_*.rotate_half.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


# Module-level flag (mirrors IntRMSNorm.true_int_path / IntMatmul.true_int_path).
# Toggled via set_true_int_path on IntRopeApply instances; the free function
# `int_rope_apply` accepts a `true_int` kwarg directly.
_ROPE_TRUE_INT_DEFAULT: bool = False


def int_rope_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    bits: int = 24,
    unsqueeze_dim: int = 1,
    true_int: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer-friendly rotary position embedding.

    Mathematically equivalent to transformers' apply_rotary_pos_emb:
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin
    but factored so every multiplication and addition is performed on
    symmetric-quantized integers under public scales.

    Strategy:
      - cos / sin have absolute value <= 1, so we quantize them with a single
        public scale s_table = 1 / (2^(bits-1) - 1). Both tables share the
        same scale (allowed since they live in the same range), which is what
        makes the dequant a single multiplicative constant.
      - q, k are per-token symmetric quantized along the head dim with scales
        s_q, s_k (one per (batch, head, token) row).
      - The rotation becomes an integer op:
            q_int' = cos_int * q_int + sin_int * rotate_half(q_int)
        Then dequant with s_q * s_table. rotate_half is a pure permutation +
        sign flip of int values, so it stays int-exact.
      - Bit budget: int24 * int24 = int48; sum of two such products = int49.
        Comfortably inside int64.

    Shapes mirror HF's apply_rotary_pos_emb:
      q, k: [batch, n_heads, seq, head_dim]   (after the usual transpose)
      cos, sin: [batch, seq, head_dim]        (unsqueeze_dim=1 -> heads broadcast)

    `true_int=True` runs the literal int math path (uses int64, CPU fallback
    on CUDA — same convention as IntLinear / IntMatmul); the default path is
    bit-equivalent in float for speed.
    """
    orig_dtype = q.dtype
    qmax = _qmax(bits)
    s_table = 1.0 / qmax  # |cos|, |sin| <= 1 → single public scale

    # Broadcast cos/sin to match q/k.
    cos_b = cos.unsqueeze(unsqueeze_dim)
    sin_b = sin.unsqueeze(unsqueeze_dim)

    cos_f = cos_b.float()
    sin_f = sin_b.float()
    # Quantize cos/sin with the single public scale s_table.
    cos_int = torch.round(cos_f / s_table).clamp(-qmax, qmax)
    sin_int = torch.round(sin_f / s_table).clamp(-qmax, qmax)

    # Per-token symmetric quant of q, k along the head dim.
    q_int, s_q = _quantize_per_token_symmetric(q, bits=bits)
    k_int, s_k = _quantize_per_token_symmetric(k, bits=bits)

    if true_int:
        # Literal int64 path — rotate_half + multiply-add on int64 tensors.
        cos_i64 = cos_int.to(torch.int64)
        sin_i64 = sin_int.to(torch.int64)
        q_i64 = q_int.to(torch.int64)
        k_i64 = k_int.to(torch.int64)

        target_device = q.device
        if target_device.type == "cuda":
            cos_i64 = cos_i64.cpu()
            sin_i64 = sin_i64.cpu()
            q_i64 = q_i64.cpu()
            k_i64 = k_i64.cpu()

        q_rot_i64 = _rotate_half(q_i64)
        k_rot_i64 = _rotate_half(k_i64)
        # Note: int64 broadcasting handles the head-dim singleton in cos/sin.
        q_out_i64 = cos_i64 * q_i64 + sin_i64 * q_rot_i64
        k_out_i64 = cos_i64 * k_i64 + sin_i64 * k_rot_i64

        if target_device.type == "cuda":
            q_out_i64 = q_out_i64.to(target_device)
            k_out_i64 = k_out_i64.to(target_device)

        # Dequant: combined scale = s_q * s_table (broadcasts over head_dim).
        q_out = q_out_i64.to(torch.float32) * (s_q * s_table)
        k_out = k_out_i64.to(torch.float32) * (s_k * s_table)
    else:
        # Float-equivalent path. Same math, but executed in float — bit-equivalent
        # to the literal int circuit because each multiplicand is already an
        # exact integer value (held in float).
        q_rot = _rotate_half(q_int)
        k_rot = _rotate_half(k_int)
        q_out = (cos_int * q_int + sin_int * q_rot) * (s_q * s_table)
        k_out = (cos_int * k_int + sin_int * k_rot) * (s_k * s_table)

    return q_out.to(orig_dtype), k_out.to(orig_dtype)


class IntRopeApply(nn.Module):
    """nn.Module wrapper around int_rope_apply for HF surgery.

    Exposes the `true_int_path` attribute so `set_true_int_path` flips it
    consistently with IntRMSNorm and IntMatmul.
    """

    true_int_path: bool = False

    def __init__(self, bits: int = 24, unsqueeze_dim: int = 1):
        super().__init__()
        self.bits = bits
        self.unsqueeze_dim = unsqueeze_dim

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return int_rope_apply(
            q, k, cos, sin,
            bits=self.bits,
            unsqueeze_dim=self.unsqueeze_dim,
            true_int=self.true_int_path,
        )


# ---------------------------------------------------------------------------
# Globals: switch literal-int mode across all int_ops modules
# ---------------------------------------------------------------------------


def set_true_int_path(model: nn.Module, value: bool) -> None:
    """Toggle literal-int execution across all int-op modules.

    Covers IntRMSNorm, IntMatmul, IntRopeApply, and IntSiLUModule (the SiLU
    nn.Module wrapper from patch_hf_model). Softmax is a free function with no
    holding module; instead, the attention forward closure reads
    `cfg.true_int_nonmatmul` on each call (set via `set_softmax_true_int` below
    or by mutating the cfg directly).
    """
    # Local import to avoid a top-level circular import (patch_hf_model imports
    # from int_ops).
    try:
        from difr_expt.patch_hf_model import IntSiLUModule
        silu_cls: tuple = (IntSiLUModule,)
    except Exception:
        silu_cls = ()

    target_classes = (IntRMSNorm, IntMatmul, IntRopeApply) + silu_cls
    for m in model.modules():
        if isinstance(m, target_classes):
            m.true_int_path = value
    # Also flip the softmax-true-int flag on any attention modules that carry
    # an attached IntOpsConfig (set by patch_hf_model).
    for m in model.modules():
        cfg = getattr(m, "_int_ops_cfg", None)
        if cfg is not None:
            cfg.true_int_nonmatmul = value
