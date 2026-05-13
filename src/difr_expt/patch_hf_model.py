"""HF model surgery: replace RMSNorm, attention softmax + Q@K.T + P@V, and
MLP SiLU with their int approximations.

We monkey-patch at the *module-instance* level rather than swapping classes,
which keeps the patches local and reversible. For each transformer block:

  - RMSNorm modules (input_layernorm, post_attention_layernorm, and any
    q_norm/k_norm on Qwen3) are swapped to `IntRMSNorm`.
  - The attention module's forward is replaced with a function that uses
    `int_softmax`, `int_matmul` for Q@K.T and P@V.
  - The MLP's `act_fn` (a SiLU module) is replaced with a thin wrapper that
    calls `int_silu`.

The model's final norm (`model.norm`) is also swapped.

`int_op_args` controls per-op precision/LUT-size parameters; defaults are
sensible for Phase 1 baseline measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from difr_expt.int_ops import (
    IntMatmul,
    IntRMSNorm,
    IntRopeApply,
    int_rope_apply,
    int_silu,
    int_softmax,
)


@dataclass
class IntOpsConfig:
    """Per-op hyperparameters for the int approximations."""

    rmsnorm_bits: int = 24
    rmsnorm_gamma_bits: int = 24
    rmsnorm_nr_iter: int = 2
    rmsnorm_lut_bits: int = 10

    softmax_lut_size: int = 1024
    softmax_x_min: float = -16.0
    softmax_bits: int = 24

    silu_lut_size: int = 4096
    silu_x_range: float = 16.0

    attn_matmul_bits: int = 24

    rope_bits: int = 24

    # Which ops to int-ify (for Phase 3 ablation)
    replace_rmsnorm: bool = True
    replace_softmax: bool = True
    replace_silu: bool = True
    replace_attn_matmul: bool = True
    replace_rope: bool = True

    # Literal-int execution toggle for the free-function ops (softmax). This is
    # the closure-captured flag the attention forward checks each call. Module
    # ops (RMSNorm, Matmul, RoPE, SiLU) read their own `true_int_path` attr.
    # `set_true_int_path` mutates this field on attached cfgs.
    true_int_nonmatmul: bool = False

    # Caches for LUTs (populated by the patcher on first call)
    softmax_cache: dict = field(default_factory=dict)
    silu_cache: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module wrappers (so we can monkey-patch identically across archs)
# ---------------------------------------------------------------------------


class IntSiLUModule(nn.Module):
    """nn.Module wrapper around int_silu for HF compatibility.

    Carries a `true_int_path` flag flipped by `set_true_int_path`; when True
    forwards through `int_silu(..., true_int=True)` which runs the literal-int
    SiLU circuit. If `make_lut_trainable()` is called, the sigmoid LUT becomes
    an `nn.Parameter` (so it can be trained against an fp32 reference).
    """

    true_int_path: bool = False

    def __init__(self, cfg: IntOpsConfig):
        super().__init__()
        self.lut_size = cfg.silu_lut_size
        self.x_range = cfg.silu_x_range
        self._cache = cfg.silu_cache
        self.lut: Optional[nn.Parameter] = None

    def make_lut_trainable(self):
        from difr_expt.int_ops import _build_sigmoid_lut
        lut, _ = _build_sigmoid_lut(self.lut_size, x_range=self.x_range)
        self.lut = nn.Parameter(lut.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int_silu(
            x,
            lut_size=self.lut_size,
            x_range=self.x_range,
            cache=self._cache if self.lut is None else None,
            true_int=self.true_int_path,
            lut_override=self.lut,
        )


class IntSoftmaxModule(nn.Module):
    """nn.Module wrapper around int_softmax. Holds the softmax-exp LUT, which
    can be promoted to an `nn.Parameter` via `make_lut_trainable()` for training
    against an fp32 reference.

    Used to replace the bare `int_softmax(...)` call inside the attention
    forward closure so the LUT is a discoverable parameter of the model.
    """

    true_int_path: bool = False

    def __init__(self, cfg: IntOpsConfig):
        super().__init__()
        self.lut_size = cfg.softmax_lut_size
        self.x_min = cfg.softmax_x_min
        self._cache = cfg.softmax_cache
        self.lut: Optional[nn.Parameter] = None

    def make_lut_trainable(self):
        from difr_expt.int_ops import _build_exp_lut
        lut, _ = _build_exp_lut(self.lut_size, x_min=self.x_min)
        self.lut = nn.Parameter(lut.clone())

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return int_softmax(
            x,
            dim=dim,
            lut_size=self.lut_size,
            x_min=self.x_min,
            cache=self._cache if self.lut is None else None,
            true_int=self.true_int_path,
            lut_override=self.lut,
        )


# ---------------------------------------------------------------------------
# Attention forward replacement
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """HF's repeat_kv re-implementation (avoids importing from model-specific files)."""
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def make_int_eager_attention_forward(
    cfg: IntOpsConfig,
    q_k_matmul: IntMatmul,
    p_v_matmul: IntMatmul,
    softmax_module: Optional["IntSoftmaxModule"] = None,
):
    """Build an eager attention forward closure that uses int ops for the
    softmax and (optionally) the two attention matmuls.

    If `softmax_module` is provided, its forward is used (lets the LUT be a
    trainable Parameter held on the module).

    The closure captures `cfg` so its `true_int_nonmatmul` flag can be flipped
    at any time (e.g. by `set_true_int_path`) to switch softmax to literal-int.

    Returns a function with the same signature as HF's `eager_attention_forward`.
    """
    use_int_softmax = cfg.replace_softmax
    use_int_matmul = cfg.replace_attn_matmul
    softmax_cache = cfg.softmax_cache
    softmax_lut_size = cfg.softmax_lut_size
    softmax_x_min = cfg.softmax_x_min

    def fwd(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        key_states = _repeat_kv(key, module.num_key_value_groups)
        value_states = _repeat_kv(value, module.num_key_value_groups)

        # Q @ K.T
        if use_int_matmul:
            attn_weights = q_k_matmul(query, key_states.transpose(2, 3)) * scaling
        else:
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # softmax
        if use_int_softmax:
            if softmax_module is not None:
                # Allow runtime override of softmax true_int from cfg
                softmax_module.true_int_path = cfg.true_int_nonmatmul
                attn_weights = softmax_module(attn_weights, dim=-1).to(query.dtype)
            else:
                attn_weights = int_softmax(
                    attn_weights, dim=-1,
                    lut_size=softmax_lut_size, x_min=softmax_x_min,
                    cache=softmax_cache,
                    true_int=cfg.true_int_nonmatmul,
                ).to(query.dtype)
        else:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query.dtype)

        # dropout (we run in eval; this is a no-op)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        # P @ V
        if use_int_matmul:
            attn_output = p_v_matmul(attn_weights, value_states)
        else:
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return fwd


def _wrap_attention_forward(attn_module: nn.Module, cfg: IntOpsConfig):
    """Replace `attn_module.forward` with one that uses our int attention.

    Preserves the rest of the call structure (q_proj, k_proj, q_norm/k_norm,
    rotary embedding, KV cache update) and only replaces the inner attention
    interface call (which is normally `eager_attention_forward` / sdpa /
    flash). We force the eager path.
    """
    # Force eager attention dispatch by overriding config attribute on the module
    # We'll set the model config to eager elsewhere, but also attach an attribute
    # so any flash/sdpa lookup is bypassed.
    # The simplest approach: directly replace the .forward method.

    q_k_matmul = IntMatmul(bits=cfg.attn_matmul_bits)
    p_v_matmul = IntMatmul(bits=cfg.attn_matmul_bits)
    softmax_module = IntSoftmaxModule(cfg)
    # Attach so they're part of the module graph (for .to(device), state-dict)
    attn_module._int_qk_matmul = q_k_matmul
    attn_module._int_pv_matmul = p_v_matmul
    attn_module._int_softmax = softmax_module
    # Attach the cfg so `set_true_int_path` (which walks modules) can flip the
    # softmax true-int flag.
    attn_module._int_ops_cfg = cfg

    int_eager_fwd = make_int_eager_attention_forward(cfg, q_k_matmul, p_v_matmul, softmax_module)

    # Save original forward for potential rollback
    if not hasattr(attn_module, "_orig_forward"):
        attn_module._orig_forward = attn_module.forward

    # Inspect the original attention to find q_norm/k_norm if present (Qwen3)
    has_q_norm = hasattr(attn_module, "q_norm") and not isinstance(attn_module.q_norm, nn.Identity)
    has_k_norm = hasattr(attn_module, "k_norm") and not isinstance(attn_module.k_norm, nn.Identity)

    # We need access to apply_rotary_pos_emb. Import lazily / based on the
    # module's class' source module.
    from importlib import import_module
    mod_name = attn_module.__class__.__module__
    src = import_module(mod_name)
    apply_rotary_pos_emb = getattr(src, "apply_rotary_pos_emb")

    # Optional int RoPE. Both Llama and Qwen2 expose the same cos/sin tables to
    # apply_rotary_pos_emb, so a single int implementation handles both bases
    # (10000 for Qwen2, 500000 for Llama-3.1) — the difference lives in the
    # precomputed tables we receive.
    if cfg.replace_rope:
        rope_module = IntRopeApply(bits=cfg.rope_bits)
        attn_module._int_rope = rope_module
    else:
        rope_module = None

    def new_forward(
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        if has_q_norm:
            query_states = attn_module.q_norm(
                attn_module.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
        else:
            query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if has_k_norm:
            key_states = attn_module.k_norm(
                attn_module.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
        else:
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn_module.layer_idx, cache_kwargs
            )

        # Use our int eager attention
        sliding_window = getattr(attn_module, "sliding_window", None)
        attn_output, attn_weights = int_eager_fwd(
            attn_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=attn_module.scaling,
            dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
            sliding_window=sliding_window,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)
        return attn_output, attn_weights

    # Bind as bound method (it won't be — we'll attach as a callable on the module)
    attn_module.forward = new_forward


# ---------------------------------------------------------------------------
# Top-level patcher
# ---------------------------------------------------------------------------


# Class names we treat as RMSNorm across the three architectures.
RMS_NORM_CLASS_NAMES = {
    "LlamaRMSNorm",
    "Qwen2RMSNorm",
    "Qwen3RMSNorm",
    "MistralRMSNorm",
}

ATTENTION_CLASS_NAMES = {
    "LlamaAttention",
    "Qwen2Attention",
    "Qwen3Attention",
    "MistralAttention",
}

MLP_CLASS_NAMES = {
    "LlamaMLP",
    "Qwen2MLP",
    "Qwen3MLP",
    "MistralMLP",
}


def _is_rmsnorm(module: nn.Module) -> bool:
    return module.__class__.__name__ in RMS_NORM_CLASS_NAMES


def _is_attention(module: nn.Module) -> bool:
    return module.__class__.__name__ in ATTENTION_CLASS_NAMES


def _is_mlp(module: nn.Module) -> bool:
    return module.__class__.__name__ in MLP_CLASS_NAMES


def patch_model_int_nonmatmul(
    model: nn.Module,
    cfg: Optional[IntOpsConfig] = None,
) -> dict[str, int]:
    """Replace non-matmul ops with int approximations in-place.

    Returns a dict of replacement counts by op type for sanity checks.
    """
    if cfg is None:
        cfg = IntOpsConfig()

    counts = {"rmsnorm": 0, "silu": 0, "attention": 0}

    # 1) Force eager attention ONLY if we're actually wrapping the attention
    # forward (softmax / attn_matmul replacement). Otherwise leave the default
    # (sdpa) — on some models (e.g. Qwen2.5-7B), the eager-vs-sdpa numerical
    # drift compounds to ~10% PPL even when no other op is replaced.
    if cfg.replace_softmax or cfg.replace_attn_matmul:
        if hasattr(model, "config"):
            model.config._attn_implementation = "eager"
            if hasattr(model.config, "text_config"):
                model.config.text_config._attn_implementation = "eager"

    # 2) Walk and patch
    if cfg.replace_rmsnorm:
        # Collect RMSNorm modules to replace (can't mutate during iteration)
        to_replace_norms: list[tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if _is_rmsnorm(module):
                to_replace_norms.append((name, module))
        for name, module in to_replace_norms:
            new_norm = IntRMSNorm.from_hf_rmsnorm(
                module,
                bits=cfg.rmsnorm_bits,
                gamma_bits=cfg.rmsnorm_gamma_bits,
                nr_iterations=cfg.rmsnorm_nr_iter,
                invsqrt_lut_bits=cfg.rmsnorm_lut_bits,
            )
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_norm)
            counts["rmsnorm"] += 1

    # 3) MLP SiLU
    if cfg.replace_silu:
        for name, module in model.named_modules():
            if _is_mlp(module):
                if hasattr(module, "act_fn"):
                    module.act_fn = IntSiLUModule(cfg)
                    counts["silu"] += 1

    # 4) Attention surgery
    if cfg.replace_softmax or cfg.replace_attn_matmul:
        for name, module in model.named_modules():
            if _is_attention(module):
                _wrap_attention_forward(module, cfg)
                counts["attention"] += 1

    return counts
