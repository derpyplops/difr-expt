"""Float-side attention wrapper for the per-layer L2 harness.

Problem this solves
-------------------
HF's attention runs Q@K.T, softmax, and P@V as inline tensor ops, not
as `nn.Module`s — so the harness's by-name hook matching can't see them
on the float (bf16) side. The FP8 student loaded via compressed_tensors
also runs those three ops inline (only `nn.Linear` gets quantized in
FP8-dynamic). To get a matching submodule on *both* sides so the
harness can diff them, we replace each attention's forward with a
float-eager wrapper that routes torch.matmul / F.softmax through
`FloatMatmul` / `FloatSoftmax` `nn.Module` wrappers — exposing
`_qk_matmul`, `_pv_matmul`, `_softmax` as hookable submodules.

Numerics are bit-identical to HF's `eager_attention_forward` in
transformers 4.57.3 (we verified the source). The only change is
routing the same calls through `nn.Module`s.
"""

from __future__ import annotations

from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F


# Class names we treat as transformer attention modules.
ATTENTION_CLASS_NAMES = {
    "LlamaAttention",
    "Qwen2Attention",
    "Qwen3Attention",
    "MistralAttention",
}


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """HF's repeat_kv re-implementation (kept local so this module doesn't
    depend on the fake-quant int patcher)."""
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Float twin modules
# ---------------------------------------------------------------------------


class FloatMatmul(nn.Module):
    """`torch.matmul` wrapped as a module so harness hooks can capture it.

    Mirrors the int side's `IntMatmul` interface (two-tensor forward) so
    the harness can diff them directly.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class FloatSoftmax(nn.Module):
    """`F.softmax(dtype=fp32)` wrapped as a module.

    HF's `eager_attention_forward` does the softmax in fp32 and casts back
    to query.dtype outside the softmax call:

        attn_weights = F.softmax(x, dim=-1, dtype=torch.float32).to(query.dtype)

    Our `IntSoftmaxModule` also returns fp32; the `.to(query.dtype)` cast
    happens at the call site in `int_eager_fwd`. We mirror that exactly:
    `FloatSoftmax.forward` returns fp32, and the caller is responsible
    for the cast. This keeps the diff between float and int softmaxes
    measured in fp32 (no double-cast rounding inside the diff).
    """

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return F.softmax(x, dim=dim, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Float-eager attention closure (bit-identical to HF eager_attention_forward)
# ---------------------------------------------------------------------------


def make_float_eager_attention_forward(qk_matmul, pv_matmul, softmax_module):
    """Eager attention forward using the three Float twin modules in place of
    inline `torch.matmul` / `F.softmax` calls. Identical math to HF's
    `transformers.models.qwen2.modeling_qwen2.eager_attention_forward`.
    """

    def fwd(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = _repeat_kv(key, module.num_key_value_groups)
        value_states = _repeat_kv(value, module.num_key_value_groups)

        attn_weights = qk_matmul(query, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = softmax_module(attn_weights, dim=-1).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = pv_matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return fwd


def wrap_attention_forward_float(attn_module: nn.Module) -> None:
    """Replace `attn_module.forward` with a float-eager wrapper that exposes
    `_qk_matmul`, `_pv_matmul`, `_softmax` as named submodules.

    Numerics: bit-identical to HF's `eager_attention_forward` (we matched
    the source for transformers 4.57.3 — the diff is just routing the
    same `torch.matmul` / `F.softmax` calls through nn.Module wrappers).

    Q/K/V/O projection, q_norm/k_norm, RoPE, KV cache update are all kept
    as in the un-wrapped float forward.
    """
    qk_matmul = FloatMatmul()
    pv_matmul = FloatMatmul()
    softmax_module = FloatSoftmax()
    attn_module._qk_matmul = qk_matmul
    attn_module._pv_matmul = pv_matmul
    attn_module._softmax = softmax_module

    if not hasattr(attn_module, "_orig_forward"):
        attn_module._orig_forward = attn_module.forward

    has_q_norm = hasattr(attn_module, "q_norm") and not isinstance(attn_module.q_norm, nn.Identity)
    has_k_norm = hasattr(attn_module, "k_norm") and not isinstance(attn_module.k_norm, nn.Identity)

    mod_name = attn_module.__class__.__module__
    src = import_module(mod_name)
    apply_rotary_pos_emb = getattr(src, "apply_rotary_pos_emb")

    float_eager_fwd = make_float_eager_attention_forward(qk_matmul, pv_matmul, softmax_module)

    def new_forward(
        hidden_states,
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

        sliding_window = getattr(attn_module, "sliding_window", None)
        attn_output, attn_weights = float_eager_fwd(
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

    attn_module.forward = new_forward


# ---------------------------------------------------------------------------
# Top-level glue
# ---------------------------------------------------------------------------


def wrap_all_attention_forwards_float(model: nn.Module) -> int:
    """Apply `wrap_attention_forward_float` to every attention module in
    the model. Returns the number wrapped."""
    n = 0
    for _name, m in model.named_modules():
        if m.__class__.__name__ in ATTENTION_CLASS_NAMES:
            wrap_attention_forward_float(m)
            n += 1
    return n


def force_eager_attn(model: nn.Module) -> None:
    """Force eager attention dispatch on the model config. Our wrappers
    interpret the eager additive-mask format; under sdpa, HF would pass a
    different mask shape and our wrappers would mis-handle it.

    Has no runtime effect once `attn.forward` is fully replaced, but
    matters for downstream code that introspects the config (e.g. HF
    generation utilities)."""
    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = "eager"


