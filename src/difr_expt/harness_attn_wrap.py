"""Float-side attention wrapper + name normalization for the per-layer L2 harness.

Problem this solves
-------------------
The int patcher (`patch_model_int_nonmatmul`) hides three integerized ops
inside the attention forward closure:

  - Q @ K.T   →  IntMatmul attached as `attn._int_qk_matmul`
  - softmax  →  IntSoftmaxModule attached as `attn._int_softmax`
  - P @ V   →  IntMatmul attached as `attn._int_pv_matmul`

These are nn.Modules and therefore individually hookable. But HF's stock
float attention runs the same ops inline (`torch.matmul`, `F.softmax`,
`torch.matmul` — no modules), so the harness's by-name hook matching
finds no float-side counterpart to diff against. Result: the harness
silently skips three of the most interesting ops.

What this module does
---------------------
1. `wrap_attention_forward_float(attn)` — replaces the float attention's
   forward with a closure that uses `FloatMatmul` / `FloatSoftmax` modules
   instead of inline `torch.matmul` / `F.softmax`. Math is bit-identical
   to HF's `eager_attention_forward` (we verified the source in
   transformers 4.57.3). Exposes `_qk_matmul`, `_pv_matmul`, `_softmax`
   on the attention module.

2. `rename_int_attn_submodules(model)` — renames the int patcher's
   `_int_qk_matmul / _int_pv_matmul / _int_softmax` to the canonical
   names `_qk_matmul / _pv_matmul / _softmax`, matching the float side.
   The int patcher's `new_forward` captures these modules in a closure
   (not via attribute access), so renaming the attribute leaves runtime
   behavior unchanged.

3. `prepare_models_for_harness(float_model, int_cfg)` — applies (1) to
   the float model, builds the int student (`patch_model_int_nonmatmul`
   + `patch_model_int_cast`), and applies (2). Forces eager attention
   dispatch on both (the wrappers parse the eager additive-mask format,
   not the sdpa bool-mask format).

RoPE caveat
-----------
The int patcher attaches an `IntRopeApply` module as `attn._int_rope`
but never invokes it — its `new_forward` calls the float
`apply_rotary_pos_emb` directly. So RoPE is currently not integerized
in this codebase and there is no per-op error to measure for it. The
harness reflects that: no `_rope` submodule is exposed.
"""

from __future__ import annotations

import copy
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

from difr_expt.patch_hf_model import (
    ATTENTION_CLASS_NAMES,
    IntOpsConfig,
    _repeat_kv,
    patch_model_int_nonmatmul,
)
from difr_expt.int_cast import patch_model_int_cast


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


def rename_int_attn_submodules(model: nn.Module) -> int:
    """Rename `_int_qk_matmul → _qk_matmul`, `_int_pv_matmul → _pv_matmul`,
    `_int_softmax → _softmax` on every int-patched attention so the names
    match the float wrapper's canonical names.

    The int patcher's `new_forward` captures the matmul/softmax modules
    as local closure variables, so this rename does not affect runtime.
    Returns the number of attention modules touched.
    """
    n = 0
    for _name, m in model.named_modules():
        if m.__class__.__name__ not in ATTENTION_CLASS_NAMES:
            continue
        for src_attr, dst_attr in [
            ("_int_qk_matmul", "_qk_matmul"),
            ("_int_pv_matmul", "_pv_matmul"),
            ("_int_softmax", "_softmax"),
        ]:
            if hasattr(m, src_attr):
                setattr(m, dst_attr, getattr(m, src_attr))
                delattr(m, src_attr)
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


def prepare_models_for_harness(
    float_model: nn.Module,
    ops_cfg: IntOpsConfig,
    weight_bits: int = 16,
    activation_bits: int = 16,
    include_lm_head: bool = True,
) -> tuple[nn.Module, nn.Module]:
    """Prep one float teacher and one full-int student for the harness with
    matching submodule names across the attention sub-ops.

    Steps:
      1. Force eager attention on the float model (config-level).
      2. Deepcopy the float model into `int_model` before wrapping (so
         the int model's class structure is the un-wrapped original;
         the int patcher's own attention wrapper will run on it).
      3. Wrap the float model's attentions with the float-eager wrapper
         (exposes `_qk_matmul`, `_pv_matmul`, `_softmax`).
      4. Patch the int model with `patch_model_int_nonmatmul` (replaces
         RMSNorm / SiLU and installs the int attention closure) and
         `patch_model_int_cast` (every Linear → IntLinear).
      5. Rename `_int_*` → `_*` on int side so names match float side.

    Returns (float_model, int_model). Caller is responsible for `.to(device)`
    and `.eval()` after this returns.
    """
    force_eager_attn(float_model)

    # Deepcopy BEFORE wrapping the float model. If we wrapped first and
    # then deepcopied, the int patcher's _wrap_attention_forward would
    # build a new IntMatmul/IntSoftmax stack on top of our float wrapper,
    # double-replacing forward and producing nonsense.
    int_model = copy.deepcopy(float_model).eval()
    for p in int_model.parameters():
        p.requires_grad = False

    wrap_all_attention_forwards_float(float_model)

    patch_model_int_nonmatmul(int_model, ops_cfg)
    patch_model_int_cast(
        int_model,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        include_lm_head=include_lm_head,
    )
    rename_int_attn_submodules(int_model)
    return float_model, int_model
