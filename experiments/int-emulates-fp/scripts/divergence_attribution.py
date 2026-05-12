"""Attribute the student-teacher logit divergence to specific ops/modules.

For each transformer block we hook:
  - input_layernorm (RMSNorm)
  - self_attn.{q,k,v,o}_proj (linears)
  - self_attn (full attention output, captures softmax effect)
  - post_attention_layernorm (RMSNorm)
  - mlp.{gate,up,down}_proj (linears)
  - mlp (full MLP output, captures SiLU effect)
  - the whole layer (residual-stream divergence at the block boundary)

Plus model.embed_tokens, model.norm, and lm_head at the boundaries.

The student and teacher share the same module hierarchy (HF Qwen2), so we
pair by `named_modules()` name. For each pair we capture the output and
compute L2 norm of (student - teacher) at the position-mean granularity.

Output: JSON with per-module L2 + L1 + cosine_sim, plus a `summary` block
that aggregates by module-role (q_proj across all layers, etc.).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

from difr_expt.train_emulate import build_models, DTYPE_MAP


def capture_module_outputs(module: nn.Module, names_filter, store: dict) -> list:
    """Register forward hooks on every named submodule matching `names_filter`.
    Output capture stores tensor as cpu fp32, detached."""
    handles = []
    for name, m in module.named_modules():
        if not names_filter(name, m):
            continue
        def make_hook(n):
            def hook(_mod, _in, output):
                t = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(t):
                    store[n] = t.detach().float().cpu()
            return hook
        handles.append(m.register_forward_hook(make_hook(name)))
    return handles


def hook_filter(name: str, m: nn.Module) -> bool:
    """Match the interesting boundaries in Qwen2."""
    if name in ("model.embed_tokens", "model.norm", "lm_head"):
        return True
    # Per-layer modules; match by suffix
    suffixes = (
        ".input_layernorm",
        ".post_attention_layernorm",
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".self_attn",
        ".mlp.gate_proj",
        ".mlp.up_proj",
        ".mlp.down_proj",
        ".mlp",
    )
    # Match layer roots like "model.layers.5"
    if re.fullmatch(r"model\.layers\.\d+", name):
        return True
    return any(name.endswith(s) for s in suffixes)


def role_of(name: str) -> str:
    """Group module name into a coarse role for aggregation."""
    if name == "model.embed_tokens":
        return "embed"
    if name == "model.norm":
        return "model.norm"
    if name == "lm_head":
        return "lm_head"
    for tail in (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp",
    ):
        if name.endswith(tail):
            return tail
    if re.fullmatch(r"model\.layers\.\d+", name):
        return "layer_block"
    return "other"


def layer_index_of(name: str) -> int:
    m = re.search(r"model\.layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--teacher-source", default="published")
    ap.add_argument("--teacher-id", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--init-from-teacher", action="store_true", default=True)
    ap.add_argument("--load-trained", default=None, help="optional best.pt to load")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[args.dtype]
    print(f"loading models in {dtype} on {device}")
    teacher, student, _ = build_models(
        model_name=args.model,
        teacher_source=args.teacher_source,
        teacher_id=args.teacher_id,
        teacher_precision="fp8_e4m3",
        teacher_block_size=32,
        teacher_quantize_act=True,
        dtype=dtype,
        device=device,
        weight_bits=24,
        activation_bits=24,
        rmsnorm_bits=24,
        softmax_lut_size=4096,
        softmax_x_min=-16.0,
        silu_lut_size=4096,
        attn_matmul_bits=24,
        trainable_matmul_weights=True,
        int_embedding=False,
        embedding_bits=24,
        int_lm_head=False,
        init_from_teacher=args.init_from_teacher,
        keep_fp32_ref=False,
        grad_checkpointing=False,
        patch_nonmatmul=True,
    )

    if args.load_trained:
        print(f"loading trained state from {args.load_trained}")
        from difr_expt.train_emulate import IntLinear, IntRMSNorm, IntSoftmaxModule, IntSiLUModule
        payload = torch.load(args.load_trained, weights_only=False, map_location="cpu")
        gammas = payload.get("rmsnorm_gamma", {})
        biases = payload.get("linear_bias", {})
        weights = payload.get("linear_weight_fp", {})
        sm_luts = payload.get("softmax_lut", {})
        silu_luts = payload.get("silu_lut", {})
        for name, m in student.named_modules():
            if isinstance(m, IntRMSNorm) and name in gammas:
                m.weight.data.copy_(gammas[name].to(m.weight.device, m.weight.dtype))
            elif isinstance(m, IntLinear):
                if name in biases and m.bias is not None and isinstance(m.bias, nn.Parameter):
                    m.bias.data.copy_(biases[name].to(m.bias.device, m.bias.dtype))
                if name in weights and m.weight_fp is not None:
                    m.weight_fp.data.copy_(weights[name].to(m.weight_fp.device, m.weight_fp.dtype))
            elif isinstance(m, IntSoftmaxModule) and name in sm_luts and m.lut is not None:
                m.lut.data.copy_(sm_luts[name].to(m.lut.device, m.lut.dtype))
            elif isinstance(m, IntSiLUModule) and name in silu_luts and m.lut is not None:
                m.lut.data.copy_(silu_luts[name].to(m.lut.device, m.lut.dtype))

    teacher.eval(); student.eval()
    s_store, t_store = {}, {}
    capture_module_outputs(student, hook_filter, s_store)
    capture_module_outputs(teacher, hook_filter, t_store)

    prompts: list[torch.Tensor] = torch.load(args.prompts, weights_only=False)
    # Use the held-out 20 eval prompts (last 20), take first N for attribution.
    eval_prompts = prompts[-20:][:args.n_prompts]
    print(f"using {len(eval_prompts)} prompts for attribution")

    # Accumulate per-module L2/L1/cosine across prompts.
    accum: dict[str, dict[str, list[float]]] = {}
    with torch.inference_mode():
        for i, ids in enumerate(eval_prompts):
            s_store.clear(); t_store.clear()
            input_ids = ids.to(device).unsqueeze(0)
            teacher(input_ids)
            student(input_ids)
            common = sorted(set(s_store) & set(t_store))
            for name in common:
                s = s_store[name]; t = t_store[name]
                if s.shape != t.shape:
                    continue
                diff = s.float() - t.float()
                # Per-position L2 along the last dim, averaged over positions.
                if diff.dim() >= 1:
                    l2 = diff.norm(dim=-1).mean().item()
                    l1 = diff.abs().sum(dim=-1).mean().item()
                    cos = nn.functional.cosine_similarity(
                        s.float().reshape(-1, s.shape[-1]),
                        t.float().reshape(-1, t.shape[-1]),
                        dim=-1,
                    ).mean().item()
                else:
                    l2 = diff.norm().item(); l1 = diff.abs().sum().item(); cos = 1.0
                a = accum.setdefault(name, {"l2": [], "l1": [], "cos": []})
                a["l2"].append(l2); a["l1"].append(l1); a["cos"].append(cos)
            print(f"  prompt {i+1}/{len(eval_prompts)}: paired {len(common)} modules")

    # Per-module aggregates.
    per_module = {}
    for name, a in accum.items():
        per_module[name] = {
            "role": role_of(name),
            "layer_idx": layer_index_of(name),
            "l2_mean": sum(a["l2"]) / len(a["l2"]),
            "l1_mean": sum(a["l1"]) / len(a["l1"]),
            "cos_mean": sum(a["cos"]) / len(a["cos"]),
            "n_obs": len(a["l2"]),
        }

    # Aggregate by role.
    by_role: dict[str, dict[str, list[float]]] = {}
    for name, info in per_module.items():
        r = info["role"]
        d = by_role.setdefault(r, {"l2": [], "l1": [], "cos": []})
        d["l2"].append(info["l2_mean"])
        d["l1"].append(info["l1_mean"])
        d["cos"].append(info["cos_mean"])

    role_summary = {}
    for r, d in by_role.items():
        role_summary[r] = {
            "n_layers": len(d["l2"]),
            "l2_mean": sum(d["l2"]) / len(d["l2"]),
            "l2_max": max(d["l2"]),
            "l1_mean": sum(d["l1"]) / len(d["l1"]),
            "cos_mean": sum(d["cos"]) / len(d["cos"]),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "per_module": per_module,
            "role_summary": role_summary,
        }, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\n=== Role summary (L2 mean across layers, sorted by L2) ===")
    for r, s in sorted(role_summary.items(), key=lambda kv: -kv[1]["l2_mean"]):
        print(f"  {r:30s}  n={s['n_layers']:3d}  l2_mean={s['l2_mean']:.4f}  l2_max={s['l2_max']:.4f}  cos={s['cos_mean']:.4f}")


if __name__ == "__main__":
    main()
