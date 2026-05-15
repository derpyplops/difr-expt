"""Per-named-module L2 error harness — real hardware FP8 only.

Runs a bf16 teacher and a real-FP8 student (loaded from a pre-quantized
checkpoint via HF + compressed_tensors, with weights kept in
`torch.float8_e4m3fn` and matmuls dispatching through
`torch._scaled_mm` on Hopper) and reports L2 divergence at every named
submodule. Two flavors per module:

  - **propagated** = ||int.output[m] − float.output[m]||₂
        Compounded error: how off is the int model's value at m when it has
        been receiving int-approximated inputs upstream. This is what the
        downstream graph actually sees.
  - **isolated** = ||M_int[m](float.args[m], float.kwargs[m]) − float.output[m]||₂
        Stateless error: how much does m alone contribute when fed the
        clean float inputs. Useful for ranking which op-classes are
        intrinsically lossy.

Final per-token logit metrics (L2, KL, top-1, top-5) are emitted too — the
ground-truth target Luke called out.

The fake-quant emulation modes have been removed: this harness exercises
the real FP8 GEMM and refuses to run otherwise (see `ScaledMmProbe`).
Requires a Hopper-class (SM_89+) GPU.

CLI:
    python -m difr_expt.run_harness \\
        --model Qwen/Qwen2.5-0.5B \\
        --student-model RedHatAI/Qwen2.5-0.5B-FP8-dynamic \\
        --dtype bfloat16 --device cuda \\
        --n-prompts 16 --max-len 256 \\
        --out experiments/layer-harness/reports/results-$(date +%F).json

The .json gets a sister .md table next to it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from difr_expt.harness_attn_wrap import (
    force_eager_attn,
    wrap_all_attention_forwards_float,
)
from difr_expt.metrics import (
    kl_div_ref_to_cand,
    logit_l2,
    top1_match,
    topk_overlap,
)
from difr_expt.run_baseline import DTYPE_MAP, load_prompts, pick_device, Config as BaselineConfig


# ---------------------------------------------------------------------------
# Module tagging
# ---------------------------------------------------------------------------


def _block_index(name: str) -> int:
    """Parse `layers.{i}.…` if present, else -1 (for embed_tokens / norm / lm_head)."""
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        if i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def _family(name: str, module: nn.Module) -> str:
    """Coarse family tag. Linears get q/k/v/o/gate/up/down, others get an
    op-class tag from the module class name."""
    short = name.rsplit(".", 1)[-1]
    cls = module.__class__.__name__
    if "lm_head" in name:
        return "head"
    if "embed_tokens" in name or "Embedding" in cls:
        return "embed"
    if short in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return short[0]  # q / k / v / o
    if short == "gate_proj":
        return "gate"
    if short == "up_proj":
        return "up"
    if short == "down_proj":
        return "down"
    if name.endswith("._qk_matmul"):
        return "qk_matmul"
    if name.endswith("._pv_matmul"):
        return "pv_matmul"
    if name.endswith("._softmax"):
        return "softmax"
    if "RMSNorm" in cls or "rmsnorm" in cls.lower():
        # The model's final norm has name "model.norm"
        if name.endswith(".norm") and "layers" not in name:
            return "final_norm"
        return "rmsnorm"
    if "SiLU" in cls or "silu" in cls.lower():
        return "silu"
    if "Softmax" in cls or "softmax" in cls.lower():
        return "softmax"
    if "Rope" in cls or "rope" in cls.lower() or "Rotary" in cls:
        return "rotary"
    if "Attention" in cls:
        return "attn"
    if "MLP" in cls or "FeedForward" in cls:
        return "mlp"
    if "Block" in cls or "DecoderLayer" in cls:
        return "block"
    return "other"


# ---------------------------------------------------------------------------
# Hook plumbing
# ---------------------------------------------------------------------------


# Module classes we never want to hook (containers / no-ops).
_SKIP_CLASSES = (nn.ModuleList, nn.ModuleDict, nn.Sequential, nn.Identity)


def _should_hook(name: str, module: nn.Module) -> bool:
    if name == "":
        return False
    if isinstance(module, _SKIP_CLASSES):
        return False
    return True


def _detach_obj(x):
    """Return a detached clone of any tensors inside x; pass other types through.

    Args/outputs can contain tensors, tuples of tensors, None, ints, etc.
    For captured outputs we only ever care about tensors; this also makes
    captures cheap to keep around across the second forward.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(_detach_obj(v) for v in x)
    if isinstance(x, list):
        return [_detach_obj(v) for v in x]
    if isinstance(x, dict):
        return {k: _detach_obj(v) for k, v in x.items()}
    return x  # ints, floats, None, custom HF cache objects (we drop those later)


def _first_tensor(x):
    """Return the first tensor inside x (handles Tensor / tuple / list / dict /
    HF ModelOutput). Returns None if no tensor is found."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    # Common HF pattern: dataclass-like object with .last_hidden_state, etc.
    if hasattr(x, "to_tuple"):
        try:
            return _first_tensor(x.to_tuple())
        except Exception:
            return None
    return None


def install_capture_hooks(model: nn.Module):
    """Hook every interesting named module. Returns (captures, handles, names).

    captures[name] = {"args": detached_args, "kwargs": detached_kwargs,
                      "output": detached_output}
    """
    captures: dict[str, dict] = {}
    handles: list = []
    names: list[str] = []

    for name, mod in model.named_modules():
        if not _should_hook(name, mod):
            continue
        names.append(name)

        def make_hook(name=name):
            def hook(module, args, kwargs, output):
                captures[name] = {
                    "args": _detach_obj(args),
                    "kwargs": _detach_obj(kwargs),
                    "output": _detach_obj(output),
                }
            return hook

        h = mod.register_forward_hook(make_hook(), with_kwargs=True)
        handles.append(h)

    return captures, handles, names


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def _pairwise_l2_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Per-position L2 along the last dim; return mean / p50 / p99 / worst.

    a, b: same shape, tensors. We promote to fp32 before subtracting.
    """
    diff = a.float() - b.float()
    # If 1-D (e.g. final logits accidentally hit this path), treat the whole
    # vector as one "position".
    if diff.dim() == 0:
        v = diff.abs().item()
        return {"mean": v, "p50": v, "p99": v, "worst": v}
    if diff.dim() == 1:
        v = diff.norm().item()
        return {"mean": v, "p50": v, "p99": v, "worst": v}
    per = diff.norm(dim=-1).flatten().float()
    return {
        "mean": per.mean().item(),
        "p50": per.median().item(),
        "p99": per.quantile(0.99).item(),
        "worst": per.max().item(),
    }


def _safe_l2(a, b) -> Optional[dict[str, float]]:
    """L2 stats between two hook outputs. Returns None if shapes/types
    don't match or there's no tensor to compare."""
    ta = _first_tensor(a)
    tb = _first_tensor(b)
    if ta is None or tb is None:
        return None
    if ta.shape != tb.shape:
        return None
    if ta.dtype.is_floating_point is False or tb.dtype.is_floating_point is False:
        # e.g. token ids going into embed_tokens — not comparable as L2.
        return None
    return _pairwise_l2_stats(ta, tb)


# ---------------------------------------------------------------------------
# Isolated call
# ---------------------------------------------------------------------------


def _strip_cache_kwargs(kwargs: dict) -> dict:
    """Make a stateless copy of kwargs for an isolated call.

    Drops the HF kv-cache (`past_key_values`, `cache_position`) so calling an
    attention module on these args doesn't mutate cached state and double-
    book positions. Everything else (attention_mask, position_embeddings) is
    preserved — those are pure inputs.
    """
    out = dict(kwargs)
    if "past_key_values" in out:
        out["past_key_values"] = None
    if "cache_position" in out:
        out["cache_position"] = None
    if "use_cache" in out:
        out["use_cache"] = False
    return out


def _isolated_call(int_model: nn.Module, name: str, args, kwargs):
    """Run the int student's submodule `name` on the float teacher's args/kwargs.

    Returns the module's output tensor, or None on any exception (we expect
    the occasional fallthrough — e.g. embed_tokens fed long ids that the
    IntEmbedding handles but a vanilla nn.Embedding wouldn't, or a custom
    HF cache type we can't reconstruct)."""
    try:
        int_mod = int_model.get_submodule(name)
    except AttributeError:
        return None
    try:
        with torch.inference_mode():
            return int_mod(*args, **_strip_cache_kwargs(kwargs))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-prompt run
# ---------------------------------------------------------------------------


@dataclass
class ModuleRow:
    name: str
    block: int
    family: str
    cls_float: str
    cls_int: str
    shape: tuple
    prop: Optional[dict[str, float]] = None
    iso: Optional[dict[str, float]] = None


def run_one_prompt(
    float_model: nn.Module,
    int_model: nn.Module,
    input_ids: torch.Tensor,
    device: str,
) -> tuple[list[ModuleRow], dict[str, float], torch.Tensor, torch.Tensor]:
    """Forward both models on `input_ids` and produce per-module L2 rows
    plus aggregate logit metrics.

    Returns (rows, logit_metrics, ref_logits, cand_logits).
    """
    cap_float, h_float, names_float = install_capture_hooks(float_model)
    cap_int, h_int, names_int = install_capture_hooks(int_model)

    ids = input_ids.to(device).unsqueeze(0)
    with torch.inference_mode():
        out_float = float_model(ids)
        out_int = int_model(ids)

    ref_logits = out_float.logits if hasattr(out_float, "logits") else out_float[0]
    cand_logits = out_int.logits if hasattr(out_int, "logits") else out_int[0]

    # Detach hooks BEFORE the iso loop. Otherwise each iso call (which invokes
    # a module's forward) re-fires its forward hook and overwrites
    # cap_int[name] for that module and all its children — silently turning
    # subsequent prop_l2 readings into iso_l2 readings for those children.
    remove_hooks(h_float)
    remove_hooks(h_int)

    common = [n for n in names_float if n in cap_float and n in cap_int]
    rows: list[ModuleRow] = []
    for name in common:
        f_cap = cap_float[name]
        i_cap = cap_int[name]
        f_mod = float_model.get_submodule(name)
        try:
            i_mod = int_model.get_submodule(name)
        except AttributeError:
            i_mod = f_mod
        f_out = _first_tensor(f_cap["output"])
        shape = tuple(f_out.shape) if f_out is not None else ()
        row = ModuleRow(
            name=name,
            block=_block_index(name),
            family=_family(name, f_mod),
            cls_float=f_mod.__class__.__name__,
            cls_int=i_mod.__class__.__name__,
            shape=shape,
        )
        row.prop = _safe_l2(f_cap["output"], i_cap["output"])

        iso_out = _isolated_call(int_model, name, f_cap["args"], f_cap["kwargs"])
        if iso_out is not None:
            row.iso = _safe_l2(f_cap["output"], iso_out)

        rows.append(row)

    # Final logits metrics.
    l2 = logit_l2(ref_logits, cand_logits).mean().item()
    kl = kl_div_ref_to_cand(ref_logits, cand_logits).mean().item()
    t1 = top1_match(ref_logits, cand_logits).float().mean().item()
    t5 = topk_overlap(ref_logits, cand_logits, k=5).mean().item()
    logit_metrics = {"logit_l2_mean": l2, "kl_mean": kl, "top1_match": t1, "top5_overlap": t5}

    return rows, logit_metrics, ref_logits.detach().cpu(), cand_logits.detach().cpu()


# ---------------------------------------------------------------------------
# Aggregation across prompts
# ---------------------------------------------------------------------------


def _accumulate(rows_per_prompt: list[list[ModuleRow]]) -> list[dict]:
    """Merge per-prompt rows into one row per module name with mean-of-mean,
    max-of-p99, max-of-worst across prompts.

    We keep it simple: per-prompt stats are already point estimates over
    that prompt's positions. Mean across prompts approximates the
    population mean (each prompt contributes one observation per stat);
    p99/worst we take the worst across prompts so the table flags the
    bad cases.
    """
    by_name: dict[str, list[ModuleRow]] = {}
    for rows in rows_per_prompt:
        for r in rows:
            by_name.setdefault(r.name, []).append(r)

    out: list[dict] = []
    for name, rows in by_name.items():
        base = rows[0]

        def reduce(field_name: str):
            vals = [getattr(r, field_name) for r in rows if getattr(r, field_name) is not None]
            if not vals:
                return None
            return {
                "mean": sum(v["mean"] for v in vals) / len(vals),
                "p50": sum(v["p50"] for v in vals) / len(vals),
                "p99": max(v["p99"] for v in vals),
                "worst": max(v["worst"] for v in vals),
                "n_prompts": len(vals),
            }

        out.append({
            "name": name,
            "block": base.block,
            "family": base.family,
            "cls_float": base.cls_float,
            "cls_int": base.cls_int,
            "shape": list(base.shape),
            "prop": reduce("prop"),
            "iso": reduce("iso"),
        })
    return out


def _aggregate_logits(metrics_per_prompt: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_per_prompt:
        return {}
    keys = metrics_per_prompt[0].keys()
    return {k: sum(m[k] for m in metrics_per_prompt) / len(metrics_per_prompt) for k in keys}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v == 0.0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e4:
        return f"{v:.2e}"
    return f"{v:.4g}"


def render_table(rows: list[dict], logit_metrics: dict[str, float], header: dict) -> str:
    """Markdown table sorted by (block, name)."""
    lines = []
    lines.append(f"# layer-harness results")
    lines.append("")
    for k, v in header.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Logit metrics (mean over prompts, over positions)")
    lines.append("")
    for k, v in logit_metrics.items():
        lines.append(f"- {k}: {_fmt(v)}")
    lines.append("")
    lines.append("## Per-module L2")
    lines.append("")
    lines.append("`prop` = propagated (int forward vs float forward), "
                 "`iso` = isolated (int module on float input vs float output).")
    lines.append("")
    lines.append("| name | blk | family | shape | prop.mean | prop.p99 | prop.worst | iso.mean | iso.p99 | iso.worst |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    def sort_key(r):
        return (r["block"] if r["block"] >= 0 else -1, r["name"])

    for r in sorted(rows, key=sort_key):
        prop = r.get("prop") or {}
        iso = r.get("iso") or {}
        lines.append(
            f"| {r['name']} | {r['block']} | {r['family']} | "
            f"{tuple(r['shape']) if r['shape'] else '—'} | "
            f"{_fmt(prop.get('mean'))} | {_fmt(prop.get('p99'))} | {_fmt(prop.get('worst'))} | "
            f"{_fmt(iso.get('mean'))} | {_fmt(iso.get('p99'))} | {_fmt(iso.get('worst'))} |"
        )

    # By-family roll-up.
    by_fam: dict[str, list[dict]] = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r)
    lines.append("")
    lines.append("## By family (mean of per-module prop.mean)")
    lines.append("")
    lines.append("| family | n | prop.mean | iso.mean |")
    lines.append("|---|---|---|---|")
    for fam, rs in sorted(by_fam.items()):
        prop_vals = [r["prop"]["mean"] for r in rs if r.get("prop")]
        iso_vals = [r["iso"]["mean"] for r in rs if r.get("iso")]
        prop_avg = sum(prop_vals) / len(prop_vals) if prop_vals else None
        iso_avg = sum(iso_vals) / len(iso_vals) if iso_vals else None
        lines.append(f"| {fam} | {len(rs)} | {_fmt(prop_avg)} | {_fmt(iso_avg)} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    model: str = "Qwen/Qwen2.5-0.5B"
    student_model: str = ""  # required: HF id of a pre-quantized FP8 checkpoint
    dtype: str = "bfloat16"
    device: str = "auto"
    n_prompts: int = 16
    max_len: int = 256
    seed: int = 42
    out: str = ""  # path to .json; sister .md is written next to it
    # Dataset
    dataset: str = "Salesforce/wikitext"
    dataset_config: str | None = "wikitext-103-raw-v1"
    dataset_split: str = "train"


def _baseline_cfg_for_loading(cfg: HarnessConfig) -> BaselineConfig:
    """Adapter so we can reuse run_baseline.load_prompts unchanged."""
    return BaselineConfig(
        model=cfg.model, dtype=cfg.dtype, n_prompts=cfg.n_prompts, max_len=cfg.max_len,
        dataset=cfg.dataset, dataset_config=cfg.dataset_config, dataset_split=cfg.dataset_split,
    )


class ScaledMmProbe:
    """Context manager that intercepts `torch._scaled_mm` and counts calls.

    Used to verify that a "real FP8" forward actually exercises the FP8 GEMM
    path — if the count is zero after a forward, the harness was silently
    measuring fp16/bf16 instead of real FP8, regardless of what the model's
    module classes claim.
    """

    def __init__(self):
        self.count = 0
        self.shapes: list[tuple] = []
        self._orig = None

    def __enter__(self):
        self._orig = torch._scaled_mm
        probe = self

        def wrapper(*args, **kwargs):
            probe.count += 1
            if args and isinstance(args[0], torch.Tensor):
                probe.shapes.append(tuple(args[0].shape))
            return probe._orig(*args, **kwargs)

        torch._scaled_mm = wrapper
        return self

    def __exit__(self, exc_type, exc, tb):
        torch._scaled_mm = self._orig


def build_models(
    float_model: nn.Module, cfg: HarnessConfig
) -> tuple[nn.Module, nn.Module]:
    """Prepare the bf16 teacher and the real-FP8 student.

    Loads `cfg.student_model` via HF + compressed_tensors, strips the lazy
    decompress hook so the FP8 weights stay in `torch.float8_e4m3fn`, and
    swaps every FP8 nn.Linear for an `FP8Linear` whose forward dispatches
    through `torch._scaled_mm` — a real Hopper FP8 GEMM. Both teacher and
    student get the float-eager attention wrap so the softmax / Q@K.T / P@V
    rows are hookable on both. FP8-dynamic doesn't quantize those three
    ops, so we expect near-zero divergence on them — which is itself the
    real measurement.

    Raises if the checkpoint doesn't load with FP8 weights, or (downstream
    via `ScaledMmProbe`) if `torch._scaled_mm` is never called during a
    forward.
    """
    if not cfg.student_model:
        raise ValueError("`--student-model` is required (HF id of a pre-quantized FP8 checkpoint)")

    force_eager_attn(float_model)

    # We deliberately do NOT pass dtype= here. compressed_tensors will load
    # the FP8 weights at their native dtype and set up its lazy decompress
    # hook. Right after this returns, every quantized Linear still has an
    # fp8_e4m3fn `weight` and a separate `weight_scale` — we strip the
    # hook and swap to FP8Linears (real _scaled_mm) before any forward
    # gets a chance to silently dequantize.
    print(f"[harness] loading FP8 student {cfg.student_model!r}…")
    int_model = AutoModelForCausalLM.from_pretrained(cfg.student_model).eval()
    for p in int_model.parameters():
        p.requires_grad = False

    # Sanity: confirm the checkpoint actually arrived with FP8 weights
    # before we touch anything.
    fp_low_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
    if hasattr(torch, "float4_e2m1fn_x2"):
        fp_low_dtypes.add(torch.float4_e2m1fn_x2)
    pre_swap = [
        (name, str(m.weight.dtype))
        for name, m in int_model.named_modules()
        if isinstance(m, nn.Linear) and m.weight.dtype in fp_low_dtypes
    ]
    if not pre_swap:
        scheme_count = sum(1 for m in int_model.modules() if hasattr(m, "quantization_scheme"))
        raise RuntimeError(
            f"FP8 path verification failed: no Linear has a low-precision weight dtype "
            f"on freshly-loaded student {cfg.student_model!r}. Found {scheme_count} modules "
            f"with a `quantization_scheme` attribute but none with FP8/FP4 weight dtypes — "
            f"the checkpoint loaded but didn't actually keep low-precision weights. "
            f"Check that the repo's quantization_config has float-quantized weights and "
            f"that compressed_tensors is installed."
        )
    pre_swap_dtypes = sorted({dt for _, dt in pre_swap})
    print(f"[harness] checkpoint has {len(pre_swap)} Linears with low-precision "
          f"weights ({pre_swap_dtypes}); swapping to _scaled_mm FP8Linears now…")

    # Strip the lazy-decompress hook + swap every FP8 nn.Linear with an
    # FP8Linear that uses `torch._scaled_mm`. Otherwise the default HF
    # behavior on first forward is to dequantize the FP8 weights to bf16
    # in-place and run a plain bf16 matmul — that's emulation, not the
    # real FP8 path. Verified empirically (the ScaledMmProbe shows zero
    # `_scaled_mm` calls otherwise).
    from difr_expt.fp8_hw_linear import (
        disable_compressed_tensors_decompress,
        replace_compressed_linears_with_fp8,
    )
    disable_compressed_tensors_decompress(int_model)
    n_swapped = replace_compressed_linears_with_fp8(int_model)
    print(f"[harness] swapped {n_swapped} compressed Linears → FP8Linear "
          f"(real torch._scaled_mm)")

    force_eager_attn(int_model)
    wrap_all_attention_forwards_float(float_model)
    wrap_all_attention_forwards_float(int_model)
    return float_model, int_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=HarnessConfig.model,
                    help="HF repo id of the bf16 teacher")
    ap.add_argument("--student-model", required=True,
                    help="HF repo id of the pre-quantized FP8 checkpoint to evaluate")
    ap.add_argument("--dtype", default=HarnessConfig.dtype, choices=list(DTYPE_MAP.keys()),
                    help="teacher dtype (bf16 recommended on Hopper)")
    ap.add_argument("--device", default=HarnessConfig.device)
    ap.add_argument("--n-prompts", type=int, default=HarnessConfig.n_prompts)
    ap.add_argument("--max-len", type=int, default=HarnessConfig.max_len)
    ap.add_argument("--seed", type=int, default=HarnessConfig.seed)
    ap.add_argument("--out", required=True,
                    help="path to results .json; sister .md goes next to it")
    args = ap.parse_args()

    cfg = HarnessConfig(
        model=args.model, student_model=args.student_model,
        dtype=args.dtype, device=args.device,
        n_prompts=args.n_prompts, max_len=args.max_len, seed=args.seed,
        out=args.out,
    )

    if not torch.cuda.is_available():
        raise SystemExit(
            "The FP8 harness requires a CUDA device; this box has none. Run on a "
            "Hopper-class GPU (H100/H200) where torch._scaled_mm is available."
        )
    caps = torch.cuda.get_device_capability(0)
    if caps < (8, 9):
        raise SystemExit(
            f"The FP8 harness needs SM_89+ (Ada Lovelace / Hopper / Blackwell); "
            f"this GPU is SM_{caps[0]}{caps[1]}. FP8 GEMM via torch._scaled_mm "
            f"is not available."
        )

    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    dtype = DTYPE_MAP[cfg.dtype]
    print(f"[harness] model={cfg.model} dtype={cfg.dtype} device={device} "
          f"n_prompts={cfg.n_prompts} max_len={cfg.max_len}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    float_model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device).eval()
    for p in float_model.parameters():
        p.requires_grad = False

    t0 = time.time()
    float_model, int_model = build_models(float_model, cfg)
    float_model = float_model.to(device).eval()
    int_model = int_model.to(device).eval()
    print(f"[harness] float wrapper + int student built in {time.time() - t0:.1f}s")

    prompts = load_prompts(_baseline_cfg_for_loading(cfg), tokenizer)
    print(f"[harness] loaded {len(prompts)} prompts; lengths={[len(p) for p in prompts]}")

    rows_per_prompt: list[list[ModuleRow]] = []
    logits_per_prompt: list[dict[str, float]] = []
    with ScaledMmProbe() as fp8_probe:
        for i, ids in enumerate(prompts):
            t0 = time.time()
            ids_t = torch.tensor(ids, dtype=torch.long)
            rows, lm, _, _ = run_one_prompt(float_model, int_model, ids_t, device)
            rows_per_prompt.append(rows)
            logits_per_prompt.append(lm)
            if i == 0:
                if fp8_probe.count == 0:
                    raise RuntimeError(
                        "FP8 path verification failed: torch._scaled_mm was never called "
                        "during the first forward. The student is not running real FP8 GEMM. "
                        "Confirm the checkpoint actually has FP8 weights and the "
                        "FP8Linear swap completed."
                    )
                print(f"[harness] _scaled_mm called {fp8_probe.count} times during "
                      f"prompt 0 forward — real FP8 path is live ✓")
            print(f"  prompt {i}: {len(rows)} modules, "
                  f"logit_l2={lm['logit_l2_mean']:.4g}, kl={lm['kl_mean']:.4g}, "
                  f"top1={lm['top1_match']:.3f}, top5={lm['top5_overlap']:.3f} "
                  f"({time.time() - t0:.1f}s)")

    rows = _accumulate(rows_per_prompt)
    logit_metrics = _aggregate_logits(logits_per_prompt)

    header = {
        "model": cfg.model, "student_model": cfg.student_model,
        "dtype": cfg.dtype, "device": device,
        "n_prompts": len(prompts), "max_len": cfg.max_len,
        "scaled_mm_calls_prompt0": fp8_probe.count,
    }

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "header": header, "logit_metrics": logit_metrics, "rows": rows,
    }, indent=2))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_table(rows, logit_metrics, header))
    print(f"[harness] wrote {out_path} and {md_path.name}")


if __name__ == "__main__":
    main()
