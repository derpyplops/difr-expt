"""Run the int-cast vs float baseline divergence measurement.

For each prompt, teacher-forces the same token sequence through both the
reference model and an int-cast copy, then computes per-position divergence
metrics (top-1 match, top-5 overlap, logit L2, KL, Gumbel margin).

Usage (smoke test on CPU):
    python -m difr_expt.run_baseline --model sshleifer/tiny-gpt2 --n-prompts 4 --max-len 64

Usage (real run on H100):
    python -m difr_expt.run_baseline \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --dtype bfloat16 --n-prompts 1000 --max-len 512 \\
        --out results/llama_8b_int32.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from difr_expt.int_cast import patch_model_int_cast, patch_model_int_embedding, set_true_int_matmul, calibrate_smooth_scales
from difr_expt.int_ops import set_true_int_path
from difr_expt.patch_hf_model import IntOpsConfig, patch_model_int_nonmatmul
import torch.nn as nn
from difr_expt.metrics import (
    aggregate,
    kl_div_ref_to_cand,
    logit_l2,
    post_gumbel_margin,
    top1_match,
    topk_overlap,
)


@dataclass
class Config:
    model: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    n_prompts: int = 4
    max_len: int = 64
    weight_bits: int = 16
    activation_bits: int = 16
    true_int_matmul: bool = False
    quant_scheme: str = "symmetric"
    group_size: int = 128
    matmul_dtype: str = "auto"
    no_quant: bool = False
    dataset: str = "Salesforce/wikitext"
    dataset_config: str | None = "wikitext-103-raw-v1"
    dataset_split: str = "train"
    temperature: float = 1.0
    seed: int = 42
    device: str = "auto"
    out: str | None = None
    include_lm_head: bool = True
    skip_substrings: list[str] = field(default_factory=list)
    smoothquant_alpha: float | None = None  # if set, run SmoothQuant calibration
    smoothquant_n_calib: int = 16
    act_clip_quantile: float | None = None  # per-token clip quantile for activation quant
    cached_bf16: bool = False  # cache dequantized weight as nn.Parameter for identical cuBLAS dispatch
    # Non-matmul int ops (RMSNorm/softmax/SiLU/attn-matmul) — Phase 1 onward
    int_nonmatmul: bool = False
    rmsnorm_bits: int = 24
    rmsnorm_nr_iter: int = 2
    softmax_lut_size: int = 1024
    softmax_x_min: float = -16.0
    silu_lut_size: int = 4096
    attn_matmul_bits: int = 24
    # Ablation flags (Phase 3 / E)
    no_int_rmsnorm: bool = False
    no_int_softmax: bool = False
    no_int_silu: bool = False
    no_int_attn_matmul: bool = False
    # Literal int execution for non-matmul ops (Phase 4)
    true_int_nonmatmul: bool = False
    # Embedding table quantization
    int_embedding: bool = False
    embedding_bits: int = 24


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_prompts(cfg: Config, tokenizer) -> list[list[int]]:
    """Load `n_prompts` prompts and return their token-id lists.

    Defaults to Salesforce/wikitext (open, no auth) like Luke's repo. Falls
    back to toy prompts if loading fails (handy for CPU smoke tests with no
    internet).
    """
    try:
        if cfg.dataset_config:
            ds = load_dataset(cfg.dataset, cfg.dataset_config, split=cfg.dataset_split, streaming=True)
        else:
            ds = load_dataset(cfg.dataset, split=cfg.dataset_split, streaming=True)
    except Exception as e:
        print(f"[warn] dataset load failed ({e}); using toy prompts")
        return _toy_prompts(tokenizer, cfg.n_prompts, cfg.max_len)

    prompts: list[str] = []
    for ex in ds:
        if "conversation" in ex and ex["conversation"]:
            user_msg = next((t["content"] for t in ex["conversation"] if t.get("role") == "user"), None)
            if user_msg:
                prompts.append(user_msg)
        elif "messages" in ex and ex["messages"]:
            user_msg = next((m["content"] for m in ex["messages"] if m.get("role") == "user"), None)
            if user_msg:
                prompts.append(user_msg)
        elif "text" in ex:
            text = ex["text"].strip()
            # wikitext has lots of empty/short rows; filter those.
            if len(text) >= 100:
                prompts.append(text)
        if len(prompts) >= cfg.n_prompts * 2:  # over-collect since we'll filter again
            break

    if not prompts:
        return _toy_prompts(tokenizer, cfg.n_prompts, cfg.max_len)

    tokens = []
    for p in prompts:
        ids = tokenizer(p, truncation=True, max_length=cfg.max_len, add_special_tokens=True).input_ids
        if len(ids) >= 32:
            tokens.append(ids)
        if len(tokens) >= cfg.n_prompts:
            break
    return tokens[: cfg.n_prompts]


def _toy_prompts(tokenizer, n: int, max_len: int) -> list[list[int]]:
    seeds = [
        "The quick brown fox jumps over the lazy dog. " * 3,
        "Once upon a time in a faraway kingdom, there lived a curious mathematician. ",
        "To compute the area of a circle, multiply pi by the radius squared. " * 2,
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole. " * 2,
    ]
    out = []
    for i in range(n):
        ids = tokenizer(seeds[i % len(seeds)], truncation=True, max_length=max_len).input_ids
        out.append(ids)
    return out


class NoQuantLinear(nn.Module):
    """A drop-in replacement for nn.Linear that runs the matmul at a chosen dtype
    without quantization. Used to isolate the pure bf16-vs-fp32 reduction-order
    question from any int-cast noise.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None,
                 compute_dtype: torch.dtype, matmul_dtype: torch.dtype | None):
        super().__init__()
        # Keep weight in compute_dtype storage (matches what nn.Linear had).
        self.register_buffer("weight", weight.detach().clone(), persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().to(compute_dtype), persistent=True)
        else:
            self.bias = None
        self.compute_dtype = compute_dtype
        self.matmul_dtype = matmul_dtype  # None => fp32 matmul; else cast inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # When matmul_dtype is set, use F.linear so the bias is fused inside
        # cuBLAS addmm (matches nn.Linear bit-for-bit at the same dtype). When
        # matmul_dtype is None ("auto"), force the fp32 matmul path to mirror
        # the int-cast default behaviour.
        if self.matmul_dtype is None:
            orig_shape = x.shape
            x_flat = x.reshape(-1, orig_shape[-1])
            w = self.weight.to(torch.float32)
            xf = x_flat.to(torch.float32)
            out = xf @ w.t()
            out = out.to(self.compute_dtype).reshape(*orig_shape[:-1], self.weight.shape[0])
            if self.bias is not None:
                out = out + self.bias
            return out
        w = self.weight.to(self.matmul_dtype)
        xf = x.to(self.matmul_dtype)
        b = self.bias.to(self.matmul_dtype) if self.bias is not None else None
        out = torch.nn.functional.linear(xf, w, b)
        return out.to(self.compute_dtype)


def _patch_model_noquant(model, matmul_dtype, skip_substrings, include_lm_head):
    replaced: dict[str, NoQuantLinear] = {}

    def should_skip(name: str) -> bool:
        if not include_lm_head and "lm_head" in name:
            return True
        return any(s in name for s in skip_substrings)

    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            to_replace.append((name, module))

    for name, linear in to_replace:
        compute_dtype = linear.weight.dtype
        wrap = NoQuantLinear(
            weight=linear.weight,
            bias=linear.bias,
            compute_dtype=compute_dtype,
            matmul_dtype=matmul_dtype,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrap)
        replaced[name] = wrap
    return replaced


@torch.inference_mode()
def per_prompt_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits [seq_len, vocab] for a single 1D input_ids tensor."""
    out = model(input_ids.unsqueeze(0))
    return out.logits[0]


def compare(
    ref_model,
    int_model,
    prompts: list[list[int]],
    device: str,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    """Run both models on each prompt; aggregate metrics two ways:
      - 'all': across every position in every prompt (strict)
      - 'last': across the last position of each prompt only (Luke-style)
    """
    all_top1, all_top5, all_l2, all_kl, all_margin = [], [], [], [], []
    last_top1, last_top5, last_l2, last_kl, last_margin = [], [], [], [], []
    n_positions = 0

    rng = torch.Generator(device=device).manual_seed(seed)

    for ids in tqdm(prompts, desc="prompts"):
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        ref_logits = per_prompt_logits(ref_model, input_ids)  # [T, V]
        int_logits = per_prompt_logits(int_model, input_ids)

        v = min(ref_logits.shape[-1], int_logits.shape[-1])
        ref_logits = ref_logits[..., :v]
        int_logits = int_logits[..., :v]

        t1 = top1_match(ref_logits, int_logits)
        t5 = topk_overlap(ref_logits, int_logits, k=5)
        l2 = logit_l2(ref_logits, int_logits)
        kl = kl_div_ref_to_cand(ref_logits, int_logits, temperature=temperature)

        u = torch.empty_like(ref_logits, dtype=torch.float32)
        u.uniform_(1e-10, 1.0, generator=rng)
        gumbel = -torch.log(-torch.log(u))
        mg = post_gumbel_margin(ref_logits, int_logits, gumbel, temperature=temperature)

        all_top1.append(t1); all_top5.append(t5); all_l2.append(l2); all_kl.append(kl); all_margin.append(mg)
        # Last position of this prompt (the position that would have generated the next token).
        last_top1.append(t1[-1:]); last_top5.append(t5[-1:])
        last_l2.append(l2[-1:]); last_kl.append(kl[-1:]); last_margin.append(mg[-1:])

        n_positions += ref_logits.shape[0]

    all_metrics = {
        "top1_match": torch.cat(all_top1),
        "top5_overlap": torch.cat(all_top5),
        "logit_l2": torch.cat(all_l2),
        "kl_ref_cand": torch.cat(all_kl),
        "gumbel_margin": torch.cat(all_margin),
    }
    last_metrics = {
        "top1_match": torch.cat(last_top1),
        "top5_overlap": torch.cat(last_top5),
        "logit_l2": torch.cat(last_l2),
        "kl_ref_cand": torch.cat(last_kl),
        "gumbel_margin": torch.cat(last_margin),
    }

    summary: dict[str, Any] = {
        "n_positions": n_positions,
        "n_prompts": len(prompts),
        "all": aggregate(all_metrics),
        "last": aggregate(last_metrics),
    }
    for tag, m in (("all", all_metrics), ("last", last_metrics)):
        for k in ("logit_l2", "kl_ref_cand", "gumbel_margin"):
            t = m[k].float()
            summary[tag][f"{k}_p50"] = t.quantile(0.5).item()
            summary[tag][f"{k}_p99"] = t.quantile(0.99).item()
            summary[tag][f"{k}_max"] = t.max().item()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--weight-bits", type=int, default=16)
    ap.add_argument("--activation-bits", type=int, default=16,
                    help="Effective bits for activation per-token quant. Set to 0 to skip activation quantization entirely (weights-only PTQ).")
    ap.add_argument("--true-int-matmul", action="store_true",
                    help="Use int32@int32->int64 matmul instead of dequant-then-fp32. CPU fallback on CUDA, slow.")
    ap.add_argument("--quant-scheme", default="symmetric",
                    choices=["symmetric", "asymmetric", "per_group_sym"],
                    help="Quantization scheme: symmetric (per-row absmax), asymmetric (per-row min/max with zero-point), or per_group_sym (per-group along in_features).")
    ap.add_argument("--group-size", type=int, default=128,
                    help="Group size for per_group_sym scheme")
    ap.add_argument("--matmul-dtype", default="auto",
                    choices=["auto", "fp32", "bf16", "fp16"],
                    help="Matmul dtype inside IntLinear._float_path. 'auto' (default) keeps fp32 matmul; bf16/fp16 cast dequant operands before matmul so reduction matches reference.")
    ap.add_argument("--no-quant", action="store_true",
                    help="Skip int quantization entirely; replace nn.Linear with a no-quant Linear that runs at matmul-dtype. Used to isolate the bf16-vs-fp32 matmul question.")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-lm-head", action="store_true",
                    help="Leave lm_head as float (only quantize transformer blocks)")
    ap.add_argument("--skip-substring", action="append", default=[],
                    help="Skip Linear modules whose qualified name contains this substring (repeatable)")
    ap.add_argument("--smoothquant-alpha", type=float, default=None,
                    help="If set (e.g. 0.5), run SmoothQuant calibration with this alpha before quantizing.")
    ap.add_argument("--smoothquant-n-calib", type=int, default=16,
                    help="Number of prompts to use for SmoothQuant calibration.")
    ap.add_argument("--act-clip-quantile", type=float, default=None,
                    help="If set (e.g. 0.999), clip per-token activations to this quantile before quant scale is computed.")
    ap.add_argument("--cached-bf16", action="store_true",
                    help="Cache dequantized weight as nn.Parameter at compute_dtype (same shape/stride as original Linear.weight). Guarantees identical cuBLAS kernel dispatch with the reference nn.Linear. Activations still get per-token quant.")
    ap.add_argument("--int-nonmatmul", action="store_true",
                    help="Replace RMSNorm/softmax/SiLU/attention matmuls with int approximations (Phase 1+).")
    ap.add_argument("--rmsnorm-bits", type=int, default=24)
    ap.add_argument("--rmsnorm-nr-iter", type=int, default=2)
    ap.add_argument("--softmax-lut-size", type=int, default=1024)
    ap.add_argument("--softmax-x-min", type=float, default=-16.0)
    ap.add_argument("--silu-lut-size", type=int, default=4096)
    ap.add_argument("--attn-matmul-bits", type=int, default=24)
    ap.add_argument("--no-int-rmsnorm", action="store_true",
                    help="Ablation: skip RMSNorm replacement.")
    ap.add_argument("--no-int-softmax", action="store_true",
                    help="Ablation: skip softmax replacement.")
    ap.add_argument("--no-int-silu", action="store_true",
                    help="Ablation: skip SiLU replacement.")
    ap.add_argument("--no-int-attn-matmul", action="store_true",
                    help="Ablation: skip attention Q@K.T / P@V replacement.")
    ap.add_argument("--true-int-nonmatmul", action="store_true",
                    help="Literal-int execution for non-matmul ops (Phase 4 validation).")
    ap.add_argument("--int-embedding", action="store_true",
                    help="Quantize the nn.Embedding table per-row symmetric to embedding_bits.")
    ap.add_argument("--embedding-bits", type=int, default=24,
                    help="Bit width for embedding table quantization (default 24).")
    ap.add_argument("--cpu-patch", action="store_true",
                    help="Load model on CPU, deepcopy and patch on CPU, then move both to GPU. Useful when fp32 8B + deepcopy + per-layer fp64 quant transient blows past 80 GB on H100. Slower per-layer quant but avoids OOM.")
    ap.add_argument("--load-trained", default=None,
                    help="Path to a .pt produced by train_nonmatmul.save_trained_deltas (keys: rmsnorm_gamma, linear_bias). Overrides matching IntRMSNorm.weight and IntLinear.bias values after patching.")
    args = ap.parse_args()

    cfg = Config(
        model=args.model,
        dtype=args.dtype,
        n_prompts=args.n_prompts,
        max_len=args.max_len,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        true_int_matmul=args.true_int_matmul,
        quant_scheme=args.quant_scheme,
        group_size=args.group_size,
        matmul_dtype=args.matmul_dtype,
        no_quant=args.no_quant,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        out=args.out,
        include_lm_head=not args.skip_lm_head,
        skip_substrings=list(args.skip_substring),
        smoothquant_alpha=args.smoothquant_alpha,
        smoothquant_n_calib=args.smoothquant_n_calib,
        act_clip_quantile=args.act_clip_quantile,
        cached_bf16=args.cached_bf16,
        int_nonmatmul=args.int_nonmatmul,
        rmsnorm_bits=args.rmsnorm_bits,
        rmsnorm_nr_iter=args.rmsnorm_nr_iter,
        softmax_lut_size=args.softmax_lut_size,
        softmax_x_min=args.softmax_x_min,
        silu_lut_size=args.silu_lut_size,
        attn_matmul_bits=args.attn_matmul_bits,
        no_int_rmsnorm=args.no_int_rmsnorm,
        no_int_softmax=args.no_int_softmax,
        no_int_silu=args.no_int_silu,
        no_int_attn_matmul=args.no_int_attn_matmul,
        true_int_nonmatmul=args.true_int_nonmatmul,
        int_embedding=args.int_embedding,
        embedding_bits=args.embedding_bits,
    )
    device = pick_device(cfg.device)
    dtype = DTYPE_MAP[cfg.dtype]
    print(f"loading {cfg.model} ({cfg.dtype}) on {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    # cpu_patch path: load on CPU, deepcopy + patch on CPU, then move both to
    # GPU. Avoids the ~64 GB resident footprint of fp32-8B-ref + fp32-8B-deepcopy
    # + per-layer fp64 quant transient that OOMs the 80 GB H100.
    cpu_patch = getattr(args, "cpu_patch", False)
    if cpu_patch:
        print(f"[cpu-patch] loading {cfg.model} ({cfg.dtype}) on cpu first")
        ref = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).eval()
        int_model = copy.deepcopy(ref)
    else:
        ref = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(device).eval()
        int_model = copy.deepcopy(ref)
    # Resolve matmul_dtype string to a torch dtype (or None for "auto")
    matmul_dtype_torch = None
    if cfg.matmul_dtype != "auto":
        matmul_dtype_torch = DTYPE_MAP[cfg.matmul_dtype]
    if cfg.no_quant:
        # No-quant ablation: replace every nn.Linear with a wrapper that runs the
        # matmul at matmul_dtype (or fp32 if auto), no rounding. Isolates the
        # pure bf16-vs-fp32 reduction-order issue from quantization noise.
        replaced = _patch_model_noquant(
            int_model,
            matmul_dtype=matmul_dtype_torch,
            skip_substrings=tuple(cfg.skip_substrings),
            include_lm_head=cfg.include_lm_head,
        )
        print(
            f"replaced {len(replaced)} Linear modules with NoQuantLinear "
            f"(matmul_dtype={cfg.matmul_dtype})"
        )
    else:
        smooth_scales = None
        if cfg.smoothquant_alpha is not None:
            # Run a calibration pass on the unmodified deep-copied model
            # before patching. Use the same prompt source as the eval set,
            # but the FIRST `smoothquant_n_calib` prompts only.
            calib_prompts = load_prompts(cfg, tokenizer)[: cfg.smoothquant_n_calib]
            calib_inputs = [torch.tensor(ids, dtype=torch.long) for ids in calib_prompts]
            print(f"SmoothQuant calibration: alpha={cfg.smoothquant_alpha}, n={len(calib_inputs)} prompts")
            smooth_scales = calibrate_smooth_scales(
                int_model,
                calibration_inputs=calib_inputs,
                device=device,
                alpha=cfg.smoothquant_alpha,
                skip_names=tuple(cfg.skip_substrings),
                include_lm_head=cfg.include_lm_head,
            )
            print(f"SmoothQuant: computed {len(smooth_scales)} per-channel rescalers")
        replaced = patch_model_int_cast(
            int_model,
            weight_bits=cfg.weight_bits,
            activation_bits=cfg.activation_bits,
            skip_names=tuple(cfg.skip_substrings),
            include_lm_head=cfg.include_lm_head,
            quant_scheme=cfg.quant_scheme,
            group_size=cfg.group_size,
            matmul_dtype=matmul_dtype_torch,
            smooth_scales=smooth_scales,
            act_clip_quantile=cfg.act_clip_quantile,
            cached_bf16=cfg.cached_bf16,
        )
        if cfg.true_int_matmul:
            set_true_int_matmul(int_model, True)
        print(
            f"replaced {len(replaced)} Linear modules with IntLinear "
            f"(w_bits={cfg.weight_bits}, a_bits={cfg.activation_bits}, "
            f"true_int={cfg.true_int_matmul}, matmul_dtype={cfg.matmul_dtype}, "
            f"smoothquant={'on' if smooth_scales is not None else 'off'}, "
            f"act_clip_q={cfg.act_clip_quantile}, cached_bf16={cfg.cached_bf16})"
        )

    # Non-matmul int op replacement (runs on top of either the IntLinear patching
    # or the no-quant path).
    if cfg.int_nonmatmul:
        ops_cfg = IntOpsConfig(
            rmsnorm_bits=cfg.rmsnorm_bits,
            rmsnorm_nr_iter=cfg.rmsnorm_nr_iter,
            softmax_lut_size=cfg.softmax_lut_size,
            softmax_x_min=cfg.softmax_x_min,
            silu_lut_size=cfg.silu_lut_size,
            attn_matmul_bits=cfg.attn_matmul_bits,
            replace_rmsnorm=not cfg.no_int_rmsnorm,
            replace_softmax=not cfg.no_int_softmax,
            replace_silu=not cfg.no_int_silu,
            replace_attn_matmul=not cfg.no_int_attn_matmul,
        )
        counts = patch_model_int_nonmatmul(int_model, ops_cfg)
        print(
            f"int-nonmatmul replacement counts: {counts} "
            f"(rmsnorm_bits={cfg.rmsnorm_bits} nr_iter={cfg.rmsnorm_nr_iter}, "
            f"softmax_lut={cfg.softmax_lut_size}, "
            f"silu_lut={cfg.silu_lut_size}, "
            f"attn_matmul_bits={cfg.attn_matmul_bits})"
        )
        if cfg.true_int_nonmatmul:
            set_true_int_path(int_model, True)
            print("set true-int-path=True for non-matmul ops")

    if cfg.int_embedding:
        emb_replaced = patch_model_int_embedding(int_model, bits=cfg.embedding_bits)
        print(
            f"replaced {len(emb_replaced)} nn.Embedding modules with IntEmbedding "
            f"(bits={cfg.embedding_bits})"
        )

    if cpu_patch:
        # Now that patching is done (transient fp64 tensors freed), move both
        # models to the target device. ref first, then int_model.
        print(f"[cpu-patch] moving ref + int_model to {device}")
        ref = ref.to(device)
        int_model = int_model.to(device)

    # Optionally load trained gamma + bias deltas (Phase 2 / Approach B) and
    # optional trained matmul weight shadows (Phase 2 / Approach C).
    if args.load_trained:
        from difr_expt.int_cast import IntLinear as _IL
        from difr_expt.int_cast import quantize_per_row, quantize_per_row_asym, quantize_per_row_groupsym
        from difr_expt.int_ops import IntRMSNorm as _IRMS
        from difr_expt.patch_hf_model import IntSiLUModule as _ISiLU
        from difr_expt.patch_hf_model import IntSoftmaxModule as _ISMax
        payload = torch.load(args.load_trained, weights_only=False, map_location=device)
        gammas = payload.get("rmsnorm_gamma", {})
        biases = payload.get("linear_bias", {})
        weights = payload.get("linear_weight_fp", {})
        sm_luts = payload.get("softmax_lut", {})
        silu_luts = payload.get("silu_lut", {})
        n_g = n_b = n_w = n_smlut = n_silulut = 0
        for name, m in int_model.named_modules():
            if isinstance(m, _IRMS) and name in gammas:
                with torch.no_grad():
                    m.weight.data.copy_(gammas[name].to(m.weight.device, m.weight.dtype))
                n_g += 1
            elif isinstance(m, _IL):
                if name in biases and m.bias is not None:
                    with torch.no_grad():
                        new_b = biases[name].to(m.bias.device, m.bias.dtype)
                        if isinstance(m.bias, nn.Parameter):
                            m.bias.data.copy_(new_b)
                        else:
                            m.bias.copy_(new_b)
                    n_b += 1
                if name in weights and m.weight_int is not None:
                    # Approach C: re-quantize the trained fp32 shadow into the
                    # frozen IntLinear's int+scale buffers. Schemes must match
                    # what patch_model_int_cast did at construction.
                    w_fp = weights[name].to(m.weight_int.device, torch.float32)
                    with torch.no_grad():
                        if m.quant_scheme == "symmetric":
                            w_int, w_scale = quantize_per_row(w_fp, bits=m.weight_bits)
                        elif m.quant_scheme == "asymmetric":
                            w_int, w_scale, w_zp = quantize_per_row_asym(w_fp, bits=m.weight_bits)
                            m.weight_zp = w_zp
                        elif m.quant_scheme == "per_group_sym":
                            w_int, w_scale = quantize_per_row_groupsym(
                                w_fp, bits=m.weight_bits, group_size=m.group_size
                            )
                        else:
                            raise ValueError(f"unknown quant_scheme={m.quant_scheme!r}")
                        m.weight_int.data.copy_(w_int)
                        m.weight_scale.data.copy_(w_scale)
                    n_w += 1
            elif isinstance(m, _ISMax) and name in sm_luts:
                v = sm_luts[name].to(device)
                m.lut = nn.Parameter(v, requires_grad=False)
                n_smlut += 1
            elif isinstance(m, _ISiLU) and name in silu_luts:
                v = silu_luts[name].to(device)
                m.lut = nn.Parameter(v, requires_grad=False)
                n_silulut += 1
        print(f"--load-trained: applied {n_g} gammas, {n_b} biases, "
              f"{n_w} matmul weight shadows, "
              f"{n_smlut} softmax LUTs, {n_silulut} silu LUTs from {args.load_trained}")

    prompts = load_prompts(cfg, tokenizer)
    print(f"loaded {len(prompts)} prompts")

    t0 = time.time()
    summary = compare(ref, int_model, prompts, device=device, temperature=cfg.temperature, seed=cfg.seed)
    summary["wallclock_s"] = time.time() - t0
    summary["config"] = asdict(cfg)
    summary["replaced_module_count"] = len(replaced)

    print("\n=== summary ===")
    print(f"  n_prompts={summary['n_prompts']}  n_positions={summary['n_positions']}")
    for tag in ("all", "last"):
        print(f"  [{tag}]")
        for k, v in summary[tag].items():
            print(f"    {k}: {v}")

    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
