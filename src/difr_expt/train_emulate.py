"""Train an int24 'proof model' to emulate a fp4/fp8 'production model'.

Strategic shift vs `train_nonmatmul.py`:

  - Old setup: teacher = fp32 reference; student = int24; goal = top-1 ≈ 100%
    against fp32. Trivial — both formats have enormous precision headroom.
  - New setup: teacher = fp4/fp8 quantized (`patch_model_low_precision`);
    student = int24; goal = student matches the teacher's noisy logits.
    This is the real DiFR pitch: prove int executes in ZK, trust int emulates
    the production fp4/fp8 model because we trained it to.

Trainable: same as Approach C+LUTs from `train_nonmatmul.py` — matmul weight
shadows + γ + biases + LUT entries.

Loss: KL(softmax(teacher/T) || softmax(student/T)) + aux * MSE(logits).
KL is the natural distillation loss when the teacher is a noisy distribution
(it focuses the student on getting the *distribution* right, not raw logit
magnitudes which fp4 quantizes coarsely).

Usage:
    python -m difr_expt.train_emulate \
        --model Qwen/Qwen2.5-0.5B \
        --prompts prompts.pt \
        --teacher-precision fp4_e2m1 \
        --lr 1e-5 --steps 500 --batch 2 \
        --out experiments/fp4-emulation/data/qwen25_fp4_run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from difr_expt.fp_quant import patch_model_low_precision
from difr_expt.int_cast import (
    IntLinear,
    patch_model_int_cast,
    patch_model_int_embedding,
)
from difr_expt.int_ops import IntRMSNorm
from difr_expt.patch_hf_model import (
    IntOpsConfig,
    IntSiLUModule,
    IntSoftmaxModule,
    patch_model_int_nonmatmul,
)
from difr_expt.metrics import (
    kl_div_ref_to_cand,
    logit_l2,
    post_gumbel_margin,
    top1_match,
    topk_overlap,
)


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass
class Config:
    model: str
    prompts: str
    out: str
    teacher_source: str = "fake_quant"  # fake_quant | published
    teacher_id: str | None = None  # required when teacher_source=published
    teacher_precision: str = "fp4_e2m1"  # fp4_e2m1 | fp8_e4m3 | fp8_e5m2 (fake_quant only)
    teacher_block_size: int = 32
    teacher_quantize_act: bool = True
    use_8bit_adamw: bool = False
    grad_checkpointing: bool = False
    lr: float = 1e-5
    lr_luts: float = 1e-3
    lr_gamma_bias: float = 1e-4
    steps: int = 500
    batch: int = 2
    eval_every: int = 100
    warmup: int = 20
    grad_clip: float = 1.0
    weight_bits: int = 24
    activation_bits: int = 24
    rmsnorm_bits: int = 24
    softmax_lut_size: int = 4096
    softmax_x_min: float = -16.0
    silu_lut_size: int = 4096
    attn_matmul_bits: int = 24
    dtype: str = "float32"
    seed: int = 42
    aux_weight: float = 1.0
    plateau_patience: int = 5
    eval_n_prompts: int = 20
    eval_max_positions: int = 50_000
    temperature: float = 1.0
    trainable_matmul_weights: bool = True
    trainable_luts: bool = True
    int_embedding: bool = False
    embedding_bits: int = 24
    int_lm_head: bool = False  # Match teacher convention (lm_head fp32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def promote_intlinear_biases(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, IntLinear) and m.bias is not None and not isinstance(m.bias, nn.Parameter):
            b = m.bias.detach().clone()
            if "bias" in m._buffers:
                del m._buffers["bias"]
            m.bias = nn.Parameter(b)
            n += 1
    return n


def promote_luts(model: nn.Module) -> tuple[int, int]:
    """Promote softmax/silu LUTs to nn.Parameter and co-locate them with the
    rest of the model. The init helpers (_build_exp_lut/_build_sigmoid_lut)
    create CPU tensors; without the .to(device) move below, the LUT param
    stays on CPU while everything else is on GPU. torch.optim.AdamW silently
    tolerates the split (CPU updates, slow); bitsandbytes' 8-bit AdamW does
    not — it requires same-device. So this fix is mandatory for 8B runs.
    """
    n_sm, n_silu = 0, 0
    device = next(model.parameters()).device
    for m in model.modules():
        if isinstance(m, IntSoftmaxModule) and m.lut is None:
            m.make_lut_trainable()
            m.lut.data = m.lut.data.to(device)
            n_sm += 1
        elif isinstance(m, IntSiLUModule) and m.lut is None:
            m.make_lut_trainable()
            m.lut.data = m.lut.data.to(device)
            n_silu += 1
    return n_sm, n_silu


def collect_trainable(
    model: nn.Module,
    include_weight_fp: bool = True,
    include_luts: bool = True,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Return param groups: {'matmul': ..., 'gamma_bias': ..., 'luts': ...}."""
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "matmul": [],
        "gamma_bias": [],
        "luts": [],
    }
    for name, m in model.named_modules():
        if isinstance(m, IntRMSNorm):
            groups["gamma_bias"].append((f"{name}.weight", m.weight))
        elif isinstance(m, IntLinear):
            if isinstance(m.bias, nn.Parameter):
                groups["gamma_bias"].append((f"{name}.bias", m.bias))
            if include_weight_fp and m.weight_fp is not None:
                groups["matmul"].append((f"{name}.weight_fp", m.weight_fp))
        elif include_luts and isinstance(m, (IntSiLUModule, IntSoftmaxModule)):
            if m.lut is not None:
                groups["luts"].append((f"{name}.lut", m.lut))
    return groups


def freeze_all_but(model: nn.Module, keep_ids: set[int]) -> None:
    for p in model.parameters():
        if id(p) not in keep_ids:
            p.requires_grad = False


def cosine_warmup(step: int, warmup: int, total: int, peak_lr: float, floor: float = 0.1) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (floor + (1.0 - floor) * cos)


def pad_collate(batch_ids: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(t.numel() for t in batch_ids)
    ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, t in enumerate(batch_ids):
        ids[i, : t.numel()] = t
        mask[i, : t.numel()] = True
    return ids, mask


@torch.inference_mode()
def evaluate(
    teacher: nn.Module,
    student: nn.Module,
    eval_prompts: list[torch.Tensor],
    device: str,
    max_positions: int,
    temperature: float,
    seed: int,
    ref_model: nn.Module | None = None,
) -> dict[str, float]:
    """Compute student-vs-teacher (the headline) metrics + optionally
    student-vs-ref and teacher-vs-ref (sanity rows).
    """
    pair_keys: list[tuple[str, nn.Module, nn.Module]] = [
        ("student_vs_teacher", teacher, student),
    ]
    if ref_model is not None:
        pair_keys.append(("student_vs_ref", ref_model, student))
        pair_keys.append(("teacher_vs_ref", ref_model, teacher))

    out: dict[str, float] = {}
    for tag, ref, cand in pair_keys:
        all_top1, all_top5, all_l2, all_kl, all_margin = [], [], [], [], []
        n_positions = 0
        rng = torch.Generator(device=device).manual_seed(seed)
        for ids in eval_prompts:
            if n_positions >= max_positions:
                break
            input_ids = ids.to(device).unsqueeze(0)
            r_logits = ref(input_ids).logits[0]
            c_logits = cand(input_ids).logits[0]
            v = min(r_logits.shape[-1], c_logits.shape[-1])
            r_logits = r_logits[..., :v]
            c_logits = c_logits[..., :v]
            t1 = top1_match(r_logits, c_logits)
            t5 = topk_overlap(r_logits, c_logits, k=5)
            l2 = logit_l2(r_logits, c_logits)
            kl = kl_div_ref_to_cand(r_logits, c_logits, temperature=temperature)
            u = torch.empty_like(r_logits, dtype=torch.float32)
            u.uniform_(1e-10, 1.0, generator=rng)
            gumbel = -torch.log(-torch.log(u))
            mg = post_gumbel_margin(r_logits, c_logits, gumbel, temperature=temperature)
            all_top1.append(t1); all_top5.append(t5); all_l2.append(l2); all_kl.append(kl); all_margin.append(mg)
            n_positions += r_logits.shape[0]
        cat = lambda xs: torch.cat(xs)
        out[f"{tag}/top1"] = cat(all_top1).float().mean().item()
        out[f"{tag}/top5"] = cat(all_top5).mean().item()
        out[f"{tag}/logit_l2_mean"] = cat(all_l2).mean().item()
        out[f"{tag}/logit_l2_p99"] = cat(all_l2).quantile(0.99).item()
        out[f"{tag}/kl_mean"] = cat(all_kl).mean().item()
        out[f"{tag}/kl_p99"] = cat(all_kl).quantile(0.99).item()
        out[f"{tag}/margin_mean"] = cat(all_margin).mean().item()
        out[f"{tag}/margin_p99"] = cat(all_margin).quantile(0.99).item()
        out[f"{tag}/n_positions"] = n_positions
    return out


def save_trained_deltas(model: nn.Module, path: Path) -> None:
    payload: dict[str, Any] = {
        "rmsnorm_gamma": {},
        "linear_bias": {},
        "linear_weight_fp": {},
        "softmax_lut": {},
        "silu_lut": {},
    }
    for name, m in model.named_modules():
        if isinstance(m, IntRMSNorm):
            payload["rmsnorm_gamma"][name] = m.weight.detach().cpu().clone()
        elif isinstance(m, IntLinear):
            if isinstance(m.bias, nn.Parameter):
                payload["linear_bias"][name] = m.bias.detach().cpu().clone()
            if m.weight_fp is not None:
                payload["linear_weight_fp"][name] = m.weight_fp.detach().cpu().clone()
        elif isinstance(m, IntSoftmaxModule) and m.lut is not None:
            payload["softmax_lut"][name] = m.lut.detach().cpu().clone()
        elif isinstance(m, IntSiLUModule) and m.lut is not None:
            payload["silu_lut"][name] = m.lut.detach().cpu().clone()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def build_models(
    model_name: str,
    teacher_source: str,
    teacher_id: str | None,
    teacher_precision: str,
    teacher_block_size: int,
    teacher_quantize_act: bool,
    dtype: torch.dtype,
    device: str,
    weight_bits: int,
    activation_bits: int,
    rmsnorm_bits: int,
    softmax_lut_size: int,
    softmax_x_min: float,
    silu_lut_size: int,
    attn_matmul_bits: int,
    trainable_matmul_weights: bool,
    int_embedding: bool,
    embedding_bits: int,
    int_lm_head: bool = False,
    keep_fp32_ref: bool = True,
    grad_checkpointing: bool = False,
) -> tuple[nn.Module, nn.Module, nn.Module | None]:
    """Build (teacher, student, ref).

    teacher_source="fake_quant":  teacher = LowPrecisionLinear-patched deepcopy of base.
    teacher_source="published":   teacher = HF checkpoint loaded from teacher_id
                                  (compressed_tensors handles fp8, fp_quant handles NVFP4).
    student = int24 model (matmul + non-matmul + optional embedding patches).
    ref    = unmodified fp32 reference (None if `keep_fp32_ref=False`).
    """
    # Load the fp32 (or bf16) base model once.
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    for p in base.parameters():
        p.requires_grad = False

    if teacher_source == "published":
        assert teacher_id is not None, "teacher_id is required when teacher_source=published"
        # HF transformers auto-detects the quantization config from the model
        # card; compressed_tensors handles fp8, fp_quant handles NVFP4. Compute
        # happens in `dtype` after on-the-fly dequant.
        teacher = AutoModelForCausalLM.from_pretrained(teacher_id, dtype=dtype).to(device).eval()
        print(f"  teacher: loaded published checkpoint {teacher_id}")
    elif teacher_source == "fake_quant":
        teacher = copy.deepcopy(base)
        n_lp = patch_model_low_precision(
            teacher,
            precision=teacher_precision,
            block_size=teacher_block_size,
            include_lm_head=False,
            quantize_act=teacher_quantize_act,
        )
        print(f"  teacher: fake-quant replaced {len(n_lp)} Linears "
              f"(precision={teacher_precision})")
    else:
        raise ValueError(
            f"unknown teacher_source: {teacher_source!r} (expected fake_quant|published)"
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = copy.deepcopy(base)
    # Match teacher convention: published fp8/fp4 checkpoints leave lm_head in
    # bf16/fp32. Mirroring that here keeps the comparison apples-to-apples and
    # avoids ~2.5GB of fp32 shadows + ~10GB of forward transients on 8B (the
    # lm_head matmul was the OOM trigger).
    replaced = patch_model_int_cast(
        student,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        trainable=trainable_matmul_weights,
        include_lm_head=int_lm_head,
    )
    print(f"  student: replaced {len(replaced)} Linears with IntLinear "
          f"(trainable={trainable_matmul_weights}, include_lm_head={int_lm_head})")
    ops_cfg = IntOpsConfig(
        rmsnorm_bits=rmsnorm_bits,
        softmax_lut_size=softmax_lut_size,
        softmax_x_min=softmax_x_min,
        silu_lut_size=silu_lut_size,
        attn_matmul_bits=attn_matmul_bits,
    )
    counts = patch_model_int_nonmatmul(student, ops_cfg)
    print(f"  student: int non-matmul counts: {counts}")
    if int_embedding:
        emb = patch_model_int_embedding(student, bits=embedding_bits)
        print(f"  student: replaced {len(emb)} embeddings with IntEmbedding")
    if grad_checkpointing:
        # HF method; enables recomputation of intermediate activations during
        # backward instead of holding them. Cuts activation memory ~5x at the
        # cost of one extra forward per step. Important for 8B on H100 80GB.
        if hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable()
            print("  student: gradient checkpointing enabled")
        else:
            print("  student: gradient_checkpointing_enable() not available — skipping")

    ref = base if keep_fp32_ref else None
    if ref is None:
        del base
    return teacher, student, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help=".pt from cache_prompts.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher-source", default="fake_quant",
                    choices=["fake_quant", "published"],
                    help="fake_quant: patch base model with LowPrecisionLinear. "
                         "published: load --teacher-id via HF (needs compressed_tensors for fp8, "
                         "fp_quant for NVFP4).")
    ap.add_argument("--teacher-id", default=None,
                    help="HF model id for published teacher (e.g. RedHatAI/Qwen2.5-0.5B-FP8-dynamic). "
                         "Required when --teacher-source=published.")
    ap.add_argument("--teacher-precision", default="fp4_e2m1",
                    choices=["fp4_e2m1", "fp8_e4m3", "fp8_e5m2"],
                    help="Only used when --teacher-source=fake_quant.")
    ap.add_argument("--teacher-block-size", type=int, default=32)
    ap.add_argument("--teacher-no-quantize-act", action="store_true",
                    help="Skip teacher activation quant (weight-only fp4/fp8); fake_quant only.")
    ap.add_argument("--use-8bit-adamw", action="store_true",
                    help="Use bitsandbytes 8-bit AdamW instead of torch AdamW. "
                         "Needed to fit 8B models on H100 80GB.")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="Enable gradient checkpointing on the student. "
                         "Needed for 8B activation memory.")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="LR for matmul weight shadows.")
    ap.add_argument("--lr-luts", type=float, default=1e-3)
    ap.add_argument("--lr-gamma-bias", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-bits", type=int, default=24)
    ap.add_argument("--activation-bits", type=int, default=24)
    ap.add_argument("--rmsnorm-bits", type=int, default=24)
    ap.add_argument("--softmax-lut-size", type=int, default=4096)
    ap.add_argument("--softmax-x-min", type=float, default=-16.0)
    ap.add_argument("--silu-lut-size", type=int, default=4096)
    ap.add_argument("--attn-matmul-bits", type=int, default=24)
    ap.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--plateau-patience", type=int, default=5)
    ap.add_argument("--eval-n-prompts", type=int, default=20)
    ap.add_argument("--eval-max-positions", type=int, default=50_000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-trainable-matmul-weights", action="store_true")
    ap.add_argument("--no-trainable-luts", action="store_true")
    ap.add_argument("--int-embedding", action="store_true")
    ap.add_argument("--embedding-bits", type=int, default=24)
    ap.add_argument("--int-lm-head", action="store_true",
                    help="Apply IntLinear to lm_head too. Default off — matches "
                         "published-teacher convention (fp32 lm_head) and avoids ~10GB "
                         "of forward transients on 8B.")
    ap.add_argument("--no-fp32-ref", action="store_true",
                    help="Don't keep an fp32 ref in memory (saves RAM on CPU runs).")
    args = ap.parse_args()

    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    cfg = Config(
        model=args.model,
        prompts=args.prompts,
        out=args.out,
        teacher_source=args.teacher_source,
        teacher_id=args.teacher_id,
        teacher_precision=args.teacher_precision,
        teacher_block_size=args.teacher_block_size,
        teacher_quantize_act=not args.teacher_no_quantize_act,
        use_8bit_adamw=args.use_8bit_adamw,
        grad_checkpointing=args.grad_checkpointing,
        lr=args.lr,
        lr_luts=args.lr_luts,
        lr_gamma_bias=args.lr_gamma_bias,
        steps=args.steps,
        batch=args.batch,
        eval_every=args.eval_every,
        warmup=args.warmup,
        grad_clip=args.grad_clip,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        rmsnorm_bits=args.rmsnorm_bits,
        softmax_lut_size=args.softmax_lut_size,
        softmax_x_min=args.softmax_x_min,
        silu_lut_size=args.silu_lut_size,
        attn_matmul_bits=args.attn_matmul_bits,
        dtype=args.dtype,
        seed=args.seed,
        aux_weight=args.aux_weight,
        plateau_patience=args.plateau_patience,
        eval_n_prompts=args.eval_n_prompts,
        eval_max_positions=args.eval_max_positions,
        temperature=args.temperature,
        trainable_matmul_weights=not args.no_trainable_matmul_weights,
        trainable_luts=not args.no_trainable_luts,
        int_embedding=args.int_embedding,
        embedding_bits=args.embedding_bits,
        int_lm_head=args.int_lm_head,
    )
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[cfg.dtype]
    print(f"[{time.strftime('%H:%M:%S')}] loading {cfg.model} ({dtype}) on {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    if cfg.teacher_source == "published" and cfg.teacher_id is None:
        raise SystemExit("--teacher-id is required when --teacher-source=published")
    teacher, student, ref = build_models(
        model_name=cfg.model,
        teacher_source=cfg.teacher_source,
        teacher_id=cfg.teacher_id,
        teacher_precision=cfg.teacher_precision,
        teacher_block_size=cfg.teacher_block_size,
        teacher_quantize_act=cfg.teacher_quantize_act,
        dtype=dtype,
        device=device,
        weight_bits=cfg.weight_bits,
        activation_bits=cfg.activation_bits,
        rmsnorm_bits=cfg.rmsnorm_bits,
        softmax_lut_size=cfg.softmax_lut_size,
        softmax_x_min=cfg.softmax_x_min,
        silu_lut_size=cfg.silu_lut_size,
        attn_matmul_bits=cfg.attn_matmul_bits,
        trainable_matmul_weights=cfg.trainable_matmul_weights,
        int_embedding=cfg.int_embedding,
        embedding_bits=cfg.embedding_bits,
        int_lm_head=cfg.int_lm_head,
        keep_fp32_ref=not args.no_fp32_ref,
        grad_checkpointing=cfg.grad_checkpointing,
    )

    n_biases = promote_intlinear_biases(student)
    print(f"  promoted {n_biases} biases to nn.Parameter")
    if cfg.trainable_luts:
        n_sm, n_silu = promote_luts(student)
        print(f"  promoted {n_sm} softmax LUTs + {n_silu} silu LUTs to nn.Parameter")

    groups = collect_trainable(
        student,
        include_weight_fp=cfg.trainable_matmul_weights,
        include_luts=cfg.trainable_luts,
    )
    keep_ids: set[int] = set()
    for g in groups.values():
        for _, p in g:
            keep_ids.add(id(p))
    freeze_all_but(student, keep_ids)
    counts = {k: (len(v), sum(p.numel() for _, p in v)) for k, v in groups.items()}
    print(f"  trainable groups: {counts}")

    param_groups = []
    if groups["matmul"]:
        param_groups.append({"params": [p for _, p in groups["matmul"]], "lr": cfg.lr})
    if groups["gamma_bias"]:
        param_groups.append({"params": [p for _, p in groups["gamma_bias"]], "lr": cfg.lr_gamma_bias})
    if groups["luts"]:
        param_groups.append({"params": [p for _, p in groups["luts"]], "lr": cfg.lr_luts})
    if cfg.use_8bit_adamw:
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise SystemExit(
                "--use-8bit-adamw requires bitsandbytes (pip install bitsandbytes)"
            ) from e
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.0)
        print("  optimizer: bnb.AdamW8bit")
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        print("  optimizer: torch.AdamW (fp32 state)")

    prompts: list[torch.Tensor] = torch.load(cfg.prompts, weights_only=False)
    print(f"  loaded {len(prompts)} prompts from {cfg.prompts}")
    rng = random.Random(cfg.seed)
    if len(prompts) <= cfg.eval_n_prompts:
        train_prompts = prompts
        eval_prompts = prompts
    else:
        train_prompts = prompts[: -cfg.eval_n_prompts]
        eval_prompts = prompts[-cfg.eval_n_prompts:]
    print(f"  train: {len(train_prompts)}  eval: {len(eval_prompts)}")

    metrics_file = open(out_dir / "metrics.jsonl", "a")
    t_start = time.time()

    def log(record: dict[str, Any]) -> None:
        record["wall_s"] = time.time() - t_start
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    print(f"[{time.strftime('%H:%M:%S')}] step 0 eval")
    pre = evaluate(
        teacher, student, eval_prompts, device,
        max_positions=cfg.eval_max_positions,
        temperature=cfg.temperature, seed=cfg.seed,
        ref_model=ref,
    )
    pre["step"] = 0; pre["lr"] = 0.0; pre["loss"] = float("nan")
    log(pre)
    print(f"  pre-train: student_vs_teacher top1={pre['student_vs_teacher/top1']:.4f}  "
          f"kl_p99={pre['student_vs_teacher/kl_p99']:.4e}  "
          f"margin_p99={pre['student_vs_teacher/margin_p99']:.4e}")
    if ref is not None:
        print(f"  pre-train: student_vs_ref top1={pre['student_vs_ref/top1']:.4f}  "
              f"teacher_vs_ref top1={pre['teacher_vs_ref/top1']:.4f}")

    save_trained_deltas(student, out_dir / "pretrain.pt")

    student.train()
    best_top1 = pre["student_vs_teacher/top1"]
    plateau = 0
    best_step = 0
    save_trained_deltas(student, out_dir / "best.pt")

    for step in range(1, cfg.steps + 1):
        batch_ids = [train_prompts[rng.randrange(len(train_prompts))] for _ in range(cfg.batch)]
        input_ids, mask = pad_collate(batch_ids, pad_id)
        input_ids = input_ids.to(device); mask = mask.to(device)

        with torch.inference_mode():
            teacher_logits = teacher(input_ids).logits
        student_logits = student(input_ids).logits
        v = min(teacher_logits.shape[-1], student_logits.shape[-1])
        s = student_logits[..., :v].float()
        t = teacher_logits[..., :v].float()

        # KL(softmax(t/T) || softmax(s/T)) — distillation loss.
        T = cfg.temperature
        log_p_t = F.log_softmax(t / T, dim=-1)
        log_p_s = F.log_softmax(s / T, dim=-1)
        p_t = log_p_t.exp()
        kl_per_pos = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
        valid = mask.float()
        kl_loss = (kl_per_pos * valid).sum() / valid.sum().clamp_min(1.0)

        # Aux: MSE between logits, normalized by teacher norm. Stabilizes when
        # KL gradients are small (e.g. low-entropy distributions).
        diff = (s - t)
        mse_per_pos = diff.pow(2).mean(dim=-1)
        mse_loss = (mse_per_pos * valid).sum() / valid.sum().clamp_min(1.0) / v

        loss = kl_loss + cfg.aux_weight * mse_loss

        # Cosine-warmup LR — scale each param group by its base lr.
        progress = cosine_warmup(step, cfg.warmup, cfg.steps, 1.0)
        for pg in optimizer.param_groups:
            base_lr = pg.get("initial_lr")
            if base_lr is None:
                # First step — capture the param group's lr as its initial.
                pg["initial_lr"] = pg["lr"]
                base_lr = pg["lr"]
            pg["lr"] = base_lr * progress

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups.values() for _, p in g],
                cfg.grad_clip,
            )
        optimizer.step()

        if step % 20 == 0 or step == 1:
            print(f"  step {step:5d}  loss={loss.item():.4e}  kl={kl_loss.item():.4e}  "
                  f"mse={mse_loss.item():.4e}  scale={progress:.3f}")

        if step % cfg.eval_every == 0:
            student.eval()
            ev = evaluate(
                teacher, student, eval_prompts, device,
                max_positions=cfg.eval_max_positions,
                temperature=cfg.temperature, seed=cfg.seed,
                ref_model=ref,
            )
            ev["step"] = step; ev["lr"] = optimizer.param_groups[0]["lr"]; ev["loss"] = loss.item()
            log(ev)
            print(
                f"  [eval @ {step}] student_vs_teacher: "
                f"top1={ev['student_vs_teacher/top1']:.4f}  "
                f"kl_p99={ev['student_vs_teacher/kl_p99']:.4e}  "
                f"margin_p99={ev['student_vs_teacher/margin_p99']:.4e}"
            )
            student.train()
            if ev["student_vs_teacher/top1"] > best_top1 + 1e-5:
                best_top1 = ev["student_vs_teacher/top1"]
                plateau = 0
                save_trained_deltas(student, out_dir / "best.pt")
                best_step = step
            else:
                plateau += 1
            if plateau >= cfg.plateau_patience and step >= cfg.warmup + 3 * cfg.eval_every:
                print(f"  plateau ({plateau} no-improvements); stopping at step {step}")
                break

    save_trained_deltas(student, out_dir / "final.pt")
    # Final eval with the best checkpoint reloaded.
    print(f"[{time.strftime('%H:%M:%S')}] final eval with best.pt")
    # On 8B, the optimizer state + student + grads occupy ~75 GB; loading best.pt
    # (30 GB) directly to GPU on top of that OOMs the 80 GB H100. Free optimizer
    # first, then load checkpoint to CPU and transfer per-tensor below.
    del optimizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload = torch.load(out_dir / "best.pt", weights_only=False, map_location="cpu")
    gammas = payload.get("rmsnorm_gamma", {})
    biases = payload.get("linear_bias", {})
    weights = payload.get("linear_weight_fp", {})
    sm_luts = payload.get("softmax_lut", {})
    silu_luts = payload.get("silu_lut", {})
    for name, m in student.named_modules():
        if isinstance(m, IntRMSNorm) and name in gammas:
            with torch.no_grad():
                m.weight.data.copy_(gammas[name].to(m.weight.device, m.weight.dtype))
        elif isinstance(m, IntLinear):
            if name in biases and m.bias is not None and isinstance(m.bias, nn.Parameter):
                with torch.no_grad():
                    m.bias.data.copy_(biases[name].to(m.bias.device, m.bias.dtype))
            if name in weights and m.weight_fp is not None:
                with torch.no_grad():
                    m.weight_fp.data.copy_(weights[name].to(m.weight_fp.device, m.weight_fp.dtype))
        elif isinstance(m, IntSoftmaxModule) and name in sm_luts:
            with torch.no_grad():
                m.lut.data.copy_(sm_luts[name].to(m.lut.device, m.lut.dtype))
        elif isinstance(m, IntSiLUModule) and name in silu_luts:
            with torch.no_grad():
                m.lut.data.copy_(silu_luts[name].to(m.lut.device, m.lut.dtype))

    student.eval()
    post = evaluate(
        teacher, student, eval_prompts, device,
        max_positions=cfg.eval_max_positions,
        temperature=cfg.temperature, seed=cfg.seed,
        ref_model=ref,
    )
    post["step"] = -1; post["lr"] = 0.0; post["loss"] = float("nan")
    post["best_step"] = best_step
    log(post)

    summary = {"pre": pre, "post": post, "best_step": best_step}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"done in {time.time() - t_start:.0f}s; "
          f"pre: top1={pre['student_vs_teacher/top1']:.4f}  "
          f"best: top1={best_top1:.4f}  "
          f"post (with best ckpt): top1={post['student_vs_teacher/top1']:.4f}")

    metrics_file.close()


if __name__ == "__main__":
    main()
