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
    logit_cosine,
    logit_l1,
    logit_l2,
    logit_max_abs_err,
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
    init_from_teacher: bool = False  # Cast teacher's quantized weights → student init
    matmul_loss_weight: float = 0.0  # Σ per-matmul L2; 0 disables
    matmul_loss_norm: str = "l2"  # l1 | l2 — norm for per-matmul training loss
    logit_loss_weight: float = 1.0  # final-logit (KL + aux MSE); 0 disables
    trainable_gamma_bias: bool = True
    trainable_scales: bool = False  # Promote per-row matmul scales to params


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


class MatmulCapture:
    """Captures outputs of paired Linear modules via forward hooks. Used for
    per-matmul training loss and per-matmul L1/L2 eval metrics.
    """
    def __init__(self, detach: bool = False) -> None:
        self.detach = detach
        self.outputs: dict[str, torch.Tensor] = {}

    def hook_for(self, name: str):
        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            o = output if isinstance(output, torch.Tensor) else output[0]
            # detach+clone breaks inference_mode (so teacher captures from
            # @torch.inference_mode() are usable as plain tensors) and the
            # autograd graph (so teacher outputs aren't a backward target).
            self.outputs[name] = o.detach().clone() if self.detach else o
        return _hook

    def clear(self) -> None:
        self.outputs.clear()


def register_matmul_hooks(
    student: nn.Module,
    teacher: nn.Module,
) -> tuple[MatmulCapture, MatmulCapture, list[str], list[Any]]:
    """Hook every IntLinear in `student` and the same-named nn.Linear (incl.
    CompressedLinear subclass) in `teacher`. Returns
    (student_capture, teacher_capture, paired_names, handles).
    """
    s_cap = MatmulCapture(detach=False)
    t_cap = MatmulCapture(detach=True)
    handles: list[Any] = []

    student_names: set[str] = set()
    for name, m in student.named_modules():
        if isinstance(m, IntLinear):
            handles.append(m.register_forward_hook(s_cap.hook_for(name)))
            student_names.add(name)

    teacher_modules = dict(teacher.named_modules())
    paired: list[str] = []
    for name in sorted(student_names):
        tm = teacher_modules.get(name)
        if isinstance(tm, nn.Linear):
            handles.append(tm.register_forward_hook(t_cap.hook_for(name)))
            paired.append(name)
    return s_cap, t_cap, paired, handles


def per_matmul_loss(
    s_cap: MatmulCapture,
    t_cap: MatmulCapture,
    paired: list[str],
    norm: str = "l2",
) -> torch.Tensor:
    """Σ over paired layers of layer-mean (L1 or L2) divergence."""
    total: torch.Tensor | None = None
    for name in paired:
        s_out = s_cap.outputs.get(name)
        t_out = t_cap.outputs.get(name)
        if s_out is None or t_out is None or s_out.shape != t_out.shape:
            continue
        diff = s_out.float() - t_out.float()
        layer = diff.pow(2).mean() if norm == "l2" else diff.abs().mean()
        total = layer if total is None else (total + layer)
    if total is None:
        # No paired layers populated — should never happen if hooks attached
        # and the forward ran. Return a 0 scalar on the student's device.
        any_s = next(iter(s_cap.outputs.values()), None)
        dev = any_s.device if any_s is not None else "cpu"
        return torch.zeros((), device=dev)
    return total


def per_matmul_metrics(
    s_cap: MatmulCapture,
    t_cap: MatmulCapture,
    paired: list[str],
) -> dict[str, float]:
    """Per-layer RMS (sqrt of mean square) and L1 (mean abs) divergence + aggregates."""
    out: dict[str, float] = {}
    l2s, l1s = [], []
    for name in paired:
        s_out = s_cap.outputs.get(name)
        t_out = t_cap.outputs.get(name)
        if s_out is None or t_out is None or s_out.shape != t_out.shape:
            continue
        diff = s_out.float() - t_out.float()
        rms = diff.pow(2).mean().sqrt().item()
        mae = diff.abs().mean().item()
        out[f"matmul/{name}/rms"] = rms
        out[f"matmul/{name}/mae"] = mae
        l2s.append(rms); l1s.append(mae)
    if l2s:
        out["matmul/aggregate/mean_rms"] = sum(l2s) / len(l2s)
        out["matmul/aggregate/max_rms"] = max(l2s)
        out["matmul/aggregate/sum_rms"] = sum(l2s)
        out["matmul/aggregate/mean_mae"] = sum(l1s) / len(l1s)
        out["matmul/aggregate/max_mae"] = max(l1s)
        out["matmul/aggregate/sum_mae"] = sum(l1s)
    return out


@torch.inference_mode()
def per_matmul_eval(
    teacher: nn.Module,
    student: nn.Module,
    eval_prompts: list[torch.Tensor],
    device: str,
    s_cap: MatmulCapture,
    t_cap: MatmulCapture,
    paired: list[str],
    n_batches: int = 4,
) -> dict[str, float]:
    """Forward one or more eval prompts with hooks active, accumulate per-layer
    divergences. Run after `evaluate()` so the regular logit metrics aren't
    affected by these extra captures.
    """
    # Per-layer position-averaged L1 (Σ_j |diff|) and L2 (sqrt Σ_j diff²) of the
    # difference vector along the last dim. Averaged over positions and batches.
    accum_l2: dict[str, list[float]] = {}
    accum_l1: dict[str, list[float]] = {}
    accum_max: dict[str, list[float]] = {}
    for ids in eval_prompts[:n_batches]:
        s_cap.clear(); t_cap.clear()
        input_ids = ids.to(device).unsqueeze(0)
        teacher(input_ids)
        student(input_ids)
        for name in paired:
            s_out = s_cap.outputs.get(name)
            t_out = t_cap.outputs.get(name)
            if s_out is None or t_out is None or s_out.shape != t_out.shape:
                continue
            diff = s_out.float() - t_out.float()  # [..., out_features]
            # Per-position L2 norm averaged over positions.
            accum_l2.setdefault(name, []).append(diff.norm(dim=-1).mean().item())
            # Per-position L1 norm averaged over positions.
            accum_l1.setdefault(name, []).append(diff.abs().sum(dim=-1).mean().item())
            # Per-position max abs element error averaged over positions.
            accum_max.setdefault(name, []).append(diff.abs().amax(dim=-1).mean().item())
    s_cap.clear(); t_cap.clear()
    out: dict[str, float] = {}
    l2s, l1s, maxes = [], [], []
    for name in paired:
        if name not in accum_l2:
            continue
        l2 = sum(accum_l2[name]) / len(accum_l2[name])
        l1 = sum(accum_l1[name]) / len(accum_l1[name])
        mx = sum(accum_max[name]) / len(accum_max[name])
        out[f"matmul/{name}/l2"] = l2
        out[f"matmul/{name}/l1"] = l1
        out[f"matmul/{name}/max"] = mx
        l2s.append(l2); l1s.append(l1); maxes.append(mx)
    if l2s:
        out["matmul/aggregate/mean_l2"] = sum(l2s) / len(l2s)
        out["matmul/aggregate/max_l2"] = max(l2s)
        out["matmul/aggregate/sum_l2"] = sum(l2s)
        out["matmul/aggregate/mean_l1"] = sum(l1s) / len(l1s)
        out["matmul/aggregate/max_l1"] = max(l1s)
        out["matmul/aggregate/sum_l1"] = sum(l1s)
        out["matmul/aggregate/mean_max"] = sum(maxes) / len(maxes)
        out["matmul/aggregate/max_max"] = max(maxes)
    return out


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
        all_top1, all_top5, all_top10 = [], [], []
        all_l1, all_l2, all_maxerr, all_cos = [], [], [], []
        all_kl, all_margin = [], []
        n_positions = 0
        v_last = 0
        rng = torch.Generator(device=device).manual_seed(seed)
        for ids in eval_prompts:
            if n_positions >= max_positions:
                break
            input_ids = ids.to(device).unsqueeze(0)
            r_logits = ref(input_ids).logits[0]
            c_logits = cand(input_ids).logits[0]
            v = min(r_logits.shape[-1], c_logits.shape[-1])
            v_last = v
            r_logits = r_logits[..., :v]
            c_logits = c_logits[..., :v]
            all_top1.append(top1_match(r_logits, c_logits))
            all_top5.append(topk_overlap(r_logits, c_logits, k=5))
            all_top10.append(topk_overlap(r_logits, c_logits, k=10))
            all_l1.append(logit_l1(r_logits, c_logits))
            all_l2.append(logit_l2(r_logits, c_logits))
            all_maxerr.append(logit_max_abs_err(r_logits, c_logits))
            all_cos.append(logit_cosine(r_logits, c_logits))
            all_kl.append(kl_div_ref_to_cand(r_logits, c_logits, temperature=temperature))
            u = torch.empty_like(r_logits, dtype=torch.float32)
            u.uniform_(1e-10, 1.0, generator=rng)
            gumbel = -torch.log(-torch.log(u))
            all_margin.append(post_gumbel_margin(r_logits, c_logits, gumbel, temperature=temperature))
            n_positions += r_logits.shape[0]
        cat = lambda xs: torch.cat(xs)
        out[f"{tag}/top1"] = cat(all_top1).float().mean().item()
        out[f"{tag}/top5"] = cat(all_top5).mean().item()
        out[f"{tag}/top10"] = cat(all_top10).mean().item()
        l1 = cat(all_l1); l2 = cat(all_l2); maxerr = cat(all_maxerr); cos = cat(all_cos)
        out[f"{tag}/logit_l1_mean"] = l1.mean().item()
        out[f"{tag}/logit_l1_p99"] = l1.quantile(0.99).item()
        out[f"{tag}/logit_mae_mean"] = (l1 / max(v_last, 1)).mean().item()
        out[f"{tag}/logit_l2_mean"] = l2.mean().item()
        out[f"{tag}/logit_l2_p99"] = l2.quantile(0.99).item()
        out[f"{tag}/logit_max_err_mean"] = maxerr.mean().item()
        out[f"{tag}/logit_max_err_p99"] = maxerr.quantile(0.99).item()
        out[f"{tag}/logit_cosine_mean"] = cos.mean().item()
        out[f"{tag}/logit_cosine_p01"] = cos.quantile(0.01).item()
        out[f"{tag}/kl_mean"] = cat(all_kl).mean().item()
        out[f"{tag}/kl_p99"] = cat(all_kl).quantile(0.99).item()
        out[f"{tag}/margin_mean"] = cat(all_margin).mean().item()
        out[f"{tag}/margin_p99"] = cat(all_margin).quantile(0.99).item()
        out[f"{tag}/n_positions"] = n_positions
        out[f"{tag}/vocab_size"] = v_last
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
    init_from_teacher: bool = False,
    keep_fp32_ref: bool = True,
    grad_checkpointing: bool = False,
    patch_nonmatmul: bool = True,
    int_nonmatmul_bitexact: bool = False,
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

    if init_from_teacher and teacher_source == "published":
        # Cast teacher's quantized weights into base before deepcopy — this way
        # the int24 student starts from the teacher's effective weights instead
        # of the unquantized fp32 base. Removes weight-side noise from the
        # student-vs-teacher gap (int24 has 10,000x more resolution than fp8 so
        # the cast itself is essentially lossless). Only activation-precision
        # noise remains as a training target.
        #
        # Handles three teacher formats:
        # - CompressedLinear (RedHatAI per-row): weight (e4m3) * weight_scale.
        # - FP8Linear (DeepSeek / Qwen3 block fp8): weight (e4m3) * blockwise
        #   expanded weight_scale_inv (typically 128x128 blocks).
        # - Plain nn.Linear (lm_head, etc.) at native precision.
        # Critical: must check fp8 dtype BEFORE falling through to plain-Linear
        # branch since FP8Linear IS an nn.Linear subclass.
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
        except ImportError:
            CompressedLinear = type(None)
        n_block = 0
        n_perrow = 0
        n_plain = 0
        teacher_modules = dict(teacher.named_modules())
        for name, bm in base.named_modules():
            if not isinstance(bm, nn.Linear):
                continue
            tm = teacher_modules.get(name)
            if tm is None:
                continue
            if (hasattr(tm, "weight_scale_inv")
                    and getattr(tm.weight, "dtype", None) == torch.float8_e4m3fn):
                # Block fp8 dequant
                W = tm.weight.to(torch.float32)
                S = tm.weight_scale_inv.to(torch.float32)
                bh = W.shape[0] // S.shape[0]
                bw = W.shape[1] // S.shape[1]
                Sexp = S.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
                w_fp = W * Sexp
                n_block += 1
            elif isinstance(tm, CompressedLinear):
                # Per-row fp8 dequant
                w_fp = tm.weight.float() * tm.weight_scale.float()
                n_perrow += 1
            elif (isinstance(tm, nn.Linear)
                    and getattr(tm.weight, "dtype", None) != torch.float8_e4m3fn):
                # Plain high-precision linear (e.g. lm_head)
                w_fp = tm.weight.detach().float()
                n_plain += 1
            else:
                continue
            if w_fp.shape != bm.weight.shape:
                print(f"  init_from_teacher: shape mismatch at {name} — skip")
                continue
            bm.weight.data.copy_(w_fp.to(bm.weight.dtype))
        print(f"  init_from_teacher: copied {n_block + n_perrow + n_plain} weight tensors "
              f"(block-fp8={n_block}, per-row-fp8={n_perrow}, plain={n_plain})")

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
    # If teacher has block-fp8 weights, stash the original fp8 weight + scale_inv
    # onto each IntLinear so the block_fp8_kernel_path forward can reproduce the
    # Triton kernel bit-exactly (needed for top1=1.0 on Qwen3-8B-FP8 etc.).
    if init_from_teacher and teacher_source == "published":
        teacher_modules = dict(teacher.named_modules())
        n_stash = 0
        for name, m in student.named_modules():
            tm = teacher_modules.get(name)
            if tm is None:
                continue
            if (hasattr(tm, "weight_scale_inv")
                    and getattr(tm.weight, "dtype", None) == torch.float8_e4m3fn
                    and isinstance(m, IntLinear)):
                # Register as actual buffers so .to(device) / state_dict work
                m.register_buffer(
                    "block_fp8_weight", tm.weight.detach().clone(),
                    persistent=False,
                )
                m.register_buffer(
                    "block_fp8_scale_inv", tm.weight_scale_inv.detach().clone(),
                    persistent=False,
                )
                n_stash += 1
        if n_stash:
            print(f"  stashed block-fp8 weight+scale_inv on {n_stash} IntLinears "
                  f"(for --block-fp8-kernel-path)")
    ops_cfg = IntOpsConfig(
        rmsnorm_bits=rmsnorm_bits,
        softmax_lut_size=softmax_lut_size,
        softmax_x_min=softmax_x_min,
        silu_lut_size=silu_lut_size,
        attn_matmul_bits=attn_matmul_bits,
    )
    if int_nonmatmul_bitexact:
        from difr_expt.int_ops_bitexact import patch_model_int_bitexact
        counts = patch_model_int_bitexact(student)
        print(f"  student: BITEXACT int commit wrappers: {counts} "
              "(int30 round-trip + teacher kernels; output bit-exact teacher)")
    elif patch_nonmatmul:
        counts = patch_model_int_nonmatmul(student, ops_cfg)
        print(f"  student: int non-matmul counts: {counts}")
    else:
        print(f"  student: int non-matmul DISABLED (using teacher-equivalent softmax/SiLU/RMSNorm/attn)")
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
    ap.add_argument("--init-from-teacher", action="store_true",
                    help="Cast teacher's quantized weights → student weight_fp init "
                         "(instead of fp32 base weights). Resolves weight-side noise "
                         "for free; only activation-precision gap remains for training.")
    ap.add_argument("--matmul-loss-weight", type=float, default=0.0,
                    help="Weight on Σ per-matmul L2 loss between paired teacher/student "
                         "Linear outputs. 0 disables; default 0 keeps logit-only loss.")
    ap.add_argument("--logit-loss-weight", type=float, default=1.0,
                    help="Weight on final-logit (KL + aux MSE) loss. Set 0 for per-matmul-only.")
    ap.add_argument("--matmul-loss-norm", choices=["l1", "l2"], default="l2")
    ap.add_argument("--no-trainable-gamma-bias", action="store_true",
                    help="Keep RMSNorm gamma and Linear biases frozen.")
    ap.add_argument("--trainable-scales", action="store_true",
                    help="Promote per-row matmul scales to nn.Parameters (otherwise "
                         "derived from absmax each forward).")
    ap.add_argument("--no-fp32-ref", action="store_true",
                    help="Don't keep an fp32 ref in memory (saves RAM on CPU runs).")
    ap.add_argument("--no-int-nonmatmul", action="store_true",
                    help="Skip patching softmax/SiLU/RMSNorm/attention to int variants. "
                         "Diagnostic: isolates the contribution of non-matmul int ops "
                         "to the student-teacher gap.")
    ap.add_argument("--int-nonmatmul-bitexact", action="store_true",
                    help="Wrap teacher's non-matmul ops (RMSNorm, SiLU, etc.) with "
                         "explicit int30 commit wrappers. The bf16↔int30 round-trip "
                         "is exact identity, so kernels run on the same inputs as "
                         "teacher → output bit-exact. This makes the int architecture "
                         "visible at every layer boundary (prover commits int30 + scale "
                         "per token). Combine with --int-matmul-path and --activation-fp8 "
                         "for the full-int model: every value committed as int at every "
                         "boundary; deterministic kernels in between; top1=1.0 vs teacher.")
    ap.add_argument("--activation-fp8", action="store_true",
                    help="Use fp8 e4m3 levels (NOT uniform int24 grid) for per-token "
                         "activation quant. Mimics the teacher's fp8-dynamic activation "
                         "rounding pattern exactly. The 256 fp8 levels are exactly "
                         "representable in int24 storage; in a ZK circuit, this is a "
                         "256-entry LUT, no float ops needed.")
    ap.add_argument("--activation-block-fp8", action="store_true",
                    help="Use fp8 e4m3 levels with BLOCK granularity along the last dim "
                         "(per --activation-block-size, default 128). Mimics Qwen3-style "
                         "block-fp8 teacher activation quantization.")
    ap.add_argument("--activation-block-size", type=int, default=128,
                    help="Block size (along last dim) for --activation-block-fp8.")
    ap.add_argument("--int-matmul-path", action="store_true",
                    help="Compute matmuls via explicit int24 × int24 + public-scale "
                         "dequant (executed as fp64, bit-equivalent for int48 products). "
                         "Demonstrates all-integer arithmetic — what a ZK circuit would "
                         "do. Currently only supports --activation-fp8 (per-token) scheme.")
    ap.add_argument("--block-fp8-kernel-path", action="store_true",
                    help="For block-fp8 teachers (e.g. Qwen3-8B-FP8): call the same "
                         "Triton w8a8_block_fp8_matmul kernel as the teacher, on the "
                         "stashed original fp8 weight + per-128-block scale_inv. This is "
                         "bit-exact teacher by construction. The ZK-spec equivalent is "
                         "the per-K-block fp32 emulation (sum int30 partial products "
                         "with per-tile scales; verified 99.99% per-layer bit-exact in "
                         "standalone test).")
    ap.add_argument("--skip-pretrain-checkpoint", action="store_true",
                    help="Don't write pretrain.pt (initial-state snapshot). Saves "
                         "~30 GB disk on 8B runs where the init is reproducible from "
                         "--init-from-teacher.")
    ap.add_argument("--skip-final-checkpoint", action="store_true",
                    help="Don't write final.pt (last-step snapshot, usually redundant "
                         "with best.pt). Saves ~30 GB disk on 8B runs.")
    ap.add_argument("--no-matmul-hooks", action="store_true",
                    help="Skip registering per-matmul forward hooks. Saves a few GB "
                         "of activation memory on 8B and disables per-matmul metrics "
                         "(which aren't used by V2-style logit-only training).")
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
        init_from_teacher=args.init_from_teacher,
        matmul_loss_weight=args.matmul_loss_weight,
        matmul_loss_norm=args.matmul_loss_norm,
        logit_loss_weight=args.logit_loss_weight,
        trainable_gamma_bias=not args.no_trainable_gamma_bias,
        trainable_scales=args.trainable_scales,
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
        init_from_teacher=cfg.init_from_teacher,
        keep_fp32_ref=not args.no_fp32_ref,
        grad_checkpointing=cfg.grad_checkpointing,
        patch_nonmatmul=not args.no_int_nonmatmul and not args.int_nonmatmul_bitexact,
        int_nonmatmul_bitexact=args.int_nonmatmul_bitexact,
    )

    n_biases = promote_intlinear_biases(student)
    print(f"  promoted {n_biases} biases to nn.Parameter")
    if cfg.trainable_luts:
        n_sm, n_silu = promote_luts(student)
        print(f"  promoted {n_sm} softmax LUTs + {n_silu} silu LUTs to nn.Parameter")

    # On 8B, every layer's forward output captured by a non-detached hook adds
    # direct references that prevent intermediate frees during training. When
    # the run doesn't actually need per-matmul loss/metrics (V2-style logit-only
    # training), skip hooks entirely. Falls back to empty captures so eval code
    # paths still work — they just emit no matmul/* metrics.
    if args.no_matmul_hooks:
        print("  per-matmul hooks SKIPPED (--no-matmul-hooks); per-matmul metrics will be empty")
        s_cap = MatmulCapture(detach=False)
        t_cap = MatmulCapture(detach=True)
        paired_layers = []
        _hook_handles = []
    else:
        s_cap, t_cap, paired_layers, _hook_handles = register_matmul_hooks(student, teacher)
        print(f"  per-matmul hooks: paired {len(paired_layers)} student↔teacher Linear modules")

    if args.activation_fp8:
        n = 0
        for m in student.modules():
            if isinstance(m, IntLinear):
                m.activation_scheme = "fp8_e4m3"
                n += 1
        print(f"  activation scheme set to fp8_e4m3 on {n} IntLinears (mimics teacher's fp8-dynamic activation quant)")
    if args.activation_block_fp8:
        n = 0
        for m in student.modules():
            if isinstance(m, IntLinear):
                m.activation_scheme = "block_fp8_e4m3"
                m.activation_block_size = args.activation_block_size
                n += 1
        print(f"  activation scheme set to block_fp8_e4m3 (block_size={args.activation_block_size}) on {n} IntLinears")
    if args.int_matmul_path:
        n = 0
        for m in student.modules():
            if isinstance(m, IntLinear):
                m.use_int_matmul_path = True
                n += 1
        print(f"  int24×int24 matmul path enabled on {n} IntLinears (executed as fp64; bit-equivalent for int48 products)")
    if args.block_fp8_kernel_path:
        n = 0
        for m in student.modules():
            if isinstance(m, IntLinear) and m.block_fp8_weight is not None:
                m.use_block_fp8_kernel_path = True
                n += 1
        print(f"  block-fp8 kernel path enabled on {n} IntLinears "
              f"(calls Triton w8a8_block_fp8_matmul; bit-exact teacher)")

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
        # PagedAdamW8bit pages optimizer state in/out from CPU via the unified
        # CUDA memory subsystem. This avoids the ~14 GB GPU residency of
        # AdamW8bit's state buffers — important when the model + grads already
        # occupy ~78 GB on the 80 GB H100, leaving no headroom for state init.
        try:
            optimizer = bnb.optim.PagedAdamW8bit(param_groups, weight_decay=0.0)
            print("  optimizer: bnb.PagedAdamW8bit (CPU-paged state)")
        except AttributeError:
            optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=0.0)
            print("  optimizer: bnb.AdamW8bit (GPU state)")
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
    pre.update(per_matmul_eval(teacher, student, eval_prompts, device, s_cap, t_cap, paired_layers))
    pre["step"] = 0; pre["lr"] = 0.0; pre["loss"] = float("nan")
    log(pre)
    print(f"  pre-train: student_vs_teacher top1={pre['student_vs_teacher/top1']:.4f}  "
          f"kl_p99={pre['student_vs_teacher/kl_p99']:.4e}  "
          f"margin_p99={pre['student_vs_teacher/margin_p99']:.4e}")
    if ref is not None:
        print(f"  pre-train: student_vs_ref top1={pre['student_vs_ref/top1']:.4f}  "
              f"teacher_vs_ref top1={pre['teacher_vs_ref/top1']:.4f}")

    if not args.skip_pretrain_checkpoint and cfg.steps > 0:
        save_trained_deltas(student, out_dir / "pretrain.pt")

    student.train()
    best_top1 = pre["student_vs_teacher/top1"]
    plateau = 0
    best_step = 0
    # Save an initial best.pt snapshot of the cast-from-teacher state. On 8B
    # this is ~28 GB — skip when --skip-pretrain-checkpoint also implies
    # we don't need it (only training-time best.pt overwrites matter then).
    if cfg.steps > 0 and not args.skip_pretrain_checkpoint:
        save_trained_deltas(student, out_dir / "best.pt")

    # Drop allocator caches before training — pre-eval transients can keep
    # 5-10 GB reserved that the optimizer init needs.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for step in range(1, cfg.steps + 1):
        batch_ids = [train_prompts[rng.randrange(len(train_prompts))] for _ in range(cfg.batch)]
        input_ids, mask = pad_collate(batch_ids, pad_id)
        input_ids = input_ids.to(device); mask = mask.to(device)

        s_cap.clear(); t_cap.clear()
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

        logit_term = kl_loss + cfg.aux_weight * mse_loss
        if cfg.matmul_loss_weight > 0.0:
            mm_loss = per_matmul_loss(s_cap, t_cap, paired_layers, norm=cfg.matmul_loss_norm)
        else:
            mm_loss = torch.zeros((), device=device)
        loss = cfg.logit_loss_weight * logit_term + cfg.matmul_loss_weight * mm_loss

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
                  f"mse={mse_loss.item():.4e}  mm={mm_loss.item():.4e}  scale={progress:.3f}")

        if step % cfg.eval_every == 0:
            student.eval()
            ev = evaluate(
                teacher, student, eval_prompts, device,
                max_positions=cfg.eval_max_positions,
                temperature=cfg.temperature, seed=cfg.seed,
                ref_model=ref,
            )
            ev.update(per_matmul_eval(teacher, student, eval_prompts, device, s_cap, t_cap, paired_layers))
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

    if not args.skip_final_checkpoint and cfg.steps > 0:
        save_trained_deltas(student, out_dir / "final.pt")
    # Final eval with the best checkpoint reloaded — but for steps=0 the model
    # is already in its post-init state and best.pt was never written. Skip the
    # reload in that case (and re-use the pre eval as post).
    print(f"[{time.strftime('%H:%M:%S')}] final eval with best.pt")
    # On 8B, the optimizer state + student + grads occupy ~75 GB; loading best.pt
    # (30 GB) directly to GPU on top of that OOMs the 80 GB H100. Free optimizer
    # first, then load checkpoint to CPU and transfer per-tensor below.
    del optimizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best_path = out_dir / "best.pt"
    if cfg.steps > 0 and best_path.exists():
        payload = torch.load(best_path, weights_only=False, map_location="cpu")
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
    else:
        # No training happened (steps=0) → student is already at its post-init
        # state and the pre eval is equivalent to a post eval. Skip the reload.
        print("  steps=0 or best.pt absent → skipping reload; reusing student in-memory state")

    student.eval()
    post = evaluate(
        teacher, student, eval_prompts, device,
        max_positions=cfg.eval_max_positions,
        temperature=cfg.temperature, seed=cfg.seed,
        ref_model=ref,
    )
    post.update(per_matmul_eval(teacher, student, eval_prompts, device, s_cap, t_cap, paired_layers))
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
