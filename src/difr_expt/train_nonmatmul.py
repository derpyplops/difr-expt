"""Train RMSNorm gammas + Linear biases to compensate for int approximation
error in the non-matmul ops (Approach B, Phase 2 of train-nonmatmul-int).

The setup:
  - Teacher: fp32 reference, frozen.
  - Student: deepcopy of teacher, with every nn.Linear replaced by a FROZEN
    IntLinear (weights in int+scale buffers, no fp32 shadow → no gradient).
    Then non-matmul ops replaced via patch_model_int_nonmatmul (gives us
    IntRMSNorm whose `weight` is already an nn.Parameter for gamma).
  - Trainable parameters: only IntRMSNorm.weight (gammas) and IntLinear.bias
    (which we promote from buffer → nn.Parameter).
  - Loss: logit-L2 vs teacher, with optional per-Linear normalized-MSE aux loss.

Saves a checkpoint as a dict of trained deltas keyed by qualified module name,
which `run_baseline.py --load-trained` re-applies.

Usage:
    python -m difr_expt.cache_prompts --tokenizer Qwen/Qwen2.5-0.5B --n-prompts 10000 --max-len 512 --out prompts.pt
    python -m difr_expt.train_nonmatmul \
        --model Qwen/Qwen2.5-0.5B --dtype float32 \
        --prompts prompts.pt \
        --weight-bits 24 --activation-bits 24 \
        --lr 1e-4 --steps 2000 --batch 4 --warmup 100 --eval-every 500 \
        --out experiments/train-nonmatmul-int/data/phase2_run
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

from difr_expt.int_cast import IntLinear, patch_model_int_cast, patch_model_int_embedding
from difr_expt.int_ops import IntRMSNorm
from difr_expt.patch_hf_model import (
    IntOpsConfig, IntSiLUModule, IntSoftmaxModule, patch_model_int_nonmatmul,
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
    lr: float = 1e-4
    steps: int = 2000
    batch: int = 4
    eval_every: int = 500
    warmup: int = 100
    grad_clip: float = 1.0
    weight_bits: int = 24
    activation_bits: int = 24
    matmul_dtype: str = "auto"
    rmsnorm_bits: int = 24
    rmsnorm_nr_iter: int = 2
    softmax_lut_size: int = 1024
    softmax_x_min: float = -16.0
    silu_lut_size: int = 4096
    attn_matmul_bits: int = 24
    dtype: str = "float32"
    seed: int = 42
    aux_weight: float = 0.0
    plateau_patience: int = 3
    eval_n_prompts: int = 100
    eval_max_positions: int = 100_000
    temperature: float = 1.0
    # Approach C: also unfreeze IntLinear weight shadows for STE training.
    trainable_matmul_weights: bool = False
    # Promote softmax + sigmoid LUT entries to nn.Parameter (train the actual
    # non-matmul approximation tables).
    trainable_luts: bool = False
    # Toggle non-matmul / embedding int patches (default off keeps Approach B behaviour
    # bit-compatible with prior runs, since this script used to always apply them).
    int_nonmatmul: bool = True
    int_embedding: bool = False
    embedding_bits: int = 24


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def promote_intlinear_biases(model: nn.Module) -> int:
    """Replace each IntLinear.bias buffer with an nn.Parameter (trainable).

    IntLinear stores bias as a register_buffer (so it's part of the state dict
    but not a parameter). For Phase 2 training we need gradients to flow into
    the bias term, so we swap it to a Parameter. Returns the count of
    biases promoted.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, IntLinear) and m.bias is not None and not isinstance(m.bias, nn.Parameter):
            b = m.bias.detach().clone()
            # Remove the buffer entry, then assign a Parameter of the same name.
            if "bias" in m._buffers:
                del m._buffers["bias"]
            m.bias = nn.Parameter(b)
            n += 1
    return n


def collect_trainable(
    model: nn.Module,
    include_weight_fp: bool = False,
    include_luts: bool = False,
) -> list[tuple[str, nn.Parameter]]:
    """Pick out the gamma + bias parameters and return (name, param) pairs.

    If `include_weight_fp` is True (Approach C), also pull in the fp32
    `weight_fp` shadow of every trainable IntLinear so the matmul weights
    train through STE alongside the gammas + biases.

    If `include_luts` is True, also include the softmax + sigmoid LUT entries
    (after they've been promoted to nn.Parameter via `make_lut_trainable()`).
    """
    params: list[tuple[str, nn.Parameter]] = []
    for name, m in model.named_modules():
        if isinstance(m, IntRMSNorm):
            params.append((f"{name}.weight", m.weight))
        elif isinstance(m, IntLinear):
            if isinstance(m.bias, nn.Parameter):
                params.append((f"{name}.bias", m.bias))
            if include_weight_fp and m.weight_fp is not None:
                params.append((f"{name}.weight_fp", m.weight_fp))
        elif include_luts and isinstance(m, (IntSiLUModule, IntSoftmaxModule)):
            if m.lut is not None:
                params.append((f"{name}.lut", m.lut))
    return params


def promote_luts(model: nn.Module) -> tuple[int, int]:
    """Convert every IntSoftmaxModule and IntSiLUModule's LUT buffer into a
    trainable nn.Parameter. Returns (n_softmax, n_silu) promoted."""
    n_sm, n_silu = 0, 0
    for m in model.modules():
        if isinstance(m, IntSoftmaxModule) and m.lut is None:
            m.make_lut_trainable()
            n_sm += 1
        elif isinstance(m, IntSiLUModule) and m.lut is None:
            m.make_lut_trainable()
            n_silu += 1
    return n_sm, n_silu


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
    ref_model: nn.Module,
    student: nn.Module,
    eval_prompts: list[torch.Tensor],
    device: str,
    max_positions: int,
    temperature: float,
    seed: int,
) -> dict[str, float]:
    all_top1, all_top5, all_l2, all_kl, all_margin = [], [], [], [], []
    n_positions = 0
    rng = torch.Generator(device=device).manual_seed(seed)

    for ids in eval_prompts:
        if n_positions >= max_positions:
            break
        input_ids = ids.to(device).unsqueeze(0)
        ref_logits = ref_model(input_ids).logits[0]
        int_logits = student(input_ids).logits[0]
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
        n_positions += ref_logits.shape[0]

    out: dict[str, float] = {}
    cat = lambda xs: torch.cat(xs)
    out["all_top1"] = cat(all_top1).float().mean().item()
    out["all_top5"] = cat(all_top5).mean().item()
    out["all_logit_l2_mean"] = cat(all_l2).mean().item()
    out["all_logit_l2_p99"] = cat(all_l2).quantile(0.99).item()
    out["all_kl_mean"] = cat(all_kl).mean().item()
    out["all_kl_p99"] = cat(all_kl).quantile(0.99).item()
    out["all_margin_mean"] = cat(all_margin).mean().item()
    out["all_margin_p99"] = cat(all_margin).quantile(0.99).item()
    out["n_positions"] = n_positions
    return out


def save_trained_deltas(model: nn.Module, path: Path) -> None:
    """Save trained gamma + bias + optional weight_fp values keyed by qualified name.

    Format:
        {
            "rmsnorm_gamma": {name: tensor[hidden]},
            "linear_bias":   {name: tensor[out]},
            "linear_weight_fp": {name: tensor[O, I]}  (Approach C only — saved
                in bf16 to halve disk),
        }
    Loading honours all keys via run_baseline.py --load-trained.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help=".pt from cache_prompts.py")
    ap.add_argument("--out", required=True, help="run dir")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-bits", type=int, default=24)
    ap.add_argument("--activation-bits", type=int, default=24)
    ap.add_argument("--matmul-dtype", default="auto",
                    choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--rmsnorm-bits", type=int, default=24)
    ap.add_argument("--rmsnorm-nr-iter", type=int, default=2)
    ap.add_argument("--softmax-lut-size", type=int, default=1024)
    ap.add_argument("--softmax-x-min", type=float, default=-16.0)
    ap.add_argument("--silu-lut-size", type=int, default=4096)
    ap.add_argument("--attn-matmul-bits", type=int, default=24)
    ap.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aux-weight", type=float, default=0.0)
    ap.add_argument("--plateau-patience", type=int, default=3)
    ap.add_argument("--eval-n-prompts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    # Approach C flags.
    ap.add_argument("--trainable-matmul-weights", action="store_true",
                    help="Also unfreeze IntLinear weight_fp shadows (Approach C). "
                         "When set, the optimizer trains gammas + biases + every "
                         "Linear's fp32 weight shadow via STE.")
    ap.add_argument("--trainable-luts", action="store_true",
                    help="Promote softmax-exp + sigmoid-silu LUT entries to "
                         "nn.Parameter so the optimizer can train them.")
    # Wiring for non-matmul / embedding patches (parity with run_baseline.py).
    ap.add_argument("--no-int-nonmatmul", action="store_true",
                    help="Skip patch_model_int_nonmatmul (default: apply). Use for "
                         "matmul-only training, e.g. recovering Approach C without "
                         "int RMSNorm/softmax/SiLU.")
    ap.add_argument("--int-nonmatmul", action="store_true",
                    help="Apply patch_model_int_nonmatmul (default-on; this flag "
                         "exists for parity with run_baseline.py and is the inverse "
                         "of --no-int-nonmatmul, which wins if both supplied).")
    ap.add_argument("--int-embedding", action="store_true",
                    help="Also patch nn.Embedding to IntEmbedding (per-vocab-row "
                         "symmetric int24 quant).")
    ap.add_argument("--embedding-bits", type=int, default=24)
    args = ap.parse_args()

    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    # `--no-int-nonmatmul` wins over `--int-nonmatmul`; default is on (matches
    # prior behaviour of this script which unconditionally applied non-matmul
    # patches).
    int_nonmatmul = not args.no_int_nonmatmul
    cfg_kwargs = {k: v for k, v in vars(args).items()
                  if k not in ("no_int_nonmatmul",)}
    cfg_kwargs["int_nonmatmul"] = int_nonmatmul
    cfg = Config(**cfg_kwargs)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[cfg.dtype]
    matmul_dtype_torch = None
    if cfg.matmul_dtype != "auto":
        matmul_dtype_torch = DTYPE_MAP[cfg.matmul_dtype]
    print(f"[{time.strftime('%H:%M:%S')}] loading {cfg.model} ({dtype}) on {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    # Teacher: frozen fp32 reference.
    teacher = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student: deepcopy → IntLinear (frozen or trainable depending on Approach
    # B/C) → int non-matmul ops (optional) → int embedding (optional).
    train_mm = cfg.trainable_matmul_weights
    print(f"[{time.strftime('%H:%M:%S')}] building student "
          f"(IntLinear trainable={train_mm}, int_nonmatmul={cfg.int_nonmatmul}, "
          f"int_embedding={cfg.int_embedding})")
    student = copy.deepcopy(teacher)
    replaced = patch_model_int_cast(
        student,
        weight_bits=cfg.weight_bits,
        activation_bits=cfg.activation_bits,
        trainable=train_mm,
        matmul_dtype=matmul_dtype_torch,
    )
    print(f"  replaced {len(replaced)} Linears with IntLinears (trainable={train_mm})")

    if cfg.int_nonmatmul:
        ops_cfg = IntOpsConfig(
            rmsnorm_bits=cfg.rmsnorm_bits,
            rmsnorm_nr_iter=cfg.rmsnorm_nr_iter,
            softmax_lut_size=cfg.softmax_lut_size,
            softmax_x_min=cfg.softmax_x_min,
            silu_lut_size=cfg.silu_lut_size,
            attn_matmul_bits=cfg.attn_matmul_bits,
        )
        counts = patch_model_int_nonmatmul(student, ops_cfg)
        print(f"  int non-matmul replacement counts: {counts}")

    if cfg.int_embedding:
        emb_replaced = patch_model_int_embedding(student, bits=cfg.embedding_bits)
        print(f"  replaced {len(emb_replaced)} nn.Embedding modules with IntEmbedding "
              f"(bits={cfg.embedding_bits})")

    n_biases = promote_intlinear_biases(student)
    print(f"  promoted {n_biases} IntLinear biases to nn.Parameter")

    if cfg.trainable_luts:
        n_sm_luts, n_silu_luts = promote_luts(student)
        print(f"  promoted {n_sm_luts} softmax LUTs + {n_silu_luts} silu LUTs to nn.Parameter")

    trainable = collect_trainable(
        student,
        include_weight_fp=train_mm,
        include_luts=cfg.trainable_luts,
    )
    keep_ids = {id(p) for _, p in trainable}
    freeze_all_but(student, keep_ids)
    n_train_params = sum(p.numel() for _, p in trainable)
    print(f"  trainable: {len(trainable)} tensors / {n_train_params:,} params")
    for name, p in trainable[:5]:
        print(f"    e.g. {name}: shape={tuple(p.shape)}, dtype={p.dtype}")

    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=cfg.lr, weight_decay=0.0)

    # Aux hooks: pair teacher.nn.Linear outputs with student.IntLinear outputs by name.
    teacher_acts: dict[str, torch.Tensor] = {}
    student_acts: dict[str, torch.Tensor] = {}
    aux_hooks: list = []
    if cfg.aux_weight > 0:
        def make_t_hook(name):
            def hook(module, inputs, output):
                teacher_acts[name] = output
            return hook

        def make_s_hook(name):
            def hook(module, inputs, output):
                student_acts[name] = output
            return hook

        for name, m in teacher.named_modules():
            if isinstance(m, nn.Linear):
                aux_hooks.append(m.register_forward_hook(make_t_hook(name)))
        for name, m in student.named_modules():
            if isinstance(m, IntLinear):
                aux_hooks.append(m.register_forward_hook(make_s_hook(name)))
        print(f"  aux hooks: {len(aux_hooks)} (aux_weight={cfg.aux_weight})")

    # Prompts
    prompts: list[torch.Tensor] = torch.load(cfg.prompts, weights_only=False)
    print(f"  loaded {len(prompts)} prompts from {cfg.prompts}")
    rng = random.Random(cfg.seed)
    train_prompts = prompts[: -cfg.eval_n_prompts]
    eval_prompts = prompts[-cfg.eval_n_prompts:]
    print(f"  train: {len(train_prompts)}  eval: {len(eval_prompts)}")

    metrics_file = open(out_dir / "metrics.jsonl", "a")
    t_start = time.time()

    def log(record: dict[str, Any]) -> None:
        record["wall_s"] = time.time() - t_start
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    # Pre-training eval.
    print(f"[{time.strftime('%H:%M:%S')}] step 0 eval")
    pre = evaluate(teacher, student, eval_prompts, device,
                   max_positions=cfg.eval_max_positions,
                   temperature=cfg.temperature, seed=cfg.seed)
    pre["step"] = 0; pre["lr"] = 0.0; pre["loss"] = float("nan")
    log(pre)
    print(f"  pre-train: top1={pre['all_top1']:.4f} l2_p99={pre['all_logit_l2_p99']:.2e} kl_p99={pre['all_kl_p99']:.2e}")

    student.train()
    best_top1 = pre["all_top1"]
    plateau = 0
    plateau_limit = cfg.plateau_patience
    best_ckpt_step = 0

    save_trained_deltas(student, out_dir / "pretrain.pt")

    for step in range(1, cfg.steps + 1):
        batch_ids = [train_prompts[rng.randrange(len(train_prompts))] for _ in range(cfg.batch)]
        input_ids, mask = pad_collate(batch_ids, pad_id)
        input_ids = input_ids.to(device); mask = mask.to(device)

        if cfg.aux_weight > 0:
            teacher_acts.clear()
            student_acts.clear()
        with torch.inference_mode():
            ref_logits = teacher(input_ids).logits
        student_logits = student(input_ids).logits
        v = min(ref_logits.shape[-1], student_logits.shape[-1])
        diff = student_logits[..., :v].float() - ref_logits[..., :v].float()
        valid = mask[..., None].float()
        logit_loss = (diff.pow(2) * valid).sum() / valid.sum().clamp_min(1.0) / v

        aux_loss = torch.tensor(0.0, device=device)
        if cfg.aux_weight > 0 and teacher_acts and student_acts:
            common = set(teacher_acts) & set(student_acts)
            terms = []
            for name in common:
                t = teacher_acts[name].float()
                s = student_acts[name].float()
                if t.shape != s.shape:
                    continue
                denom = t.pow(2).mean().clamp_min(1e-12)
                terms.append((s - t).pow(2).mean() / denom)
            if terms:
                aux_loss = torch.stack(terms).mean()

        loss = logit_loss + cfg.aux_weight * aux_loss

        lr_now = cosine_warmup(step, cfg.warmup, cfg.steps, cfg.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            if cfg.aux_weight > 0:
                print(f"  step {step:5d} lr={lr_now:.2e} loss={loss.item():.4e} "
                      f"logit={logit_loss.item():.4e} aux={aux_loss.item():.4e}")
            else:
                print(f"  step {step:5d} lr={lr_now:.2e} loss={loss.item():.4e}")
        if step % cfg.eval_every == 0:
            student.eval()
            ev = evaluate(teacher, student, eval_prompts, device,
                          max_positions=cfg.eval_max_positions,
                          temperature=cfg.temperature, seed=cfg.seed)
            ev["step"] = step; ev["lr"] = lr_now; ev["loss"] = loss.item()
            log(ev)
            print(
                f"  [eval @ {step}] top1={ev['all_top1']:.4f}  l2_p99={ev['all_logit_l2_p99']:.2e}  "
                f"kl_p99={ev['all_kl_p99']:.2e}  margin_p99={ev['all_margin_p99']:.2e}"
            )
            student.train()
            if ev["all_top1"] > best_top1 + 1e-5:
                best_top1 = ev["all_top1"]
                plateau = 0
                save_trained_deltas(student, out_dir / "best.pt")
                best_ckpt_step = step
            else:
                plateau += 1
            if ev["all_top1"] >= 0.9999:
                print(f"  *** hit stretch target at step {step}; stopping")
                break
            if plateau >= plateau_limit and step >= cfg.warmup + 3 * cfg.eval_every:
                print(f"  *** plateau ({plateau} no-improvements); stopping at step {step}")
                break

    save_trained_deltas(student, out_dir / "final.pt")
    metrics_file.close()
    print(f"done in {time.time() - t_start:.0f}s; best all_top1 = {best_top1:.4f} @ step {best_ckpt_step}")


if __name__ == "__main__":
    main()
