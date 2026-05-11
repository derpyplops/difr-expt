"""Train an int-cast student to match a bf16 reference (logit-L2, STE).

Usage:
    python -m difr_expt.cache_prompts --tokenizer <model> --n-prompts 10000 --max-len 512 --out prompts.pt
    python -m difr_expt.train \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --prompts prompts.pt \\
        --lr 1e-6 --steps 5000 --batch 4 \\
        --eight-bit-adam \\
        --out runs/llama_lr1e-6

Layout per run dir:
    metrics.jsonl       — one row per eval, with step, loss, top-1, etc.
    step0000.pt         — initial checkpoint (no training applied)
    step1000.pt, ...    — periodic checkpoints (weight_fp + optim state)
    final.pt            — last checkpoint
    config.json         — the CLI args
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from difr_expt.int_cast import IntLinear, patch_model_int_cast
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
    lr: float = 1e-6
    steps: int = 5000
    batch: int = 4
    eval_every: int = 250
    ckpt_every: int = 1000
    warmup: int = 100
    grad_clip: float = 1.0
    weight_bits: int = 16
    activation_bits: int = 16
    dtype: str = "bfloat16"
    seed: int = 42
    eight_bit_adam: bool = False
    grad_checkpoint: bool = False
    aux_weight: float = 0.0
    plateau_patience: int = 3
    eval_n_prompts: int = 100
    eval_max_positions: int = 100_000  # cap to keep eval fast
    temperature: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(student: nn.Module, lr: float, eight_bit: bool):
    """Return an AdamW over weight_fp parameters of every IntLinear.

    With `eight_bit=True`, uses bitsandbytes' 8-bit AdamW to keep optimizer
    state in 8 bits (cuts memory ~4×). Required for 8B models on 80 GB H100.
    """
    params = []
    for m in student.modules():
        if isinstance(m, IntLinear) and m.weight_fp is not None:
            params.append(m.weight_fp)
    n_params = sum(p.numel() for p in params)
    if eight_bit:
        import bitsandbytes as bnb  # type: ignore
        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    return opt, n_params


def freeze_non_intlinear(student: nn.Module) -> None:
    """Freeze every parameter that is not the weight_fp of an IntLinear."""
    intlin_param_ids = set()
    for m in student.modules():
        if isinstance(m, IntLinear) and m.weight_fp is not None:
            intlin_param_ids.add(id(m.weight_fp))
    for p in student.parameters():
        if id(p) not in intlin_param_ids:
            p.requires_grad = False


def cosine_warmup(step: int, warmup: int, total: int, peak_lr: float, floor: float = 0.1) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (floor + (1.0 - floor) * cos)


def pad_collate(batch_ids: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of 1D id tensors to the longest length; return (ids, mask)."""
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
    """Compute the same headline metrics as the baseline harness."""
    all_top1, all_top5, all_l2, all_kl, all_margin = [], [], [], [], []
    last_top1, last_top5 = [], []
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
        last_top1.append(t1[-1:]); last_top5.append(t5[-1:])
        n_positions += ref_logits.shape[0]

    out: dict[str, float] = {}
    cat = lambda xs: torch.cat(xs)
    out["all_top1"] = cat(all_top1).float().mean().item()
    out["all_top5"] = cat(all_top5).mean().item()
    out["all_logit_l2_mean"] = cat(all_l2).mean().item()
    out["all_kl_mean"] = cat(all_kl).mean().item()
    out["all_kl_p99"] = cat(all_kl).quantile(0.99).item()
    out["all_margin_mean"] = cat(all_margin).mean().item()
    out["all_margin_p99"] = cat(all_margin).quantile(0.99).item()
    out["last_top1"] = cat(last_top1).float().mean().item()
    out["last_top5"] = cat(last_top5).mean().item()
    out["n_positions"] = n_positions
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="path to .pt from cache_prompts.py")
    ap.add_argument("--out", required=True, help="run dir for checkpoints + metrics")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-bits", type=int, default=16)
    ap.add_argument("--activation-bits", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eight-bit-adam", action="store_true",
                    help="Use bitsandbytes 8-bit AdamW (required for 8B on 80GB H100)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Enable HF gradient checkpointing on the student (cuts activation memory ~10x at the cost of extra forward compute). Required for 8B on 80GB.")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="If >0, add per-Linear normalized-MSE loss (Luke's aux). Default 0 = pure logit-L2.")
    ap.add_argument("--plateau-patience", type=int, default=3,
                    help="Stop after this many evals with no top-1 improvement. Default 3.")
    ap.add_argument("--eval-n-prompts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    # Unbuffered stdout so nohup logs show progress live, not in 4KB chunks.
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    cfg = Config(**vars(args))
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

    # Teacher: frozen bf16 reference. Stays in inference-mode forever.
    teacher = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student: deepcopy of teacher, then Linears swapped for trainable IntLinears.
    print(f"[{time.strftime('%H:%M:%S')}] building student (int-cast)")
    student = copy.deepcopy(teacher)
    replaced = patch_model_int_cast(
        student,
        weight_bits=cfg.weight_bits,
        activation_bits=cfg.activation_bits,
        trainable=True,
    )
    print(f"  replaced {len(replaced)} Linears with trainable IntLinears")
    freeze_non_intlinear(student)
    if cfg.grad_checkpoint:
        # Required for 8B on 80GB. HF's checkpointing recomputes activations
        # in backward; saves ~40 GB of activation memory at the cost of an
        # extra forward per step. use_reentrant=False is required because we
        # use param-grad on weight_fp directly (the reentrant version trips
        # on this); modern HF defaults to False anyway.
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("  gradient checkpointing enabled")

    # Per-matmul aux loss plumbing. Hooks capture each Linear's output (teacher)
    # and IntLinear's output (student); the aux term is mean over those layers
    # of normalized-MSE between corresponding outputs. The dicts are populated
    # on every forward and cleared after the loss is computed.
    teacher_acts: dict[str, torch.Tensor] = {}
    student_acts: dict[str, torch.Tensor] = {}
    aux_hooks: list = []
    if cfg.aux_weight > 0:
        def make_t_hook(name):
            def hook(module, inputs, output):
                # Teacher runs under inference_mode; just stash.
                teacher_acts[name] = output
            return hook

        def make_s_hook(name):
            def hook(module, inputs, output):
                # Student needs grad through output.
                student_acts[name] = output
            return hook

        for name, m in teacher.named_modules():
            if isinstance(m, nn.Linear):
                aux_hooks.append(m.register_forward_hook(make_t_hook(name)))
        for name, m in student.named_modules():
            if isinstance(m, IntLinear):
                aux_hooks.append(m.register_forward_hook(make_s_hook(name)))
        print(f"  aux hooks registered on {len(aux_hooks)} modules (aux_weight={cfg.aux_weight})")

    optimizer, n_trainable = build_optimizer(student, cfg.lr, eight_bit=cfg.eight_bit_adam)
    print(f"  trainable params: {n_trainable:,}  (8-bit Adam: {cfg.eight_bit_adam})")

    # Prompts
    prompts: list[torch.Tensor] = torch.load(cfg.prompts, weights_only=False)
    print(f"  loaded {len(prompts)} prompts from {cfg.prompts}")
    rng = random.Random(cfg.seed)
    train_prompts = prompts[: -cfg.eval_n_prompts]
    eval_prompts = prompts[-cfg.eval_n_prompts:]
    print(f"  train: {len(train_prompts)}  eval: {len(eval_prompts)}")

    metrics_file = open(out_dir / "metrics.jsonl", "a")

    def log(record: dict[str, Any]) -> None:
        record["wall_s"] = time.time() - t_start
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    def save_ckpt(step: int) -> None:
        # bf16 to halve disk (7.5B params * 4 → * 2 = 15 GB instead of 30 GB).
        # Precision loss is negligible for storing a trained student — we keep
        # the optimizer's fp32 shadow in memory; this is just the artifact.
        ckpt = {
            "step": step,
            "weight_fp": {
                name: m.weight_fp.detach().to(torch.bfloat16).cpu()
                for name, m in student.named_modules()
                if isinstance(m, IntLinear) and m.weight_fp is not None
            },
        }
        # Atomic write: write to .tmp then rename, so a torch.save crash
        # doesn't leave a half-baked file occupying disk.
        tmp_path = out_dir / f"step{step:05d}.pt.tmp"
        final_path = out_dir / f"step{step:05d}.pt"
        torch.save(ckpt, tmp_path)
        tmp_path.rename(final_path)
        # Delete any older step-N checkpoints to keep disk usage to one ckpt
        # at a time (8B + intermediate saves easily blow 80 GB vast disks).
        for old in out_dir.glob("step*.pt"):
            if old != final_path and "tmp" not in old.name:
                try:
                    old.unlink()
                except OSError:
                    pass

    t_start = time.time()
    # Pre-training eval (step 0) so we have a baseline to compare against.
    print(f"[{time.strftime('%H:%M:%S')}] step 0 eval")
    pre = evaluate(teacher, student, eval_prompts, device,
                   max_positions=cfg.eval_max_positions,
                   temperature=cfg.temperature, seed=cfg.seed)
    pre["step"] = 0; pre["lr"] = 0.0; pre["loss"] = float("nan")
    log(pre)
    print(f"  pre-train: top1={pre['all_top1']:.4f} margin_mean={pre['all_margin_mean']:.2e}")
    # Don't save step-0; it's the same as the model card. Saves ~15 GB of
    # disk on tight vast allocations.

    student.train()
    best_top1 = pre["all_top1"]
    plateau = 0
    plateau_limit = cfg.plateau_patience

    for step in range(1, cfg.steps + 1):
        # Sample a batch of prompts (with replacement to avoid epoch friction).
        batch_ids = [train_prompts[rng.randrange(len(train_prompts))] for _ in range(cfg.batch)]
        input_ids, mask = pad_collate(batch_ids, pad_id)
        input_ids = input_ids.to(device); mask = mask.to(device)

        if cfg.aux_weight > 0:
            teacher_acts.clear()
            student_acts.clear()
        with torch.inference_mode():
            ref_logits = teacher(input_ids).logits  # [B, T, V]
        # Need grad through student.
        student_logits = student(input_ids).logits  # [B, T, V]
        v = min(ref_logits.shape[-1], student_logits.shape[-1])
        diff = student_logits[..., :v].float() - ref_logits[..., :v].float()
        # Mask padding before mean so loss is averaged over real positions only.
        valid = mask[..., None].float()  # [B, T, 1]
        logit_loss = (diff.pow(2) * valid).sum() / valid.sum().clamp_min(1.0) / v

        aux_loss = torch.tensor(0.0, device=device)
        if cfg.aux_weight > 0 and teacher_acts and student_acts:
            common = set(teacher_acts) & set(student_acts)
            terms = []
            for name in common:
                t = teacher_acts[name].float()
                s = student_acts[name].float()
                # Match shapes (in rare cases shapes can differ slightly).
                if t.shape != s.shape:
                    continue
                # Normalized MSE per Linear, then mean across Linears.
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
            torch.nn.utils.clip_grad_norm_(
                [m.weight_fp for m in student.modules()
                 if isinstance(m, IntLinear) and m.weight_fp is not None],
                cfg.grad_clip,
            )
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
            ev["step"] = step
            ev["lr"] = lr_now
            ev["loss"] = loss.item()
            log(ev)
            print(
                f"  [eval @ {step}] top1={ev['all_top1']:.4f}  margin_mean={ev['all_margin_mean']:.2e}  "
                f"margin_p99={ev['all_margin_p99']:.2e}  kl_mean={ev['all_kl_mean']:.2e}"
            )
            student.train()
            if ev["all_top1"] > best_top1 + 1e-5:
                best_top1 = ev["all_top1"]
                plateau = 0
            else:
                plateau += 1
            if ev["all_top1"] >= 0.9999:
                print(f"  *** hit stretch target at step {step}; stopping")
                save_ckpt(step)
                break
            if plateau >= plateau_limit and step >= cfg.warmup + 3 * cfg.eval_every:
                print(f"  *** plateau ({plateau} no-improvements); stopping at step {step}")
                save_ckpt(step)
                break
        if step % cfg.ckpt_every == 0:
            save_ckpt(step)

    save_ckpt(step)
    # Symlink to final.pt
    final_path = out_dir / f"step{step:05d}.pt"
    final_link = out_dir / "final.pt"
    if final_link.exists() or final_link.is_symlink():
        final_link.unlink()
    final_link.symlink_to(final_path.name)
    metrics_file.close()
    print(f"done in {time.time() - t_start:.0f}s; best all_top1 = {best_top1:.4f}")


if __name__ == "__main__":
    main()
