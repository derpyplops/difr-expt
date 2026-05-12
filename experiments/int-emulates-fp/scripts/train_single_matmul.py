"""Train a SINGLE Linear's weight_fp to minimize the L2 of its output vs the
teacher's output at the same point in the network.

Everything else is frozen: same student forward, same teacher forward, only
this one Linear's weight is updated. Hooks capture the chosen layer's
(student, teacher) outputs each step.

If this fails to reduce L2, training itself is broken. If it succeeds, the
issue with full-network training is elsewhere (joint dynamics, generalization,
overfitting to specific activation patterns, etc.).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from difr_expt.train_emulate import build_models, DTYPE_MAP, IntLinear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--teacher-source", default="published")
    ap.add_argument("--teacher-id", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--target-module", default="model.layers.12.mlp.gate_proj",
                    help="Name of the Linear to train (everything else frozen)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[args.dtype]
    print(f"loading on {device}")
    teacher, student, _ = build_models(
        model_name=args.model,
        teacher_source=args.teacher_source,
        teacher_id=args.teacher_id,
        teacher_precision="fp8_e4m3", teacher_block_size=32, teacher_quantize_act=True,
        dtype=dtype, device=device,
        weight_bits=24, activation_bits=24, rmsnorm_bits=24,
        softmax_lut_size=4096, softmax_x_min=-16.0, silu_lut_size=4096,
        attn_matmul_bits=24,
        trainable_matmul_weights=True,
        int_embedding=False, embedding_bits=24, int_lm_head=False,
        init_from_teacher=True, keep_fp32_ref=False, grad_checkpointing=False,
        patch_nonmatmul=True,
    )
    teacher.eval(); student.eval()

    # Find the target module and freeze everything else.
    target_mod = dict(student.named_modules()).get(args.target_module)
    if not isinstance(target_mod, IntLinear):
        raise SystemExit(f"target {args.target_module!r} is not an IntLinear; got {type(target_mod).__name__}")
    if target_mod.weight_fp is None:
        raise SystemExit(f"target {args.target_module!r} has no weight_fp (trainable=False?)")
    print(f"  training module: {args.target_module}")
    print(f"    weight_fp shape: {tuple(target_mod.weight_fp.shape)}  device={target_mod.weight_fp.device}")

    # Freeze everything; unfreeze only the target's weight_fp.
    for p in student.parameters():
        p.requires_grad = False
    target_mod.weight_fp.requires_grad = True
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")

    # Hook teacher's matching module AND student's target to capture outputs.
    teacher_mod = dict(teacher.named_modules()).get(args.target_module)
    if teacher_mod is None or not isinstance(teacher_mod, nn.Linear):
        raise SystemExit(f"teacher has no matching {args.target_module} (got {type(teacher_mod).__name__})")
    s_out: dict = {}; t_out: dict = {}
    def hook(store):
        def _h(_m, _in, output):
            store["v"] = output if isinstance(output, torch.Tensor) else output[0]
        return _h
    target_mod.register_forward_hook(hook(s_out))
    teacher_mod.register_forward_hook(hook(t_out))

    optimizer = torch.optim.AdamW([target_mod.weight_fp], lr=args.lr, weight_decay=0.0)

    prompts: list[torch.Tensor] = torch.load(args.prompts, weights_only=False)
    # 80 train / 20 eval split, same as our other experiments.
    train_prompts = prompts[:80]
    eval_prompts = prompts[-20:]
    import random
    rng = random.Random(42)
    print(f"  train={len(train_prompts)}  eval={len(eval_prompts)}")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(out_path, "w")

    def pad_collate(batch_ids, pad_id=0):
        max_len = max(t.numel() for t in batch_ids)
        ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
        for i, t in enumerate(batch_ids):
            ids[i, : t.numel()] = t
        return ids

    @torch.inference_mode()
    def measure_l2(prompts_subset):
        l2s = []
        for ids in prompts_subset:
            s_out.clear(); t_out.clear()
            x = ids.to(device).unsqueeze(0)
            teacher(x)
            student(x)
            if "v" in s_out and "v" in t_out and s_out["v"].shape == t_out["v"].shape:
                diff = s_out["v"].float() - t_out["v"].float()
                l2s.append(diff.norm(dim=-1).mean().item())
        return sum(l2s) / max(1, len(l2s))

    initial_train = measure_l2(train_prompts)
    initial_eval = measure_l2(eval_prompts)
    print(f"  step 0: train L2 = {initial_train:.4f}  eval L2 = {initial_eval:.4f}")
    log.write(json.dumps({"step": 0, "train_l2_full": initial_train, "eval_l2": initial_eval}) + "\n")
    log.flush()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch_ids = [train_prompts[rng.randrange(len(train_prompts))] for _ in range(args.batch)]
        input_ids = pad_collate(batch_ids).to(device)
        s_out.clear(); t_out.clear()
        with torch.inference_mode():
            teacher(input_ids)
        student(input_ids)
        s = s_out["v"].float()
        t = t_out["v"].float().detach()
        diff = s - t
        loss = diff.pow(2).mean()
        # Per-position L2 norm (matches eval).
        train_l2 = diff.norm(dim=-1).mean().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([target_mod.weight_fp], 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            tr_l2 = measure_l2(train_prompts)
            ev_l2 = measure_l2(eval_prompts)
            print(f"  step {step:5d}  train_full L2={tr_l2:.4f}  eval L2={ev_l2:.4f}  step_batch L2={train_l2:.4f}  loss={loss.item():.4e}")
            log.write(json.dumps({"step": step, "train_l2_full": tr_l2, "eval_l2": ev_l2, "step_batch_l2": train_l2, "loss": loss.item()}) + "\n")
            log.flush()

    final_eval = measure_l2(eval_prompts)
    print(f"\ndone in {time.time()-t0:.0f}s")
    print(f"  initial eval L2: {initial_eval:.4f}")
    print(f"  final   eval L2: {final_eval:.4f}  (Δ = {(final_eval-initial_eval)/initial_eval*100:+.1f}%)")
    log.close()


if __name__ == "__main__":
    main()
