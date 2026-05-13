"""Eval: int-cast student vs FP8 production teacher (no training).

Loads a published fp8 model as teacher, builds an int24 student
(IntLinear + int non-matmul + IntEmbedding) initialized from teacher
weights via `build_models(init_from_teacher=True)`, then runs the same
100-prompt wikitext eval used by baseline-int-cast / train-nonmatmul-int.

Output: a single jsonl line with student_vs_teacher/* metrics (top1,
margin_mean, margin_p99, kl_*, logit_*, n_positions).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from difr_expt.train_emulate import build_models, evaluate


def load_wikitext_prompts(tokenizer, n_prompts: int, max_len: int) -> list[torch.Tensor]:
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts: list[str] = []
    for ex in ds:
        text = ex["text"].strip()
        if len(text) >= 100:
            prompts.append(text)
        if len(prompts) >= n_prompts * 2:
            break
    out: list[torch.Tensor] = []
    for p in prompts:
        ids = tokenizer(
            p, truncation=True, max_length=max_len, add_special_tokens=True
        ).input_ids
        if len(ids) >= 32:
            out.append(torch.tensor(ids, dtype=torch.long))
        if len(out) >= n_prompts:
            break
    return out[:n_prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="HF id of bf16 base model (e.g. Qwen/Qwen2.5-0.5B)")
    ap.add_argument("--teacher-id", required=True,
                    help="HF id of fp8 production teacher (e.g. RedHatAI/Qwen2.5-0.5B-FP8-dynamic)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--weight-bits", type=int, default=24)
    ap.add_argument("--activation-bits", type=int, default=24)
    ap.add_argument("--rmsnorm-bits", type=int, default=24)
    ap.add_argument("--attn-matmul-bits", type=int, default=24)
    ap.add_argument("--softmax-lut-size", type=int, default=4096)
    ap.add_argument("--silu-lut-size", type=int, default=4096)
    ap.add_argument("--int-embedding", action="store_true")
    ap.add_argument("--embedding-bits", type=int, default=24)
    ap.add_argument("--int-lm-head", action="store_true",
                    help="Also quantize lm_head (teacher leaves it fp).")
    ap.add_argument("--activation-scheme", default="uniform",
                    choices=["uniform", "fp8_e4m3", "block_fp8_e4m3"],
                    help="Student activation quant: uniform int24 (default) or per-token fp8 e4m3 or per-128-block fp8 e4m3 (matches teacher's block-fp8 dynamic exactly).")
    ap.add_argument("--block-fp8-kernel-path", action="store_true",
                    help="Use teacher's Triton w8a8_block_fp8_matmul kernel on stashed fp8 weights. Required for Qwen3-style block-fp8 teachers.")
    ap.add_argument("--no-patch-nonmatmul", action="store_true",
                    help="Skip non-matmul int patching (Linears still int-cast).")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-fp32-ref", action="store_true",
                    help="Also report student_vs_ref / teacher_vs_ref where ref = bf16 base model. Triples memory; off by default.")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.base_model} / teacher={args.teacher_id} on {device} dtype={args.dtype}")

    # build_models constructs IntLinear with trainable=False (the default for
    # this eval). To allow setting activation_scheme=fp8_e4m3, IntLinear's
    # _trainable_forward path is the one that supports it — so flip trainable=True
    # only when fp8 act scheme is requested (the weight tensors are still frozen
    # via no_grad; trainable just enables the alternate forward path).
    use_trainable = args.activation_scheme in ("fp8_e4m3", "block_fp8_e4m3")
    teacher, student, ref = build_models(
        model_name=args.base_model,
        teacher_source="published",
        teacher_id=args.teacher_id,
        teacher_precision="fp8_e4m3",
        teacher_block_size=128,
        teacher_quantize_act=True,
        dtype=dtype,
        device=device,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        rmsnorm_bits=args.rmsnorm_bits,
        softmax_lut_size=args.softmax_lut_size,
        softmax_x_min=-16.0,
        silu_lut_size=args.silu_lut_size,
        attn_matmul_bits=args.attn_matmul_bits,
        trainable_matmul_weights=use_trainable,
        int_embedding=args.int_embedding,
        embedding_bits=args.embedding_bits,
        int_lm_head=args.int_lm_head,
        init_from_teacher=True,
        keep_fp32_ref=args.keep_fp32_ref,
        patch_nonmatmul=not args.no_patch_nonmatmul,
        int_nonmatmul_bitexact=False,
    )

    if args.activation_scheme in ("fp8_e4m3", "block_fp8_e4m3"):
        from difr_expt.int_cast import IntLinear as _IL
        n_set = 0
        for n, m in student.named_modules():
            if isinstance(m, _IL):
                m.activation_scheme = args.activation_scheme
                m.use_int_matmul_path = False
                if args.block_fp8_kernel_path:
                    m.use_block_fp8_kernel_path = True
                n_set += 1
        print(f"  set activation_scheme={args.activation_scheme} on {n_set} IntLinears"
              f" (block_fp8_kernel_path={args.block_fp8_kernel_path})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = load_wikitext_prompts(tokenizer, args.n_prompts, args.max_len)
    print(f"loaded {len(prompts)} prompts")

    t0 = time.time()
    metrics = evaluate(
        teacher=teacher,
        student=student,
        eval_prompts=prompts,
        device=device,
        max_positions=10**9,
        temperature=1.0,
        seed=args.seed,
        ref_model=ref,
    )
    wall = time.time() - t0
    metrics["wallclock_s"] = wall
    metrics["config"] = {
        "base_model": args.base_model,
        "teacher_id": args.teacher_id,
        "dtype": args.dtype,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "rmsnorm_bits": args.rmsnorm_bits,
        "attn_matmul_bits": args.attn_matmul_bits,
        "softmax_lut_size": args.softmax_lut_size,
        "silu_lut_size": args.silu_lut_size,
        "int_embedding": args.int_embedding,
        "embedding_bits": args.embedding_bits,
        "int_lm_head": args.int_lm_head,
        "patch_nonmatmul": not args.no_patch_nonmatmul,
        "n_prompts": args.n_prompts,
        "max_len": args.max_len,
    }
    print("\n=== student_vs_teacher ===")
    for k, v in metrics.items():
        if k.startswith("student_vs_teacher/"):
            print(f"  {k}: {v}")
    print(f"  wallclock_s: {wall:.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(metrics, default=str) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
