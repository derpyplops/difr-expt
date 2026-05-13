"""Compute wikitext perplexity for fp8 teacher, int24 student, bf16 base.

PPL is the canonical "downstream quality" metric for language models. If the
int24 student's PPL is close to the bf16 base PPL (and close to the fp8
teacher's PPL), it's "performing similarly" in the way that matters for
inference.

For each model: cross-entropy of predicting token t+1 from prefix [0..t],
averaged across all positions. PPL = exp(mean CE).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from difr_expt.train_emulate import build_models


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
        ids = tokenizer(p, truncation=True, max_length=max_len, add_special_tokens=True).input_ids
        if len(ids) >= 32:
            out.append(torch.tensor(ids, dtype=torch.long))
        if len(out) >= n_prompts:
            break
    return out[:n_prompts]


@torch.inference_mode()
def model_ppl(model, prompts: list[torch.Tensor], device: str) -> tuple[float, int]:
    total_loss = 0.0
    total_tokens = 0
    for ids in prompts:
        input_ids = ids.to(device).unsqueeze(0)
        out = model(input_ids).logits  # [1, T, V]
        # Compare logits[t] to next-token labels[t+1]
        shift_logits = out[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    mean_loss = total_loss / max(total_tokens, 1)
    ppl = float(torch.tensor(mean_loss).exp())
    return ppl, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--teacher-id", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--weight-bits", type=int, default=24)
    ap.add_argument("--activation-bits", type=int, default=24)
    ap.add_argument("--rmsnorm-bits", type=int, default=24)
    ap.add_argument("--attn-matmul-bits", type=int, default=24)
    ap.add_argument("--softmax-lut-size", type=int, default=4096)
    ap.add_argument("--silu-lut-size", type=int, default=4096)
    ap.add_argument("--int-embedding", action="store_true")
    ap.add_argument("--no-patch-nonmatmul", action="store_true")
    ap.add_argument("--no-int-rmsnorm", action="store_true")
    ap.add_argument("--no-int-softmax", action="store_true")
    ap.add_argument("--no-int-silu", action="store_true")
    ap.add_argument("--no-int-attn-matmul", action="store_true")
    ap.add_argument("--no-int-rope", action="store_true")
    ap.add_argument("--activation-scheme", default="uniform",
                    choices=["uniform", "fp8_e4m3", "block_fp8_e4m3"],
                    help="Student activation quant grid: uniform int24 (default) or per-token fp8 e4m3 or per-128-block fp8 e4m3.")
    ap.add_argument("--softmax-x-min", type=float, default=-16.0)
    ap.add_argument("--init-from-base", action="store_true",
                    help="Build IntLinear weights from the bf16 base model directly "
                         "(no teacher fp8 round-trip). Default initializes from fp8 teacher.")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.base_model} / teacher={args.teacher_id} on {device} dtype={args.dtype}")

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
        softmax_x_min=args.softmax_x_min,
        silu_lut_size=args.silu_lut_size,
        attn_matmul_bits=args.attn_matmul_bits,
        trainable_matmul_weights=use_trainable,
        int_embedding=args.int_embedding,
        embedding_bits=24,
        int_lm_head=False,
        init_from_teacher=not args.init_from_base,
        keep_fp32_ref=True,
        patch_nonmatmul=False,  # we'll re-patch below with selective ops
        int_nonmatmul_bitexact=False,
    )
    if args.activation_scheme in ("fp8_e4m3", "block_fp8_e4m3"):
        from difr_expt.int_cast import IntLinear as _IL
        n_set = 0
        for m in student.modules():
            if isinstance(m, _IL):
                m.activation_scheme = args.activation_scheme
                m.use_int_matmul_path = False
                n_set += 1
        print(f"  set activation_scheme={args.activation_scheme} on {n_set} IntLinears")
    if not args.no_patch_nonmatmul:
        from difr_expt.patch_hf_model import IntOpsConfig as _Cfg, patch_model_int_nonmatmul as _patch
        ops_cfg = _Cfg(
            rmsnorm_bits=args.rmsnorm_bits,
            softmax_lut_size=args.softmax_lut_size,
            softmax_x_min=args.softmax_x_min,
            silu_lut_size=args.silu_lut_size,
            attn_matmul_bits=args.attn_matmul_bits,
            replace_rmsnorm=not args.no_int_rmsnorm,
            replace_softmax=not args.no_int_softmax,
            replace_silu=not args.no_int_silu,
            replace_attn_matmul=not args.no_int_attn_matmul,
            replace_rope=not args.no_int_rope,
        )
        counts = _patch(student, ops_cfg)
        print(f"  int non-matmul counts: {counts} (ablations: "
              f"no_rms={args.no_int_rmsnorm} no_sm={args.no_int_softmax} "
              f"no_silu={args.no_int_silu} no_attn={args.no_int_attn_matmul} no_rope={args.no_int_rope})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = load_wikitext_prompts(tokenizer, args.n_prompts, args.max_len)
    print(f"loaded {len(prompts)} prompts")

    # Reference with eager attention (apples-to-apples vs int_student which
    # forces eager when attention ops are wrapped). For Qwen2.5 some sizes,
    # eager-vs-sdpa drift is huge (8% on 7B), so this matters.
    from transformers import AutoModelForCausalLM as _ACL
    print("[loading] bf16 base with eager attention for apples-to-apples comparison")
    ref_eager = _ACL.from_pretrained(args.base_model, dtype=dtype, attn_implementation="eager").to(device).eval()

    results = {}
    for name, m in (("bf16_base", ref), ("bf16_base_eager", ref_eager), ("fp8_teacher", teacher), ("int_student", student)):
        t0 = time.time()
        ppl, n_tokens = model_ppl(m, prompts, device)
        wall = time.time() - t0
        results[name] = {"ppl": ppl, "n_tokens": n_tokens, "wall_s": wall}
        print(f"  {name}: ppl={ppl:.4f} ({n_tokens} tokens, {wall:.1f}s)")

    results["config"] = vars(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps(results, default=str) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
