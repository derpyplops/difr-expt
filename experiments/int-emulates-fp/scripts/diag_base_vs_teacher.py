"""Diagnostic: compare base model (after cast-from-teacher) directly to the
teacher, with no int patching at all. Tells us the irreducible gap between
the unquantized base + cast weights and the published fp8 teacher."""

from __future__ import annotations

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def cast_teacher_weights_into_base(teacher, base):
    """Dequantize teacher's fp8 weights and copy into base. Handles:

    - CompressedLinear (RedHatAI per-row fp8): weight (e4m3) * weight_scale.
    - FP8Linear (DeepSeek / Qwen3 block fp8): weight (e4m3) * weight_scale_inv,
      where scale is one scalar per [block_h, block_w] tile (typically 128x128).
    - Plain nn.Linear with non-fp8 weights (e.g. lm_head left in bf16).

    Critical: must check fp8 dtype BEFORE falling through to plain-Linear branch.
    FP8Linear is a torch.nn.Linear subclass, so isinstance check is not enough.
    """
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
    except ImportError:
        CompressedLinear = type(None)
    teacher_modules = dict(teacher.named_modules())
    n = 0
    n_block = 0
    n_perrow = 0
    n_plain = 0
    for name, bm in base.named_modules():
        if not isinstance(bm, torch.nn.Linear):
            continue
        tm = teacher_modules.get(name)
        if tm is None:
            continue
        # Block fp8: weight is float8_e4m3fn + weight_scale_inv [bh, bw].
        if (hasattr(tm, "weight_scale_inv")
                and getattr(tm.weight, "dtype", None) == torch.float8_e4m3fn):
            W = tm.weight.to(torch.float32)
            S = tm.weight_scale_inv.to(torch.float32)
            bh = W.shape[0] // S.shape[0]
            bw = W.shape[1] // S.shape[1]
            Sexp = S.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
            w_fp = W * Sexp
            n_block += 1
        # Per-row fp8: CompressedLinear with weight + weight_scale [out, 1].
        elif isinstance(tm, CompressedLinear):
            w_fp = tm.weight.float() * tm.weight_scale.float()
            n_perrow += 1
        # Plain Linear (e.g. lm_head) at high precision.
        elif (isinstance(tm, torch.nn.Linear)
                and getattr(tm.weight, "dtype", None) != torch.float8_e4m3fn):
            w_fp = tm.weight.detach().float()
            n_plain += 1
        else:
            continue
        if w_fp.shape != bm.weight.shape:
            continue
        bm.weight.data.copy_(w_fp.to(bm.weight.dtype))
        n += 1
    print(f"  copied {n} weight tensors from teacher into base "
          f"(block-fp8={n_block}, per-row-fp8={n_perrow}, plain={n_plain})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--teacher-id", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading base and teacher in {dtype}")
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_id, dtype=dtype).to(device).eval()
    cast_teacher_weights_into_base(teacher, base)

    prompts: list[torch.Tensor] = torch.load(args.prompts, weights_only=False)
    eval_prompts = prompts[-args.n_prompts:]
    print(f"  evaluating on {len(eval_prompts)} held-out prompts")

    n_pos = 0
    n_match = 0
    kl_sum = 0.0
    kls = []
    with torch.inference_mode():
        for ids in eval_prompts:
            input_ids = ids.to(device).unsqueeze(0)
            tl = teacher(input_ids).logits[0].float()
            bl = base(input_ids).logits[0].float()
            v = min(tl.shape[-1], bl.shape[-1])
            tl = tl[..., :v]; bl = bl[..., :v]
            n_match += (tl.argmax(-1) == bl.argmax(-1)).sum().item()
            n_pos += tl.shape[0]
            log_pt = torch.log_softmax(tl, dim=-1)
            log_pb = torch.log_softmax(bl, dim=-1)
            kl_per = (log_pt.exp() * (log_pt - log_pb)).sum(-1)
            kls.append(kl_per.cpu())
    top1 = n_match / max(1, n_pos)
    kl = torch.cat(kls)
    print(f"top1 (base-with-cast vs teacher) = {top1:.4f}  "
          f"kl_mean={kl.mean().item():.4e}  kl_p99={kl.quantile(0.99).item():.4e}  "
          f"n_positions={n_pos}")


if __name__ == "__main__":
    main()
