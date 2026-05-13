"""Capture per-matmul records from a fake-FP8-quant'd model on wikitext.

For each prompt, we run a forward pass through a `LowPrecisionLinear`-patched
model and hook every replaced Linear. For every call we save:

  - input  X_fp32 (the dequantized fp8 activation as the kernel saw it)
  - weight W_fp32 (the dequantized fp8 weight, frozen at construction)
  - bias   (optional, fp32) — stripped from Y so the residual is matmul-only
  - output Y_fp32 (matmul-only, bias subtracted off)
  - per-token activation scale s_X  (shape [T])
  - per-row   weight   scale  s_W  (shape [D])
  - layer name, prompt index, matmul family (q/k/v/o/gate/up/down)

We save records as a flat list of dicts keyed by integers. Each `.pt` file is
one prompt's worth of records across all matmuls — keeps file sizes small
and lets us train/val-split at the prompt level cheaply.

Default config targets Qwen2.5-0.5B. Per prompt ≈ T·D cells per matmul
× ~7 matmuls per block × 24 blocks ≈ a few hundred million bytes if stored
as fp32. We store X, Y, W in float16 to keep things manageable; the fp8 grid
is representable in fp16 exactly (fp16 has 11 mantissa bits, fp8 e4m3 has 4,
so casting fp8-dequant fp32 → fp16 is lossless).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from difr_expt.fp_quant import (
    LowPrecisionLinear,
    fake_quantize_fp8,
    patch_model_low_precision,
)


# Map a Linear's qualified name to a coarse "matmul family" label.
# Qwen/Llama-style transformer block names. We treat embed/lm_head as 'other'.
def matmul_family(name: str) -> str:
    if "q_proj" in name:
        return "q"
    if "k_proj" in name:
        return "k"
    if "v_proj" in name:
        return "v"
    if "o_proj" in name:
        return "o"
    if "gate_proj" in name:
        return "gate"
    if "up_proj" in name:
        return "up"
    if "down_proj" in name:
        return "down"
    return "other"


def block_index_from_name(name: str) -> int:
    # Qwen/Llama: model.layers.{i}.<sub>.<linear>
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        return int(parts[i + 1])
    return -1


@dataclass
class CaptureRec:
    name: str          # module qual name
    family: str        # q/k/v/o/gate/up/down/other
    block: int         # transformer block index or -1
    prompt: int        # prompt index
    X_q: torch.Tensor  # [T, K] fp16  (dequant fp8 activation as seen by kernel)
    W_q: torch.Tensor  # [D, K] fp16  (dequant fp8 weight)
    Y: torch.Tensor    # [T, D] fp32 (matmul output, bias subtracted)
    s_X: torch.Tensor  # [T]   fp32   (per-token absmax/448)
    s_W: torch.Tensor  # [D]   fp32   (per-row   absmax/448, constant per record)
    has_bias: bool


def install_hooks(
    model: nn.Module
) -> tuple[list[CaptureRec], list, dict, dict]:
    """Wrap each LowPrecisionLinear with a forward that captures the record.

    Returns (records, handles, cur_prompt, weights_dict).

    weights_dict[name] = {"W_q": fp16 [D, K], "s_W": fp32 [D], "has_bias": bool}
    Saved once per model, NOT once per prompt — weights are constant across
    prompts so storing per-record blows up disk by 100×.
    """
    records: list[CaptureRec] = []
    handles: list = []  # store (module, original_forward) for unwrap

    cur_prompt = {"idx": 0}
    weights: dict[str, dict] = {}

    def make_wrapper(mod: LowPrecisionLinear, name: str):
        orig_forward = mod.forward
        fam = matmul_family(name)
        blk = block_index_from_name(name)

        # Precompute the per-row weight scale once. LowPrecisionLinear
        # stored only the dequantized weight, so we re-derive s_W from it.
        # Per-row absmax / 448 (same recipe as fake_quantize_fp8 with per_row=True).
        W_dq = mod.weight.detach().to(torch.float32)
        fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
        s_W = W_dq.abs().amax(dim=-1).clamp_min(1e-30) / fp8_max

        # Stash weights in the dedicated dict.
        weights[name] = {
            "W_q": W_dq.to(torch.float16).cpu(),
            "s_W": s_W.to(torch.float32).cpu(),
            "has_bias": mod.bias is not None,
            "family": fam,
            "block": blk,
        }

        def wrapped_forward(x: torch.Tensor) -> torch.Tensor:
            # Compute the per-token activation scale exactly the way
            # `fake_quantize_fp8(..., per_row=True, axis=-1)` does for [..., K]
            # tensors. The function uses absmax along axis=-1 → per-token.
            if mod.quantize_act:
                x_fp32 = x.detach().to(torch.float32)
                # Per-token absmax across the last dim:
                s_X = x_fp32.abs().amax(dim=-1, keepdim=False).clamp_min(1e-30) / fp8_max
                # Compute x_q exactly the way the original module does:
                x_q = fake_quantize_fp8(x, torch.float8_e4m3fn, per_row=True, axis=-1)
            else:
                x_q = x
                s_X = torch.ones(x.shape[:-1], device=x.device, dtype=torch.float32)

            # Run the real forward to get Y (so the model's logits stay correct).
            out = orig_forward(x)

            # Subtract the bias to get pure-matmul Y. (Stays differentiable in
            # the autograd graph; that's fine — we're in inference_mode anyway.)
            if mod.bias is not None:
                Y = out - mod.bias
            else:
                Y = out

            # Reshape inputs to flat 2D [T, K] and 2D [T, D]:
            x_q_2d = x_q.reshape(-1, x_q.shape[-1])
            Y_2d = Y.reshape(-1, Y.shape[-1])
            s_X_flat = s_X.reshape(-1)

            rec = CaptureRec(
                name=name,
                family=fam,
                block=blk,
                prompt=cur_prompt["idx"],
                X_q=x_q_2d.detach().to(torch.float16).cpu(),
                W_q=torch.empty(0),       # placeholder — read from shared file
                Y=Y_2d.detach().to(torch.float32).cpu(),
                s_X=s_X_flat.detach().to(torch.float32).cpu(),
                s_W=torch.empty(0),       # placeholder
                has_bias=mod.bias is not None,
            )
            records.append(rec)
            return out

        # Store original so we can unwrap if needed.
        handles.append((mod, orig_forward))
        mod.forward = wrapped_forward

    for name, m in list(model.named_modules()):
        if isinstance(m, LowPrecisionLinear):
            make_wrapper(m, name)

    return records, handles, cur_prompt, weights


def restore_forwards(handles: list) -> None:
    for mod, orig in handles:
        mod.forward = orig


def load_wikitext_prompts(tokenizer, n_prompts: int, max_len: int) -> list[torch.Tensor]:
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    texts: list[str] = []
    for ex in ds:
        t = ex["text"].strip()
        if len(t) >= 200:
            texts.append(t)
        if len(texts) >= n_prompts * 2:
            break
    out: list[torch.Tensor] = []
    for t in texts:
        ids = tokenizer(
            t, truncation=True, max_length=max_len, add_special_tokens=True
        ).input_ids
        if len(ids) >= 32:
            out.append(torch.tensor(ids, dtype=torch.long))
        if len(out) >= n_prompts:
            break
    return out[:n_prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--precision", default="fp8_e4m3",
                    choices=["fp8_e4m3", "fp8_e5m2", "fp4_e2m1"])
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cpu")
    # Capture filtering: e.g. only certain blocks/families to keep size sane
    ap.add_argument("--blocks", default="",
                    help="comma-separated block indices to keep, '' = all")
    ap.add_argument("--families", default="",
                    help="comma-separated family labels (q,k,v,o,gate,up,down), '' = all")
    ap.add_argument("--out", required=True, help="output dir for per-prompt .pt files")
    ap.add_argument("--max-records-per-prompt", type=int, default=0,
                    help="0 = all; non-zero caps total records per prompt for smoke runs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keep_blocks: set[int] | None = (
        set(int(x) for x in args.blocks.split(",") if x.strip()) if args.blocks else None
    )
    keep_families: set[str] | None = (
        set(args.families.split(",")) if args.families else None
    )

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = args.device

    print(f"Loading {args.model} @ {dtype} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Patching with LowPrecisionLinear (fp8-fake-quant)...")
    n = patch_model_low_precision(
        model, precision=args.precision, block_size=32, include_lm_head=False,
        quantize_act=True,
    )
    print(f"  patched {len(n)} Linears")

    print("Installing capture hooks...")
    records, handles, cur_prompt, weights = install_hooks(model)
    # Save the shared weights once.
    weights_path = out / "weights.pt"
    torch.save(weights, weights_path)
    print(f"  saved shared weights → {weights_path.name} "
          f"({sum(v['W_q'].numel() * 2 for v in weights.values()) / 1e6:.1f} MB)")

    print(f"Loading {args.n_prompts} wikitext prompts @ max_len={args.max_len}...")
    prompts = load_wikitext_prompts(tokenizer, args.n_prompts, args.max_len)
    print(f"  loaded {len(prompts)} prompts; lengths={[p.numel() for p in prompts]}")

    saved = []
    t0 = time.time()
    for i, ids in enumerate(prompts):
        cur_prompt["idx"] = i
        records.clear()
        with torch.inference_mode():
            _ = model(ids.to(device).unsqueeze(0))
        # Filter
        kept = []
        for r in records:
            if keep_blocks is not None and r.block not in keep_blocks:
                continue
            if keep_families is not None and r.family not in keep_families:
                continue
            kept.append(r)
            if args.max_records_per_prompt and len(kept) >= args.max_records_per_prompt:
                break
        out_path = out / f"prompt_{i:04d}.pt"
        torch.save({
            "model": args.model,
            "precision": args.precision,
            "dtype": args.dtype,
            "prompt_idx": i,
            "input_ids": ids,
            "records": [
                {
                    "name": r.name,
                    "family": r.family,
                    "block": r.block,
                    "prompt": r.prompt,
                    "X_q": r.X_q,
                    # W_q, s_W intentionally omitted — load from weights.pt
                    "Y":   r.Y,
                    "s_X": r.s_X,
                    "has_bias": r.has_bias,
                } for r in kept
            ],
        }, out_path)
        saved.append(str(out_path))
        elapsed = time.time() - t0
        n_cells = sum(r.Y.numel() for r in kept)
        print(f"  prompt {i}: {len(kept)} records, {n_cells} output cells, "
              f"saved {out_path.name} ({elapsed:.1f}s elapsed)")

    restore_forwards(handles)

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({
        "model": args.model,
        "precision": args.precision,
        "dtype": args.dtype,
        "n_prompts": len(saved),
        "files": [Path(p).name for p in saved],
        "blocks_kept": sorted(keep_blocks) if keep_blocks else "all",
        "families_kept": sorted(keep_families) if keep_families else "all",
    }, indent=2))
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
