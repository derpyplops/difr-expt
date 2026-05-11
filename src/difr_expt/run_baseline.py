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

from difr_expt.int_cast import patch_model_int_cast, set_true_int_matmul
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
    dataset: str = "Salesforce/wikitext"
    dataset_config: str | None = "wikitext-103-raw-v1"
    dataset_split: str = "train"
    temperature: float = 1.0
    seed: int = 42
    device: str = "auto"
    out: str | None = None
    include_lm_head: bool = True
    skip_substrings: list[str] = field(default_factory=list)


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
    """Run both models on each prompt; accumulate per-position metric tensors."""
    all_top1 = []
    all_top5 = []
    all_l2 = []
    all_kl = []
    all_margin = []
    n_positions = 0

    rng = torch.Generator(device=device).manual_seed(seed)

    for ids in tqdm(prompts, desc="prompts"):
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        ref_logits = per_prompt_logits(ref_model, input_ids)  # [T, V]
        int_logits = per_prompt_logits(int_model, input_ids)
        # Pre-prompt position 0 has no meaningful "next token" target in a teacher-
        # forced setup against the prompt itself, but we still compare logits at
        # every position because that's where divergence accumulates.

        # Match vocab sizes (some models pad lm_head; both copies share the same).
        v = min(ref_logits.shape[-1], int_logits.shape[-1])
        ref_logits = ref_logits[..., :v]
        int_logits = int_logits[..., :v]

        all_top1.append(top1_match(ref_logits, int_logits))
        all_top5.append(topk_overlap(ref_logits, int_logits, k=5))
        all_l2.append(logit_l2(ref_logits, int_logits))
        all_kl.append(kl_div_ref_to_cand(ref_logits, int_logits, temperature=temperature))

        # Gumbel noise per position (1, V).
        u = torch.empty_like(ref_logits, dtype=torch.float32)
        u.uniform_(1e-10, 1.0, generator=rng)
        gumbel = -torch.log(-torch.log(u))
        all_margin.append(post_gumbel_margin(ref_logits, int_logits, gumbel, temperature=temperature))

        n_positions += ref_logits.shape[0]

    metrics = {
        "top1_match": torch.cat(all_top1),
        "top5_overlap": torch.cat(all_top5),
        "logit_l2": torch.cat(all_l2),
        "kl_ref_cand": torch.cat(all_kl),
        "gumbel_margin": torch.cat(all_margin),
    }
    summary = aggregate(metrics)
    summary["n_positions"] = n_positions
    summary["n_prompts"] = len(prompts)

    # Percentile reporting for the metrics we care about most.
    for k in ("logit_l2", "kl_ref_cand", "gumbel_margin"):
        t = metrics[k].float()
        summary[f"{k}_p50"] = t.quantile(0.5).item()
        summary[f"{k}_p99"] = t.quantile(0.99).item()
        summary[f"{k}_max"] = t.max().item()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--weight-bits", type=int, default=16)
    ap.add_argument("--activation-bits", type=int, default=16)
    ap.add_argument("--true-int-matmul", action="store_true",
                    help="Use int32@int32->int64 matmul instead of dequant-then-fp32. CPU fallback on CUDA, slow.")
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
    args = ap.parse_args()

    cfg = Config(
        model=args.model,
        dtype=args.dtype,
        n_prompts=args.n_prompts,
        max_len=args.max_len,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        true_int_matmul=args.true_int_matmul,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        out=args.out,
        include_lm_head=not args.skip_lm_head,
        skip_substrings=list(args.skip_substring),
    )
    device = pick_device(cfg.device)
    dtype = DTYPE_MAP[cfg.dtype]
    print(f"loading {cfg.model} ({cfg.dtype}) on {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    ref = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(device).eval()
    int_model = copy.deepcopy(ref)
    replaced = patch_model_int_cast(
        int_model,
        weight_bits=cfg.weight_bits,
        activation_bits=cfg.activation_bits,
        skip_names=tuple(cfg.skip_substrings),
        include_lm_head=cfg.include_lm_head,
    )
    if cfg.true_int_matmul:
        set_true_int_matmul(int_model, True)
    print(
        f"replaced {len(replaced)} Linear modules with IntLinear "
        f"(w_bits={cfg.weight_bits}, a_bits={cfg.activation_bits}, true_int={cfg.true_int_matmul})"
    )

    prompts = load_prompts(cfg, tokenizer)
    print(f"loaded {len(prompts)} prompts")

    t0 = time.time()
    summary = compare(ref, int_model, prompts, device=device, temperature=cfg.temperature, seed=cfg.seed)
    summary["wallclock_s"] = time.time() - t0
    summary["config"] = asdict(cfg)
    summary["replaced_module_count"] = len(replaced)

    print("\n=== summary ===")
    for k, v in summary.items():
        if k == "config":
            continue
        print(f"  {k}: {v}")

    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
