"""Tokenize wikitext prompts once and save to disk.

Storing as a list of int64 token-id tensors is cheap (~5M tokens × 8 bytes ≈
40 MB at 10k prompts × 512 tokens) and removes the dataloader from the
training hot path. Save once per (tokenizer, n_prompts, max_len, seed)
triple and reuse across LR runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_wikitext(
    tokenizer_name: str,
    n_prompts: int,
    max_len: int,
    min_chars: int = 100,
    seed: int = 42,
) -> list[torch.Tensor]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )
    # Deterministic shuffle of the stream so seed=42 gives the same prompts
    # across runs. Streaming Dataset.shuffle uses a small reservoir buffer.
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    tensors: list[torch.Tensor] = []
    seen = 0
    for ex in ds:
        seen += 1
        text = ex["text"].strip()
        if len(text) < min_chars:
            continue
        ids = tok(text, truncation=True, max_length=max_len, add_special_tokens=True).input_ids
        if len(ids) < 32:  # too-short prompts add noise
            continue
        tensors.append(torch.tensor(ids, dtype=torch.long))
        if len(tensors) >= n_prompts:
            break
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n-prompts", type=int, default=10_000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    prompts = tokenize_wikitext(
        tokenizer_name=args.tokenizer,
        n_prompts=args.n_prompts,
        max_len=args.max_len,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompts, out)
    n_tokens = sum(t.numel() for t in prompts)
    print(
        f"wrote {len(prompts)} prompts ({n_tokens} tokens) to {out} "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
