"""Cache a tiny set of wikitext prompts for the smoke test.

This is the same idea as `difr_expt.cache_prompts` but constrained to a handful
of short prompts so the smoke test (CPU, tiny model) runs in seconds.
"""
from __future__ import annotations

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True)
        tensors: list[torch.Tensor] = []
        for ex in ds:
            text = ex["text"].strip()
            if len(text) < 80:
                continue
            ids = tok(text, truncation=True, max_length=args.max_len,
                      add_special_tokens=True).input_ids
            if len(ids) < 8:
                continue
            tensors.append(torch.tensor(ids, dtype=torch.long))
            if len(tensors) >= args.n:
                break
    except Exception as e:
        print(f"[warn] dataset load failed: {e}; using toy prompts")
        seeds = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "To compute the area of a circle, multiply pi by the radius squared.",
            "Mathematics is the language with which God has written the universe.",
            "The only way to do great work is to love what you do.",
            "Once upon a time in a faraway kingdom there lived a curious mathematician.",
            "All happy families are alike each unhappy family is unhappy in its own way.",
            "It was the best of times it was the worst of times.",
        ]
        tensors = []
        for s in seeds[: args.n]:
            ids = tok(s, truncation=True, max_length=args.max_len,
                      add_special_tokens=True).input_ids
            tensors.append(torch.tensor(ids, dtype=torch.long))

    torch.save(tensors, args.out)
    print(f"wrote {len(tensors)} prompts to {args.out}")


if __name__ == "__main__":
    main()
