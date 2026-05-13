"""Turn fit_residual_models.py JSON output into a markdown results table.

Two views:
  - pooled across all matmul families
  - per-family breakout

Columns: model, abs_mean, abs_p50, abs_p99, abs_p99.9, abs_worst,
         signed_mean, bit_exact%, proof_cost (qualitative).

Proof-cost mapping (per-cell ops) is hard-coded per model name. Update
this dict if you add new model variants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROOF_COST = {
    "R1":  "0 ops (no correction)",
    "R2":  "1 add (global constant)",
    "R3":  "1 lookup (per-family)",
    "R4":  "1 mult + 1 add (a + b·Y')",
    "R5":  "1 lookup per (family,block,d)",
    "R7":  "2 mult (α · s_X · s_W · Y')",
    "R7b": "3 mult + 2 add (affine in s_X·s_W·Y', Y')",
    "R8":  "5 mult + 4 add over (sum_pos, sum_neg, Y', s·Y') — sums are K-sums",
    "R9":  "6 mult + 5 add over (Y', s·Y', clip counts, absmax)",
    "R10": "10 mult + 9 add over per-row/per-col δ summaries + cross-products",
    "R10b":"per-family R10 (≤10 coefs × 7 families)",
    "R11": "3 K-sum matmuls per cell (X'·δ_W, W'·δ_X, δ_X·δ_W) + 3 mult + 3 add",
    "R11a":"2 K-sum matmuls per cell (drops δ·δ) + 2 mult + 2 add",
    "R11_fixed": "Same as R11 but coefs fixed to (1,1,-1); no learned params",
    "R11_per_family": "Same as R11 but per-family coefs (4 × 7 = 28 numbers)",
    "R12": "2 K-sum sign-matmuls (X'·sign(δ_W), sign(δ_X)·W') — 1-bit δ commit",
}


def fmt(x, prec=2):
    if isinstance(x, int):
        return str(x)
    return f"{x:.{prec}f}"


def write_pooled(rows, out):
    header = "| Model | abs_mean | p50 | p99 | p99.9 | worst | signed_mean | bit_exact% | Proof cost |"
    sep    = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        m = r["model"]
        lines.append(
            f"| {m} | {fmt(r['abs_mean'])} | {fmt(r['abs_p50'], 0)} | "
            f"{fmt(r['abs_p99'], 0)} | {fmt(r['abs_p99.9'], 0)} | "
            f"{r['abs_worst']} | {fmt(r['signed_mean'])} | "
            f"{100*r['bit_exact_frac']:.2f}% | "
            f"{PROOF_COST.get(m, '?')} |"
        )
    out.write("\n".join(lines) + "\n")


def write_per_family(rows, out):
    fams = ["q", "k", "v", "o", "gate", "up", "down"]
    out.write("\n### Per-matmul-family breakouts\n\n")
    for fam in fams:
        out.write(f"\n#### family = {fam}\n\n")
        header = "| Model | abs_mean | abs_p99 | worst | signed_mean | bit_exact% |"
        sep    = "|---|---|---|---|---|---|"
        out.write(header + "\n" + sep + "\n")
        for r in rows:
            f = r.get("per_family", {}).get(fam)
            if not f:
                continue
            out.write(
                f"| {r['model']} | {fmt(f['abs_mean'])} | "
                f"{fmt(f['abs_p99'])} | {f['abs_worst']} | "
                f"{fmt(f['signed_mean'])} | "
                f"{100*f['bit_exact_frac']:.2f}% |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Residual model results")
    args = ap.parse_args()

    blob = json.load(open(args.fit_json))
    rows = blob["results"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# {args.title}\n\n")
        f.write(f"- Data: `{blob['data_dir']}`\n")
        f.write(f"- Train prompts: {blob['train_prompts']}\n")
        f.write(f"- Val prompts: {blob['val_prompts']}\n")
        n = rows[0].get("n_val_cells") if rows else None
        if n:
            f.write(f"- Held-out cells: {n:,}\n")
        f.write("\n### Pooled across all matmul families\n\n")
        write_pooled(rows, f)
        write_per_family(rows, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
