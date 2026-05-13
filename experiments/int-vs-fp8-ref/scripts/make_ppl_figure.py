"""Wikitext PPL across 6 models: fully-int student vs bf16 base vs fp8 teacher.

Two comparisons:
- vs bf16 sdpa base (production-loadout reference)
- vs bf16 eager base (apples-to-apples — int student uses eager attention)

The eager-vs-sdpa drift is a pytorch numerical quirk separate from int quant.
On Qwen2.5-{1.5B,7B} it's huge (3% / 8%), which inflated the apparent int cost
when we compared int-eager vs base-sdpa. The clean comparison (int vs eager)
shows the int approximations cost at most ~1.7% PPL on any model tested.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "figures" / "ppl_compare.png"


def load(name: str) -> dict:
    with open(DATA / name) as f:
        return json.loads(f.readline())


rows = [
    ("Qwen2.5-0.5B", "qwen25_0p5b_ppl_v2.jsonl"),
    ("Qwen2.5-1.5B", "qwen25_1p5b_ppl_v2.jsonl"),
    ("Qwen2.5-3B",   "qwen25_3b_ppl_v2.jsonl"),
    ("Qwen2.5-7B",   "qwen25_7b_ppl_v2.jsonl"),
    ("Qwen3-8B",     "qwen3_8b_ppl_v2.jsonl"),
    ("Llama-3.1-8B", "llama31_8b_ppl_v2.jsonl"),
]

labels, sdpa, eager, fp8, intd = [], [], [], [], []
for name, p in rows:
    if not (DATA / p).exists():
        continue
    d = load(p)
    labels.append(name)
    sdpa.append(d["bf16_base"]["ppl"])
    eager.append(d["bf16_base_eager"]["ppl"])
    fp8.append(d["fp8_teacher"]["ppl"])
    intd.append(d["int_student"]["ppl"])

x = np.arange(len(labels))
w = 0.20
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# Panel A: absolute PPL
bars1 = ax1.bar(x - 1.5*w, sdpa,  w, color="#666", edgecolor="black", linewidth=0.5, label="bf16 base (sdpa)")
bars2 = ax1.bar(x - 0.5*w, eager, w, color="#aaa", edgecolor="black", linewidth=0.5, label="bf16 base (eager)")
bars3 = ax1.bar(x + 0.5*w, fp8,   w, color="#d59f00", edgecolor="black", linewidth=0.5, label="fp8 teacher")
bars4 = ax1.bar(x + 1.5*w, intd,  w, color="#1f9b54", edgecolor="black", linewidth=0.5, label="int24 student (full int)")
ax1.set_ylabel("wikitext perplexity (lower is better)")
ax1.set_title("Absolute PPL — fully-int model with int matmul + int RMSNorm/SiLU/softmax/attn/RoPE")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax1.legend(loc="upper right", frameon=True, fontsize=8)
ax1.grid(axis="y", alpha=0.3, linewidth=0.4)
for bars, vs in [(bars1, sdpa), (bars2, eager), (bars3, fp8), (bars4, intd)]:
    for b, v in zip(bars, vs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7)

# Panel B: delta vs eager base (apples-to-apples since int student uses eager)
fp8_e_delta = [100 * (f - e) / e for f, e in zip(fp8, eager)]
int_e_delta = [100 * (i - e) / e for i, e in zip(intd, eager)]
sdpa_e_delta = [100 * (s - e) / e for s, e in zip(sdpa, eager)]
bars5 = ax2.bar(x - w, sdpa_e_delta, w, color="#666", edgecolor="black", linewidth=0.5, label="bf16 sdpa (drift from eager)")
bars6 = ax2.bar(x,     fp8_e_delta,  w, color="#d59f00", edgecolor="black", linewidth=0.5, label="fp8 teacher")
bars7 = ax2.bar(x + w, int_e_delta,  w, color="#1f9b54", edgecolor="black", linewidth=0.5, label="int24 student")
ax2.set_ylabel("PPL delta vs bf16-eager base (%)")
ax2.set_title("Delta from eager base (the apples-to-apples ref)\nInt student is within ±1.7% across 6 models")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.legend(loc="lower right", frameon=True, fontsize=8)
ax2.grid(axis="y", alpha=0.3, linewidth=0.4)
for bars, vs in [(bars5, sdpa_e_delta), (bars6, fp8_e_delta), (bars7, int_e_delta)]:
    for b, v in zip(bars, vs):
        ax2.text(b.get_x() + b.get_width() / 2,
                 v + (0.06 if v > 0 else -0.18),
                 f"{v:+.2f}%",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

fig.suptitle("Fully-int model matches bf16 base PPL within 1.7% across 6 models (sdpa-vs-eager drift is a separate pytorch issue)",
             fontsize=11, y=1.01)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
