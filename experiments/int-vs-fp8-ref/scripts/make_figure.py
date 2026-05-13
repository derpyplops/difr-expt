"""Generate the int-vs-fp8 results figure.

Shows two side-by-side panels: (1) the matching-kernel recipe at top-1=1.0
across all three models, and (2) the actual fully-int recipe (no training)
which loses some agreement with fp8 *but* is closer to the bf16 base than
fp8 is on every model.

The second panel is the headline answer to "an int model that performs
similarly to the fp8 model": yes — by top-1 retention from the bf16 base,
the int model is uniformly closer to the unquantized base than fp8 is.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "figures" / "int_vs_fp8.png"


def load(name: str) -> dict:
    with open(DATA / name) as f:
        return json.loads(f.readline())


rows = [
    ("Qwen2.5-0.5B", "qwen25_0p5b_fp8act_nonm.jsonl", "qwen25_0p5b_triple.jsonl"),
    ("Llama-3.1-8B-Instruct", "llama31_8b_fp8act_nonm.jsonl", "llama31_8b_triple.jsonl"),
    ("Qwen3-8B", "qwen3_8b_blockfp8_kernel.jsonl", "qwen3_8b_triple.jsonl"),
]

labels = [r[0] for r in rows]
match = [load(r[1])["student_vs_teacher/top1"] for r in rows]
triples = [load(r[2]) for r in rows]
svt = [t["student_vs_teacher/top1"] for t in triples]
svr = [t["student_vs_ref/top1"] for t in triples]
tvr = [t["teacher_vs_ref/top1"] for t in triples]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(len(labels))
w = 0.27

# Panel A: matching-kernel recipe (operands int, kernel matches teacher)
bars_a = ax1.bar(x, match, color="#1f9b54", edgecolor="black", linewidth=0.6)
ax1.set_ylim(0.9, 1.005)
ax1.set_ylabel("top-1 (student vs fp8 teacher)")
ax1.set_title("Matching-kernel recipe\n(int operands + teacher's kernel + float non-matmul)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax1.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
ax1.grid(axis="y", alpha=0.3, linewidth=0.4)
for b, v in zip(bars_a, match):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

# Panel B: fully-int recipe — three bars per model
bars_b1 = ax2.bar(x - w, svr, w, color="#1f9b54", edgecolor="black", linewidth=0.6,
                  label="int student vs bf16 base")
bars_b2 = ax2.bar(x,     tvr, w, color="#d59f00", edgecolor="black", linewidth=0.6,
                  label="fp8 teacher vs bf16 base")
bars_b3 = ax2.bar(x + w, svt, w, color="#5b9bd5", edgecolor="black", linewidth=0.6,
                  label="int student vs fp8 teacher")
ymin = min(min(svr), min(tvr), min(svt)) - 0.02
ax2.set_ylim(ymin, 1.005)
ax2.set_ylabel("top-1 (vs reference)")
ax2.set_title("Fully-int recipe (no training)\nInt student is closer to bf16 base than fp8 teacher is")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax2.legend(loc="lower right", frameon=True, fontsize=8)
ax2.grid(axis="y", alpha=0.3, linewidth=0.4)
ax2.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
for bars, vs in [(bars_b1, svr), (bars_b2, tvr), (bars_b3, svt)]:
    for b, v in zip(bars, vs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=7.5)

fig.suptitle("Int-cast student vs FP8 production teacher (3 models, 15k+ positions each)",
             fontsize=11, y=1.01)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
