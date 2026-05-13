"""Wikitext PPL comparison: bf16 base vs fp8 teacher vs int24 student.

The PPL panel is the headline answer to "does the int model perform similarly
to fp8 on a downstream-quality metric?" — on 3 of 4 models the int24 student
PPL is essentially identical to (or slightly better than) the bf16 base PPL,
and better than the fp8 teacher's PPL.
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
    ("Qwen2.5-0.5B", "qwen25_0p5b_ppl.jsonl"),
    ("Qwen2.5-1.5B", "qwen25_1p5b_ppl.jsonl"),
    ("Qwen3-8B", "qwen3_8b_ppl.jsonl"),
    ("Llama-3.1-8B", "llama31_8b_ppl.jsonl"),
]

labels = []
base_ppl = []
fp8_ppl = []
int_ppl = []
for name, path in rows:
    if not (DATA / path).exists():
        continue
    d = load(path)
    labels.append(name)
    base_ppl.append(d["bf16_base"]["ppl"])
    fp8_ppl.append(d["fp8_teacher"]["ppl"])
    int_ppl.append(d["int_student"]["ppl"])

x = np.arange(len(labels))
w = 0.28
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: absolute PPL
bars_a1 = ax1.bar(x - w, base_ppl, w, color="#777", edgecolor="black", linewidth=0.6, label="bf16 base")
bars_a2 = ax1.bar(x,     fp8_ppl,  w, color="#d59f00", edgecolor="black", linewidth=0.6, label="fp8 teacher")
bars_a3 = ax1.bar(x + w, int_ppl,  w, color="#1f9b54", edgecolor="black", linewidth=0.6, label="int24 student")
ax1.set_ylabel("wikitext perplexity (lower is better)")
ax1.set_title("Absolute PPL — int24 student tracks bf16 base on 3 of 4 models")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax1.legend(loc="upper right", frameon=True, fontsize=9)
ax1.grid(axis="y", alpha=0.3, linewidth=0.4)
for bars, vs in [(bars_a1, base_ppl), (bars_a2, fp8_ppl), (bars_a3, int_ppl)]:
    for b, v in zip(bars, vs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)

# Panel B: relative delta vs base
fp8_delta = [100 * (f - b) / b for f, b in zip(fp8_ppl, base_ppl)]
int_delta = [100 * (i - b) / b for i, b in zip(int_ppl, base_ppl)]
bars_b1 = ax2.bar(x - w/2, fp8_delta, w, color="#d59f00", edgecolor="black", linewidth=0.6, label="fp8 teacher Δ")
bars_b2 = ax2.bar(x + w/2, int_delta, w, color="#1f9b54", edgecolor="black", linewidth=0.6, label="int24 student Δ")
ax2.set_ylabel("PPL delta vs bf16 base (%)")
ax2.set_title("Quantization noise (closer to 0 is better)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.legend(loc="upper right", frameon=True, fontsize=9)
ax2.grid(axis="y", alpha=0.3, linewidth=0.4)
for bars, vs in [(bars_b1, fp8_delta), (bars_b2, int_delta)]:
    for b, v in zip(bars, vs):
        ax2.text(b.get_x() + b.get_width() / 2,
                 v + (0.05 if v > 0 else -0.15),
                 f"{v:+.2f}%",
                 ha="center",
                 va="bottom" if v >= 0 else "top",
                 fontsize=8)

fig.suptitle("Fully-int model wikitext PPL: matches base on 3/4 models, beats fp8 on PPL",
             fontsize=11, y=1.01)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
