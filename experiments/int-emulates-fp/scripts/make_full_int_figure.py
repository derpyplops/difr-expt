"""Generate the full-int-model results figure: top1 + Gumbel margin per model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "full_int_model"
OUT = Path(__file__).resolve().parents[1] / "figures" / "full_int_model.png"


def load(name: str) -> dict:
    with open(DATA / name) as f:
        return json.loads(f.readline())


rows = [
    ("Qwen2.5-0.5B", "RedHatAI fp8\n(per-row)", load("qwen25_0p5b.jsonl")),
    ("Llama-3.1-8B-Instruct", "RedHatAI fp8\n(per-row)", load("llama31_8b.jsonl")),
    ("Qwen3-8B", "Qwen native\n(block-fp8)", load("qwen3_8b.jsonl")),
]

labels = [r[0] for r in rows]
teachers = [r[1] for r in rows]
top1 = [r[2]["student_vs_teacher/top1"] for r in rows]
margin_mean = [r[2]["student_vs_teacher/margin_mean"] for r in rows]
margin_p99 = [r[2]["student_vs_teacher/margin_p99"] for r in rows]
n_pos = [int(r[2]["student_vs_teacher/n_positions"]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.1]})

# Panel A: top-1 bars
x = np.arange(len(labels))
colors = ["#1f9b54", "#1f9b54", "#1f9b54"]  # all bit-exact now
bars1 = ax1.bar(x, top1, color=colors, edgecolor="black", linewidth=0.6)
ax1.set_ylim(0.9, 1.005)
ax1.set_ylabel("top-1 (student vs teacher)")
ax1.set_title("Top-1 agreement on held-out positions")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax1.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
ax1.grid(axis="y", alpha=0.3, linewidth=0.4)
for i, (b, v) in enumerate(zip(bars1, top1)):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.text(b.get_x() + b.get_width() / 2, 0.905, f"n={n_pos[i]}\n{teachers[i]}",
             ha="center", va="bottom", fontsize=7.5, color="black", alpha=0.7)

# Panel B: Gumbel margin (mean + p99) bars
w = 0.38
bars2a = ax2.bar(x - w / 2, margin_mean, w, color="#5b9bd5",
                 edgecolor="black", linewidth=0.6, label="mean")
bars2b = ax2.bar(x + w / 2, margin_p99, w, color="#ed7d31",
                 edgecolor="black", linewidth=0.6, label="p99")
ax2.set_ylabel("Gumbel margin (δ, clipped at 50)")
ax2.set_title("Token-DiFR Gumbel margin (lower is better; 0 = bit-exact)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax2.legend(loc="upper left", frameon=True)
ax2.grid(axis="y", alpha=0.3, linewidth=0.4)
for b, v in list(zip(bars2a, margin_mean)) + list(zip(bars2b, margin_p99)):
    label = "0.000" if v < 1e-6 else f"{v:.3f}"
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, label,
             ha="center", va="bottom", fontsize=8)
ax2.set_ylim(0, max(max(margin_p99), 0.05) * 1.2)

fig.suptitle("Full int-emulation of fp8 production models: bit-exact teacher on 3/3 models",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
