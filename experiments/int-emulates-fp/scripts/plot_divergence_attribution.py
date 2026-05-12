"""Plot the divergence attribution results from `divergence_attribution.py`."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Group "role" tags into colors and category for the bar chart.
CATEGORY = {
    "embed": "boundary",
    "model.norm": "boundary",
    "lm_head": "boundary",
    "input_layernorm": "rmsnorm",
    "post_attention_layernorm": "rmsnorm",
    "self_attn.q_proj": "attn matmul",
    "self_attn.k_proj": "attn matmul",
    "self_attn.v_proj": "attn matmul",
    "self_attn.o_proj": "attn matmul",
    "self_attn": "attn (softmax-effect)",
    "mlp.gate_proj": "mlp matmul",
    "mlp.up_proj": "mlp matmul",
    "mlp.down_proj": "mlp matmul",
    "mlp": "mlp (silu-effect)",
    "layer_block": "residual stream",
}
CATEGORY_COLORS = {
    "boundary": "#444",
    "rmsnorm": "#1f77b4",
    "attn matmul": "#ff7f0e",
    "attn (softmax-effect)": "#d62728",
    "mlp matmul": "#2ca02c",
    "mlp (silu-effect)": "#9467bd",
    "residual stream": "#7f7f7f",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON from divergence_attribution.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    data = json.load(open(args.input))
    per_module = data["per_module"]
    role_summary = data["role_summary"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Per-role bar chart (L2 mean across layers) ----------------
    roles_sorted = sorted(role_summary.keys(), key=lambda r: -role_summary[r]["l2_mean"])
    l2s = [role_summary[r]["l2_mean"] for r in roles_sorted]
    cats = [CATEGORY.get(r, "other") for r in roles_sorted]
    colors = [CATEGORY_COLORS.get(c, "#888") for c in cats]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(roles_sorted)), l2s, color=colors)
    ax.set_yticks(range(len(roles_sorted)))
    ax.set_yticklabels(roles_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean L2 of (student_output − teacher_output), per position, averaged over layers")
    ax.set_title(f"Per-module divergence{args.title_suffix}")
    # Annotate values + n_layers
    for i, r in enumerate(roles_sorted):
        s = role_summary[r]
        ax.text(s["l2_mean"], i, f"  {s['l2_mean']:.2f}  (n={s['n_layers']})",
                va="center", fontsize=8)
    # Legend by category
    seen = []
    handles, labels = [], []
    for c, col in CATEGORY_COLORS.items():
        if c in [CATEGORY.get(r, "other") for r in roles_sorted] and c not in seen:
            seen.append(c)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=col))
            labels.append(c)
    ax.legend(handles, labels, loc="lower right", fontsize=8)
    plt.tight_layout()
    p1 = out_dir / "per_role_l2.png"
    plt.savefig(p1, dpi=130)
    plt.close()
    print(f"wrote {p1}")

    # --- Plot 2: Per-layer trajectory ----------------------------------------
    # For each interesting role, plot L2 vs layer index. Use the per-module
    # records; layer_idx is -1 for boundary modules, skip those.
    by_role_layer: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for name, info in per_module.items():
        if info["layer_idx"] < 0:
            continue
        by_role_layer[info["role"]].append((info["layer_idx"], info["l2_mean"]))

    fig, ax = plt.subplots(figsize=(11, 6))
    role_order = [
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn",          # softmax-effect
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp",                # silu-effect
        "layer_block",
    ]
    for r in role_order:
        if r not in by_role_layer:
            continue
        pts = sorted(by_role_layer[r])
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        cat = CATEGORY.get(r, "other")
        color = CATEGORY_COLORS.get(cat, "#888")
        linestyle = "-" if "matmul" in cat or cat in ("rmsnorm", "residual stream") else "--"
        ax.plot(xs, ys, label=r, color=color, linestyle=linestyle, marker="o", markersize=3)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Mean L2(student − teacher) at module output")
    ax.set_title(f"Divergence trajectory by depth{args.title_suffix}")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = out_dir / "per_layer_trajectory.png"
    plt.savefig(p2, dpi=130)
    plt.close()
    print(f"wrote {p2}")

    # --- Plot 3: Category aggregated -----------------------------------------
    # Sum L2 contributions across same-category roles to give a high-level pie.
    cat_totals: dict[str, float] = defaultdict(float)
    cat_counts: dict[str, int] = defaultdict(int)
    for r, s in role_summary.items():
        c = CATEGORY.get(r, "other")
        # Use sum across layers (l2_mean * n_layers) so a 24-layer op counts more
        cat_totals[c] += s["l2_mean"] * s["n_layers"]
        cat_counts[c] += s["n_layers"]

    fig, ax = plt.subplots(figsize=(9, 6))
    cats_sorted = sorted(cat_totals.keys(), key=lambda c: -cat_totals[c])
    totals = [cat_totals[c] for c in cats_sorted]
    means = [cat_totals[c] / cat_counts[c] for c in cats_sorted]
    colors = [CATEGORY_COLORS.get(c, "#888") for c in cats_sorted]
    x = range(len(cats_sorted))
    bars = ax.bar(x, totals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(cats_sorted, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Σ L2 (sum across all layers/modules in category)")
    ax.set_title(f"Aggregate divergence by op category{args.title_suffix}")
    for i, (t, m) in enumerate(zip(totals, means)):
        ax.text(i, t, f"Σ={t:.0f}\nmean={m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(totals) * 1.2)
    plt.tight_layout()
    p3 = out_dir / "per_category_total.png"
    plt.savefig(p3, dpi=130)
    plt.close()
    print(f"wrote {p3}")


if __name__ == "__main__":
    main()
