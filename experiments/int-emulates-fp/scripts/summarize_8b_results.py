"""Summarize 8B reproduction results into a markdown table.

Reads:
  - experiments/int-emulates-fp/data/<short>_base_vs_teacher.json   (diag)
  - experiments/int-emulates-fp/data/<short>_v1_naive/summary.json
  - experiments/int-emulates-fp/data/<short>_v13_high_bits/summary.json
  - experiments/int-emulates-fp/data/<short>_v2_weights_only/summary.json
  - experiments/int-emulates-fp/data/v1_naive/summary.json  (0.5B reference)
  - experiments/int-emulates-fp/data/v2_weights_only/summary.json  (0.5B ref)
  - experiments/int-emulates-fp/data/v13_high_bits_no_quant/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def get_top1(summary, key="pre"):
    """Best-effort extract student_vs_teacher top1 from a summary file."""
    if summary is None:
        return None
    return summary.get(key, {}).get("student_vs_teacher/top1")


def get_kl(summary, key="pre"):
    if summary is None:
        return None
    return summary.get(key, {}).get("student_vs_teacher/kl_p99")


def get_l2(summary, key="pre"):
    if summary is None:
        return None
    return summary.get(key, {}).get("student_vs_teacher/logit_l2_p99")


def fmt(x, prec=4):
    if x is None:
        return "—"
    try:
        return f"{x:.{prec}f}"
    except Exception:
        return str(x)


def main():
    root = Path("experiments/int-emulates-fp/data")
    rows = []

    # 0.5B (already known)
    qwen25 = {
        "model": "Qwen2.5-0.5B",
        "diag": load_json(Path("experiments/int-emulates-fp/data/qwen25_0_5b_base_vs_teacher.json")),
        "v1": load_json(root / "v1_naive" / "summary.json"),
        "v13": load_json(root / "v13_high_bits_no_quant" / "summary.json"),
        "v2": load_json(root / "v2_weights_only" / "summary.json"),
    }
    # 0.5B diag we don't have a JSON for; the teacher_vs_ref from v1's pre is
    # essentially the same comparison (0.9340 in the existing 0.5B v1).
    if qwen25["v1"] is not None:
        # teacher_vs_ref is base-vs-teacher (since ref is base in train_emulate)
        v1_pre = qwen25["v1"]["pre"]
        qwen25_diag_top1 = v1_pre.get("teacher_vs_ref/top1")
    else:
        qwen25_diag_top1 = None
    rows.append({
        "model": "Qwen2.5-0.5B (0.5B reference)",
        "base_vs_teacher": qwen25_diag_top1,
        "v1": get_top1(qwen25["v1"], "pre"),
        "v13": get_top1(qwen25["v13"], "pre"),
        "v2_post": get_top1(qwen25["v2"], "post"),
        "v2_best_step": qwen25["v2"]["best_step"] if qwen25["v2"] else None,
    })

    for short, label in [
        ("qwen3_8b", "Qwen3-8B"),
        ("llama31_8b", "Llama-3.1-8B-Instruct"),
    ]:
        diag = load_json(root / f"{short}_base_vs_teacher.json")
        v1 = load_json(root / f"{short}_v1_naive" / "summary.json")
        v13 = load_json(root / f"{short}_v13_high_bits" / "summary.json")
        v2 = load_json(root / f"{short}_v2_weights_only" / "summary.json")
        rows.append({
            "model": label,
            "base_vs_teacher": diag["top1_base_vs_teacher"] if diag else None,
            "v1": get_top1(v1, "pre"),
            "v13": get_top1(v13, "pre"),
            "v2_post": get_top1(v2, "post"),
            "v2_best_step": v2["best_step"] if v2 else None,
            "v1_kl_p99": get_kl(v1, "pre"),
            "v1_logit_l2_p99": get_l2(v1, "pre"),
        })

    print("\n## Top-1 against teacher (8B reproduction)\n")
    print("| Model | base-vs-teacher | V1 (int24 naive) | V13 (bits=31) | V2 (trained) | V2 best step |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['model']} | {fmt(r['base_vs_teacher'])} | {fmt(r['v1'])} | "
            f"{fmt(r['v13'])} | {fmt(r.get('v2_post'))} | "
            f"{r.get('v2_best_step') if r.get('v2_best_step') is not None else '—'} |"
        )

    # Also dump full JSON for later use
    print("\n## Raw data\n")
    print("```json")
    print(json.dumps(rows, indent=2, default=str))
    print("```")


if __name__ == "__main__":
    main()
