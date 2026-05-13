"""Compute Y', Ỹ, and r = Y' - Ỹ from captured records.

Two cast modes:

  - "tight":  operand-lossless on the fp8e4m3 grid. X' = round(x_q · 2^B_OP / s_X),
              W' = round(w_q · 2^B_OP / s_W). Output Ỹ = round(Y · 2^(2*B_OP) /
              (s_X · s_W)). With B_OP=9 this is operand-lossless on fp8e4m3
              (smallest subnormal is 2^-9, so all codes round to integers
              losslessly). The residual r captures only kernel reduction
              drift + output-cast rounding.

  - "int8":   operand-lossy. X' = round(x_q · 127 / (s_X · FP8_MAX)) so
              X' ∈ [-127, 127]. Similarly for W'. Output is scaled to the
              same coordinate. The residual r captures the full quantization
              error (codebook, cancellation, outlier — all the sources in
              the plan doc).

  - "int16":  operand-lossless on the fp8e4m3 grid (since the grid only has
              256 codes and 17 bits suffice). Same recipe as int8 with
              max = 32767.

Output: a single flat dataset per cast mode, one row per output cell,
augmented with per-row features that future residual models will consume:

  cell_features = {
      "matmul_id": which capture record this came from (int)
      "family":    matmul family code (q=0,k=1,v=2,o=3,gate=4,up=5,down=6)
      "block":     transformer block index (int)
      "prompt":    prompt index (int)
      "t":         token index within prompt
      "d":         output column index
      "s_X":       per-token scale (fp32)
      "s_W":       per-row weight scale (fp32)
      "Y_prime":   the int matmul value (int64)
      "Y_tilde":   the cast of the production Y (int64)
      "r":         Y_prime - Y_tilde (int64)
  }

Plus, for downstream residual models that want them, we save the operand
sums per (t, d):

  features = {
      "sum_pos":     Σ_{k: X'·W' > 0} X'·W'        (int64)
      "sum_neg":     Σ_{k: X'·W' < 0} (-X'·W')     (int64)
      "abs_sum":     Σ |X'·W'|                      (int64)
      # sanity: sum_pos - sum_neg should equal Y'
      "topk_xw_mass": Σ of largest 4 |X'·W'| products (int64)
      "n_clip_X":     count of |X'| at saturation   (int32)
      "n_clip_W":     count of |W'| at saturation   (int32)
      "x_absmax":     max |X'| across k             (int32)
      "w_absmax":     max |W'| across k             (int32)
  }

Per-cell features (top-k mass, clip counts) require Σ-along-k work
identical in cost to the matmul itself. We compute them once here and save;
proof cost is tracked separately.

The dataset is stored as a dict of 1-D tensors so a residual model can
slice it directly. We save one .pt per source prompt to preserve the
train/val split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


FAMILY_CODE = {"q": 0, "k": 1, "v": 2, "o": 3, "gate": 4, "up": 5, "down": 6, "other": 7}
FP8_E4M3_MAX = 448.0


def cast_tight(
    X_q: torch.Tensor, W_q: torch.Tensor, Y: torch.Tensor,
    s_X: torch.Tensor, s_W: torch.Tensor, B_OP: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Operand-lossless cast on the fp8e4m3 grid.

    X' = round(x_q · 2^B_OP / s_X),  W' = round(w_q · 2^B_OP / s_W).
    Ỹ  = round(Y · 2^(2·B_OP) / (s_X · s_W))   (per-cell scale s_X[t]·s_W[d])
    """
    scale = float(1 << B_OP)
    # X_q, W_q in fp16; cast up to fp32 for the divide.
    X_f = X_q.to(torch.float64)
    W_f = W_q.to(torch.float64)
    Y_f = Y.to(torch.float64)

    X_exact = X_f * scale / s_X.to(torch.float64).unsqueeze(-1)
    W_exact = W_f * scale / s_W.to(torch.float64).unsqueeze(-1)
    X_prime_f = torch.round(X_exact)
    W_prime_f = torch.round(W_exact)
    X_prime = X_prime_f.to(torch.int32)
    W_prime = W_prime_f.to(torch.int32)
    delta_X = (X_prime_f - X_exact).to(torch.float64)
    delta_W = (W_prime_f - W_exact).to(torch.float64)

    out_scale = (scale * scale)  # 2^(2*B_OP)
    cell_scale = s_X.to(torch.float64).unsqueeze(-1) * s_W.to(torch.float64).unsqueeze(0)
    Y_tilde = torch.round(Y_f * out_scale / cell_scale).to(torch.int64)
    return X_prime, W_prime, Y_tilde, delta_X, delta_W


def cast_intN(
    X_q: torch.Tensor, W_q: torch.Tensor, Y: torch.Tensor,
    s_X: torch.Tensor, s_W: torch.Tensor, n_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Operand-lossy cast to intN.

    X' = round(x_q · MAX / (s_X · FP8_MAX))  ∈ [-MAX, MAX]
    Same coord on output: Ỹ = round(Y · MAX² / (s_X · s_W · FP8_MAX²))

    Also returns δ_X = X' - exact, δ_W = W' - exact (each in float64).
    These are the per-product rounding errors that drive the residual's
    leading term r ≈ α (c_X·δ_W + c_W·δ_X).
    """
    MAX = (1 << (n_bits - 1)) - 1  # 127 for int8, 32767 for int16
    X_f = X_q.to(torch.float64)
    W_f = W_q.to(torch.float64)
    Y_f = Y.to(torch.float64)

    # exact int-coord values (no rounding):
    X_exact = X_f * MAX / (s_X.to(torch.float64).unsqueeze(-1) * FP8_E4M3_MAX)
    W_exact = W_f * MAX / (s_W.to(torch.float64).unsqueeze(-1) * FP8_E4M3_MAX)
    X_prime_f = torch.round(X_exact).clamp(-MAX, MAX)
    W_prime_f = torch.round(W_exact).clamp(-MAX, MAX)
    X_prime = X_prime_f.to(torch.int32)
    W_prime = W_prime_f.to(torch.int32)
    delta_X = (X_prime_f - X_exact).to(torch.float64)
    delta_W = (W_prime_f - W_exact).to(torch.float64)

    # Output cast: same scaling factor squared.
    out_scale = (MAX * MAX) / (FP8_E4M3_MAX * FP8_E4M3_MAX)
    cell_scale = s_X.to(torch.float64).unsqueeze(-1) * s_W.to(torch.float64).unsqueeze(0)  # [T, D]
    Y_tilde = torch.round(Y_f * out_scale / cell_scale).to(torch.int64)
    return X_prime, W_prime, Y_tilde, delta_X, delta_W


def int_matmul(X_prime: torch.Tensor, W_prime: torch.Tensor) -> torch.Tensor:
    """Compute Y' = X' @ W'.T  exactly in integer arithmetic.

    With int16 operands and K up to ~10k, the partial-sum max is
    2^15 · 2^15 · 10^4 ≈ 2^44, comfortably within int64.

    With int8 operands and K up to ~10k, it's 2^7 · 2^7 · 10^4 ≈ 2^28 — easy.

    With tight cast (B_OP=9) the operands sit in [-FP8_MAX·512, FP8_MAX·512] =
    [-229376, 229376] — about int19. Squared is int38, times K up to 14336
    is ≈ 2^52, still in int64 but tighter.

    Torch's int matmul requires CPU (no GPU int matmul for int32 in 2.7).
    We cast to int64 and use float64 matmul via integer-preserving path: the
    fastest exact-integer approach is `(X.long() @ W.long().T)` which torch
    will dispatch through CPU int64 GEMM if available, else fall back. For
    safety we use a python loop with chunked int64 matmul.
    """
    Xl = X_prime.to(torch.int64)
    Wl = W_prime.to(torch.int64)
    # torch int64 matmul on CPU is slow. We use float64 reduction on CPU
    # with integer-exact semantics, since |X|, |W| are small enough that
    # the partial sums stay within float64's 53-bit mantissa.
    #   X: [T, K], W: [D, K], want Y: [T, D] = X @ W.T
    # float64 reduction over K is exact as long as each partial sum is
    # representable. For int19 × int19 × 16k = int38 * 16k ≈ 2^51, within
    # the 53-bit mantissa range. For int19² × 4k MLP intermediate=14336:
    # 2^38 · 14336 ≈ 2^52 — still fits. So we can use fp64 matmul.
    Y_prime_f = Xl.to(torch.float64) @ Wl.to(torch.float64).T
    # Now snap to integer; the result is exact if no overflow occurred.
    Y_prime = Y_prime_f.round().to(torch.int64)
    # Sanity: assert that the round didn't change the value, modulo fp64 epsilon.
    diff = (Y_prime_f - Y_prime.to(torch.float64)).abs().max().item()
    if diff > 1e-3:
        raise RuntimeError(
            f"int matmul fp64-reduction lost precision: max diff = {diff}"
        )
    return Y_prime


def compute_features(
    X_prime: torch.Tensor, W_prime: torch.Tensor, top_k: int = 4
) -> dict[str, torch.Tensor]:
    """Per-cell features from operand int values, costing Σ-along-k each.

    sum_pos[t, d] = Σ_{k: X'_t,k · W'_d,k > 0} X'_t,k · W'_d,k
    sum_neg[t, d] = Σ_{k: X'_t,k · W'_d,k < 0} (-X'_t,k · W'_d,k)
    abs_sum[t, d] = Σ |X'_t,k · W'_d,k|
    topk_xw[t, d] = Σ of largest top_k |X'_t,k · W'_d,k|  (FP4 "big eats rest" diag)

    Returns dict of [T, D] tensors (int64).
    """
    Xl = X_prime.to(torch.int64)
    Wl = W_prime.to(torch.int64)
    # We need element-wise products per (t, d, k). That's [T, D, K] memory,
    # which can blow up. For Qwen2.5-0.5B with T=256, D=896 or 4864, K=896 or
    # 4864, the largest intermediate is 256·4864·896 ≈ 1.1B entries * 8B = 9GB.
    # Too large. We chunk along D.
    T, K = Xl.shape
    D, _ = Wl.shape
    sum_pos = torch.zeros(T, D, dtype=torch.int64)
    sum_neg = torch.zeros(T, D, dtype=torch.int64)
    abs_sum = torch.zeros(T, D, dtype=torch.int64)
    topk_xw = torch.zeros(T, D, dtype=torch.int64)
    # Auto-pick chunk so [T, D_chunk, K] stays under ~256M int64 = 2GB.
    max_entries = 256 * 1024 * 1024
    d_chunk = max(1, max_entries // max(1, T * K))
    for d0 in range(0, D, d_chunk):
        d1 = min(D, d0 + d_chunk)
        # [T, 1, K] * [1, d1-d0, K] = [T, d1-d0, K]
        prod = Xl[:, None, :] * Wl[None, d0:d1, :]  # int64
        pos_mask = prod > 0
        neg_mask = prod < 0
        sum_pos[:, d0:d1] = (prod * pos_mask).sum(dim=-1)
        sum_neg[:, d0:d1] = (-prod * neg_mask).sum(dim=-1)
        abs_sum[:, d0:d1] = prod.abs().sum(dim=-1)
        # top-k abs across k
        absprod = prod.abs()
        topk_vals, _ = absprod.topk(min(top_k, K), dim=-1)
        topk_xw[:, d0:d1] = topk_vals.sum(dim=-1)
    return {
        "sum_pos": sum_pos,
        "sum_neg": sum_neg,
        "abs_sum": abs_sum,
        "topk_xw": topk_xw,
    }


def build_one_prompt(rec_file: Path, out_file: Path, cast_mode: str,
                     compute_full_features: bool, top_k: int = 4,
                     B_OP: int = 9, intN_bits: int = 8) -> dict:
    """Process all records in a prompt's .pt and emit a flat per-cell dataset.

    Returns summary stats for printing.
    """
    blob = torch.load(rec_file, map_location="cpu", weights_only=False)
    records = blob["records"]

    # We'll concatenate per-cell features across all records in this prompt.
    out_chunks: dict[str, list] = {
        "matmul_id": [], "family": [], "block": [], "prompt": [],
        "t": [], "d": [],
        "s_X": [], "s_W": [],
        "Y_prime": [], "Y_tilde": [], "r": [],
        "n_clip_X": [], "n_clip_W": [],
        "x_absmax": [], "w_absmax": [],
        # leading-term proxies: O(K)-per-row / O(K)-per-col deltas.
        # These are CHEAP — committed once per row/column, not per cell.
        "dX_sum_t":    [],  # per-row Σ_k δ_X[t,k]
        "dW_sum_d":    [],  # per-col Σ_k δ_W[d,k]
        "dX_abs_sum_t":[],  # per-row Σ_k |δ_X[t,k]|
        "dW_abs_sum_d":[],  # per-col Σ_k |δ_W[d,k]|
        "Xp_sum_t":    [],  # per-row Σ_k X'[t,k]
        "Wp_sum_d":    [],  # per-col Σ_k W'[d,k]
        "Xp_abs_sum_t":[],  # per-row Σ_k |X'[t,k]|
        "Wp_abs_sum_d":[],  # per-col Σ_k |W'[d,k]|
        # Per-cell "leading-term" mixed dot products. These cost a full
        # K-sum per cell — same order as the matmul. Useful as an
        # *upper-bound* feature: they should give the residual its best
        # achievable fit. We compute them only when --compute-features.
    }
    if compute_full_features:
        for k in ("sum_pos", "sum_neg", "abs_sum", "topk_xw",
                  "Xp_dW", "Wp_dX", "dX_dW"):
            out_chunks[k] = []

    n_cells_total = 0
    for matmul_id, r in enumerate(records):
        X_q = r["X_q"]    # [T, K]  fp16
        W_q = r["W_q"]    # [D, K]  fp16
        Y   = r["Y"]      # [T, D]  fp32
        s_X = r["s_X"]    # [T]
        s_W = r["s_W"]    # [D]
        family = r["family"]
        block = r["block"]
        prompt = r["prompt"]

        if cast_mode == "tight":
            X_prime, W_prime, Y_tilde, delta_X, delta_W = cast_tight(
                X_q, W_q, Y, s_X, s_W, B_OP=B_OP
            )
            max_op = (1 << B_OP) * int(FP8_E4M3_MAX)  # ~229376 with B=9
        elif cast_mode.startswith("int"):
            n_bits = int(cast_mode[3:])  # int8, int16, int24
            X_prime, W_prime, Y_tilde, delta_X, delta_W = cast_intN(
                X_q, W_q, Y, s_X, s_W, n_bits=n_bits
            )
            max_op = (1 << (n_bits - 1)) - 1
        else:
            raise ValueError(f"unknown cast_mode {cast_mode!r}")

        Y_prime = int_matmul(X_prime, W_prime)        # [T, D] int64
        r_val = Y_prime - Y_tilde                       # [T, D] int64

        T, D = Y_prime.shape
        cell_t = torch.arange(T).unsqueeze(-1).expand(T, D).reshape(-1)
        cell_d = torch.arange(D).unsqueeze(0).expand(T, D).reshape(-1)

        # per-row features
        out_chunks["matmul_id"].append(torch.full((T * D,), matmul_id, dtype=torch.int32))
        out_chunks["family"].append(torch.full((T * D,), FAMILY_CODE.get(family, 7), dtype=torch.int8))
        out_chunks["block"].append(torch.full((T * D,), block, dtype=torch.int16))
        out_chunks["prompt"].append(torch.full((T * D,), prompt, dtype=torch.int16))
        out_chunks["t"].append(cell_t.to(torch.int16))
        out_chunks["d"].append(cell_d.to(torch.int32))
        # broadcasted scales:
        s_X_b = s_X.unsqueeze(-1).expand(T, D).reshape(-1)
        s_W_b = s_W.unsqueeze(0).expand(T, D).reshape(-1)
        out_chunks["s_X"].append(s_X_b.contiguous())
        out_chunks["s_W"].append(s_W_b.contiguous())
        out_chunks["Y_prime"].append(Y_prime.reshape(-1))
        out_chunks["Y_tilde"].append(Y_tilde.reshape(-1))
        out_chunks["r"].append(r_val.reshape(-1))

        # cheap per-cell features that don't need per-(t,d,k) products
        # clip counts on X (per-token) and W (per-row) — same for all d (or t)
        n_clip_X_t = (X_prime.abs() >= max_op).sum(dim=-1).to(torch.int32)  # [T]
        n_clip_W_d = (W_prime.abs() >= max_op).sum(dim=-1).to(torch.int32)  # [D]
        x_absmax_t = X_prime.abs().amax(dim=-1).to(torch.int32)             # [T]
        w_absmax_d = W_prime.abs().amax(dim=-1).to(torch.int32)             # [D]
        out_chunks["n_clip_X"].append(
            n_clip_X_t.unsqueeze(-1).expand(T, D).reshape(-1).contiguous()
        )
        out_chunks["n_clip_W"].append(
            n_clip_W_d.unsqueeze(0).expand(T, D).reshape(-1).contiguous()
        )
        out_chunks["x_absmax"].append(
            x_absmax_t.unsqueeze(-1).expand(T, D).reshape(-1).contiguous()
        )
        out_chunks["w_absmax"].append(
            w_absmax_d.unsqueeze(0).expand(T, D).reshape(-1).contiguous()
        )

        # delta reductions (per-row, per-col) — cheap features
        dX_sum_t     = delta_X.sum(dim=-1)             # [T]
        dW_sum_d     = delta_W.sum(dim=-1)             # [D]
        dX_abs_sum_t = delta_X.abs().sum(dim=-1)       # [T]
        dW_abs_sum_d = delta_W.abs().sum(dim=-1)       # [D]
        Xp_sum_t     = X_prime.to(torch.float64).sum(dim=-1)     # [T]
        Wp_sum_d     = W_prime.to(torch.float64).sum(dim=-1)     # [D]
        Xp_abs_sum_t = X_prime.to(torch.float64).abs().sum(dim=-1)  # [T]
        Wp_abs_sum_d = W_prime.to(torch.float64).abs().sum(dim=-1)  # [D]
        for key, src, dim in [
            ("dX_sum_t",     dX_sum_t,     "T"),
            ("dW_sum_d",     dW_sum_d,     "D"),
            ("dX_abs_sum_t", dX_abs_sum_t, "T"),
            ("dW_abs_sum_d", dW_abs_sum_d, "D"),
            ("Xp_sum_t",     Xp_sum_t,     "T"),
            ("Wp_sum_d",     Wp_sum_d,     "D"),
            ("Xp_abs_sum_t", Xp_abs_sum_t, "T"),
            ("Wp_abs_sum_d", Wp_abs_sum_d, "D"),
        ]:
            if dim == "T":
                bcast = src.to(torch.float32).unsqueeze(-1).expand(T, D)
            else:
                bcast = src.to(torch.float32).unsqueeze(0).expand(T, D)
            out_chunks[key].append(bcast.reshape(-1).contiguous())

        if compute_full_features:
            feats = compute_features(X_prime, W_prime, top_k=top_k)
            for k, v in feats.items():
                out_chunks[k].append(v.reshape(-1))
            # Per-cell mixed dot products X'·δ_W, W'·δ_X, δ_X·δ_W
            # Each is a [T, K] · [D, K] -> [T, D] reduction. EXPENSIVE.
            Xp_f = X_prime.to(torch.float64)
            Wp_f = W_prime.to(torch.float64)
            Xp_dW = Xp_f @ delta_W.T   # [T, D]
            Wp_dX = delta_X @ Wp_f.T   # [T, D]
            dX_dW = delta_X @ delta_W.T  # [T, D]
            for k, v in [("Xp_dW", Xp_dW), ("Wp_dX", Wp_dX), ("dX_dW", dX_dW)]:
                out_chunks[k].append(v.to(torch.float32).reshape(-1).contiguous())

        n_cells_total += T * D

    flat: dict[str, torch.Tensor] = {}
    for k, chunks in out_chunks.items():
        flat[k] = torch.cat(chunks, dim=0)
    flat["meta"] = {
        "cast_mode": cast_mode,
        "B_OP": B_OP if cast_mode == "tight" else None,
        "intN_bits": intN_bits if cast_mode.startswith("int") else None,
        "n_cells": n_cells_total,
        "n_records": len(records),
    }
    torch.save(flat, out_file)

    # Headline summary for the printer
    r = flat["r"]
    abs_r = r.abs()
    n_zero = (r == 0).sum().item()
    return {
        "n_records": len(records),
        "n_cells": n_cells_total,
        "r_zero_frac": n_zero / max(n_cells_total, 1),
        "r_abs_mean": abs_r.float().mean().item(),
        "r_abs_p99": torch.quantile(abs_r.float(), 0.99).item(),
        "r_abs_max": abs_r.max().item(),
        "r_min": r.min().item(),
        "r_max": r.max().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cast-mode", default="tight",
                    choices=["tight", "int8", "int16", "int24"])
    ap.add_argument("--B-op", type=int, default=9,
                    help="bit shift for tight cast (default 9 = lossless on fp8e4m3)")
    ap.add_argument("--compute-features", action="store_true",
                    help="also compute per-cell sum_pos/sum_neg/abs_sum/topk_xw "
                         "(expensive, but enables sign-split & outlier residual models)")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    rec_dir = Path(args.records_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(rec_dir.glob("prompt_*.pt"))
    print(f"Found {len(files)} prompt files in {rec_dir}")
    print(f"Cast mode: {args.cast_mode}; features: {args.compute_features}")

    summary = []
    t0 = time.time()
    for f in files:
        out_f = out_dir / f"residuals_{f.stem}.pt"
        s = build_one_prompt(
            f, out_f, args.cast_mode, args.compute_features,
            top_k=args.top_k, B_OP=args.B_op,
        )
        elapsed = time.time() - t0
        s["file"] = f.name
        summary.append(s)
        print(f"  {f.name}: r_zero={s['r_zero_frac']:.4f} r_abs_mean={s['r_abs_mean']:.3g} "
              f"r_abs_p99={s['r_abs_p99']:.3g} r_abs_max={s['r_abs_max']} "
              f"r_range=[{s['r_min']}, {s['r_max']}] "
              f"n_cells={s['n_cells']} ({elapsed:.1f}s)")

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "cast_mode": args.cast_mode,
        "B_OP": args.B_op,
        "compute_features": args.compute_features,
        "files": [Path(s["file"]).name for s in summary],
        "summary": summary,
    }, indent=2))
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
