# Iteration log: int-residual

Append-only, one bullet per attempt / observation / micro-decision. Finer
granularity than `EXPERIMENT_LOG.md` (which is one bullet per
milestone/session). This log captures the actual iteration trail during the
6-hour autonomous run on 2026-05-13.

Format: `- YYYY-MM-DD HH:MM — <attempt or observation>: <what happened> :: <what I'll try next>`

## 2026-05-13

- 2026-05-13 — scaffold: created `experiments/int-residual/` with plan.md
  pointing at fake-quant FP8E4M3 teacher on Qwen2.5-0.5B (CPU-friendly, no
  download). Smoke-tested capture: 1 prompt × 32 tok × block-0 → 7 records,
  ~400k output cells, runs in ~2s on CPU.
- 2026-05-13 — cast choices documented:
    - **Tight cast** (B=9): X' = round(x_q · 512 / s_X), W' = round(w_q · 512 / s_W).
      Operand-lossless on the fp8e4m3 grid (all codes are integer multiples of
      2^-9). Residual captures only accumulator drift + output round.
    - **Lossy cast** (int8): X' = round(x_q · 127 / (s_X · 448)) so X' ∈
      [-127, 127]. Operand-lossy — this is what a Freivalds prover would
      run. Residual captures full quantization error.
  Run both. Tight cast bounds the "is this trivially LUT-solvable" question
  (Luke's critique).
- 2026-05-13 — note for Luke's LUT critique: R7 in the plan (`r̂ = round(α ·
  s_X · s_W · Y')`) is essentially a per-cell LUT keyed on scales+Y'. If R7
  nukes the residual to 0 on the tight-cast dataset, the answer is "LUT
  wins, residual modeling unnecessary in this regime." If R7 fails on the
  lossy-cast dataset, that's the evidence Daniel needs that
  cancellation/outlier features matter.
- 2026-05-13 — **tight-cast smoke result is a surprise**. Built residuals
  with B_OP=9 on 1 prompt × 32 tok × block-0 → r_abs_mean ≈ 9e6, max ≈
  1.5e8 (in int-coord), zero rate 0%. Algebra: the tight-cast output rule
  Ỹ = round(Y · 2^18 / (s_X · s_W)) AMPLIFIES the kernel's fp32
  accumulator drift by 1/(s_X·s_W) ≈ 1e11 into int coord. r in int coord
  ≈ Y' · K · 2^-23 ≈ 2^24 ≈ 10^7 — matches observation. So the tight cast,
  although "operand-lossless", has residuals dominated by fp32 reduction
  drift in Y. *Useful as a bound on the reduction-noise-only case* but the
  cast is not what a Freivalds prover would actually run. Switching the
  primary cast to int8 (`X' = round(x_q · 127 / (s_X · 448))`), where r
  contains real, structured quantization error from the codebook collapse
  to 256 codes.

- 2026-05-13 — **int8-cast smoke: simple residual models do nothing**.
  Fit R1–R9 on 1 prompt × block-0 × 7 matmul families (405k cells). All
  models cluster around r_abs_mean ≈ 273, p99 ≈ 996, worst ≈ 3043,
  bit-exact rate ≈ 0.1%. Conclusion: features that are functions of
  (Y', s_X, s_W, ΣX, ΣW, n_clip, x_absmax, w_absmax, sum_pos, sum_neg)
  carry **almost no information** about r. This rules out Luke's
  "scale-LUT" approach (R7) on int8 cast specifically.

- 2026-05-13 — **Algebraic derivation, then R11 hits the ceiling.** Let
  c_X = x_q/s_X be the fp8 codepoint and β = MAX/FP8_MAX (MAX=127 for
  int8). Then X' = round(β·c_X), δ_X = X' − β·c_X, with |δ_X| ≤ 0.5.
  Expanding Y' = ΣX'·W' and Ỹ ≈ β²·Σc_X·c_W:
      r = Y' − Ỹ = X'·δ_W + δ_X·W' − δ_X·δ_W − rounding(Ỹ)
  This is an EXACT decomposition modulo the ≤0.5 output rounding (and
  any accumulator drift if Y wasn't computed in exact arithmetic). Fit
  R11 = a + b·(X'·δ_W) + c·(δ_X·W') + d·(δ_X·δ_W) by LSQ on 1 prompt:
  coefs ≈ (−0.02, +0.999, +0.999, −0.997) ≈ (0, 1, 1, −1), as the
  algebra predicts. r_abs_mean drops to **2.78** (from 273 → 100×
  reduction). p99 = 11, worst = 45, bit-exact = 13.1%. Per-family:
  v_proj has the lowest residual (1.94, 18% bit-exact); q_proj the
  highest (3.62, 9.6%). The remaining 2.78 floor is fp32 accumulator
  drift in Y; computed Y in fp64 from the captured operands and
  confirmed the residual collapses to ≤ 0.5 modulo Ỹ rounding.

- 2026-05-13 — **Proof-cost note for R11.** Each of X'·δ_W, δ_X·W',
  δ_X·δ_W is a K-sum per output cell — same cost as the base matmul.
  So R11 is ~3 extra Freivalds-checkable int matmuls on top of Y'.
  Not free, but small constant overhead. δ_X, δ_W are deterministic
  from X_q, W_q (rounding direction of fractional part); no new
  commitments needed.

- 2026-05-13 — **What's interesting going forward**: (a) try R11
  variants that drop the smallest term δ_X·δ_W → 2-matmul cost. (b)
  approximate X'·δ_W with sign-quantized δ_W (1 extra matmul). (c)
  see if larger T/K and per-prompt distribution shifts the floor.
  (d) report per-family breakouts. (e) repeat on FP4 — algebra is
  different there (block scales along k means r also picks up scale
  field variation, source #1 from the plan).
