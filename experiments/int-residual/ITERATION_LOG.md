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

- 2026-05-13 — **Held-out validation for R11a (2-matmul variant).**
  Captured `r11a_val/` = 4 wikitext prompts × Qwen2.5-0.5B blocks
  {0, 11, 23} × T=128 (21 records/prompt, ~4.5M cells/prompt, 782MB
  raw + 2.0GB residual+features). Cross-prompt cross-block split:
  train={0,1}, val={2,3}. Fit R1, R11, R11a, R11_fixed. Results
  (val): R1 abs_mean=270, bit-ex=0.13%; **R11=2.64, p99=12, worst=59,
  bit-ex=15.0%; R11a=3.55, p99=14, worst=63, bit-ex=9.6%**;
  R11_fixed=2.64 (identical to R11 — fit coefs are (0.008, 1.000,
  1.000, −0.999), the algebra is exact). Cost-benefit on R11a: drop
  1 of 3 K-sum matmuls, pay +34% abs_mean and lose ~36% of the
  bit-exact rate (15.0% → 9.6%). **Surprise**: under R1, `down_proj`
  is the *worst* family (abs_mean 442 vs ~250 elsewhere — K=4864 is
  5.4× the others); under R11 it becomes the *best* (1.95 vs ~2.6 —
  the algebra cancels the K-scaling); under R11a it becomes *worst
  again* (5.02 vs ~3.4 — δ_X·δ_W is a K-sum whose variance grows
  with K, so dropping it costs more on `down_proj`). This is a
  family-dependent argument for keeping the δ_X·δ_W term on
  MLP-intermediate layers but dropping it elsewhere; full-R11 only
  needs to pay 1 extra matmul there. Filed at
  `reports/r11a_val_fit.json`. Held-out R11=2.64 vs same-prompt
  smoke R11=2.78 — the gap is noise, not generalization failure.

- 2026-05-13 — **R11_hybrid_byK (R11 on down_proj, R11a elsewhere).**
  Added `model_R11_hybrid_byK` in `fit_residual_models.py` to test
  the cost-asymmetric hybrid suggested by the K-scaling observation.
  Per-block extra-matmul cost: hybrid = (6·2 + 1·3)/7 ≈ 2.14, vs R11=3
  and R11a=2. Result on r11a_val (val={2,3}): abs_mean=3.34 (vs R11a
  3.55, R11 2.64), p99=13, worst=63, bit-ex=10.8%. Recovers ~22% of
  the R11a→R11 gap for ~7% extra cost over R11a. Per-family confirms
  hybrid matches R11 on `down_proj` (1.95) and matches R11a elsewhere
  (3.0–3.6). Pareto-better than R11a but well short of R11. Filed at
  `reports/r11a_val_hybrid_fit.json`.

- 2026-05-13 — **FP4 test confirms the algebra is wrong without block
  scales.** Captured FP4 (`r11a_val_fp4/`) on the same 4 prompts ×
  blocks {0,11,23} × T=128. The current capture computes per-token
  `s_X` and per-row `s_W` from absmax/fp8_max — *wrong shape* for
  FP4 (MXFP4 uses one scale per block of 32 along K, so s_X is
  [T, K/32] and s_W is [D, K/32]). Built residuals with int8 cast
  using these aggregate scales; R1 abs_mean=1109 (4× the FP8 baseline,
  consistent with FP4's coarser codebook). **R11 abs_mean=1073 — barely
  better than R1 (3% relative improvement, vs FP8's 100× drop)**. R11
  coefs on FP4 = (−7.06, 1.005, 0.977, −1.14) — first-order coefs still
  ~1, so the per-product δ algebra is locally correct, but the dominant
  residual mass lives in *block-scale variation along K* (source #1
  from the plan) which is NOT representable as X'·δ_W or W'·δ_X with
  a single scale per row/col. To handle FP4 properly, the residual
  decomposition needs a per-block term: r ≈ Σ_b (s_X[t,b]·s_W[d,b] −
  c) · partial_sum_b. Out of scope for this session — capture+
  build_residuals need to be refactored to expose block scales.
  Filed at `reports/r11a_val_fp4_fit.json`.

- 2026-05-13 — **Larger T: floor is essentially flat.** Captured
  `r11a_val_T512/` = 4 prompts × block 0 only × max_len=512.
  Wikitext entries are short, actual lengths were 166/109/115/229
  (avg ~155 vs the T=128 cap on the earlier run). R1=268.85,
  R11=2.40, R11a=3.35, bit-ex(R11)=15.8%. R11 floor at block-0,
  T~155 = 2.40 vs all-blocks T=128 = 2.64 — the gap is block
  selection (block 0 alone has lower per-cell residual than the
  block 11/23 average), NOT T scaling. So the 2.4–2.8 floor is
  intrinsic per-cell (driven by fp32 accumulator drift bounded
  by O(√K · ε)), independent of T. The earlier per-family
  observation supports this: `down_proj` (K=4864) sits at 1.95
  under R11, *lower* than the K=896 families (~2.6) — accumulator
  drift doesn't grow linearly with K because Y itself grows with K
  and the per-cell rounding floor is bounded. Filed at
  `reports/r11a_val_T512_fit.json`. (Stretching T further would
  need a different prompt source — wikitext-103 paragraphs cap at
  ~200 tokens for Qwen2.5 tokenizer.)


- 2026-05-13 — **X/W asymmetry: W'·δ_X carries almost all the signal.**
  Swept the cheap 1- and 2-feature variants on `r11a_val_int8`
  held-out: R11b_X (keep only X'·δ_W) → abs_mean=**253.00** (~R1
  baseline 270 — useless); R11b_W (keep only W'·δ_X) → abs_mean=
  **87.44** (3× better than R1 at 1/3 of R11's cost). Sign-quantized
  variants: R12 (both sides sign) = 134.97; R12a_W (full W'·δ_X +
  sign(δ_W)·X') = **44.36**, beating R11b_W by 2× at +1 sign-matmul
  cost; R12a_X (full X'·δ_W + sign(δ_X)·W') = 126.20, much worse.
  **The asymmetry**: dropping or sign-quantizing the δ_X side hurts
  far more than the δ_W side. Hypothesis — activations are skewed
  (post-SiLU outputs are one-sided, post-RMSNorm has structure) so
  the conditional mean of δ_X given X' is non-zero and carries
  predictive signal, while weights are roughly symmetric (Gaussian
  around 0) so δ_W's contribution averages out across K. Practical
  consequence: a low-cost residual model that only carries W'·δ_X
  (1 K-sum) recovers most of the Y-magnitude structure. Filed at
  `reports/r11a_val_sweep_fit.json`. New Pareto front:
    1-matmul R11b_W:  abs_mean=87,    bit-ex=0.4%
    2-matmul R12a_W:  abs_mean=44,    bit-ex=0.8%  (1 full + 1 sign)
    2-matmul R11a:    abs_mean=3.55,  bit-ex=9.6%
    ~2.14-matmul hybrid: abs_mean=3.34, bit-ex=10.8%
    3-matmul R11:     abs_mean=2.64,  bit-ex=15.0%

- 2026-05-13 — **Rank-1 hypothesis falsified: there is no zero-K-sum
  residual model.** Tried R13 = LSQ over per-row × per-col interactions
  {dX_sum_t · Wp_sum_d, dW_sum_d · Xp_sum_t, plus abs-magnitude pairs},
  hoping that the W'·δ_X signal would decompose as
  `δ̄_X[t] · Σ_k W'[d,k]` (a rank-1 outer product of features already
  committed at O(K) cost per row/col, no per-cell K-sums needed). Result:
  R13 abs_mean=270.03, R13a_W abs_mean=270.05 — identical to R1 baseline.
  R14 (rank-1 features + full W'·δ_X) = 87.40, matches R11b_W (87.44) to
  rounding — confirms the rank-1 features add zero on top of the K-sum.
  Why this fails: `W'·δ_X = δ̄_X[t]·ΣW'[d] + Σ W'[d,k]·ε[t,k]` where ε is
  zero-mean noise. With Var(W')~2500, K~900, Var(ε)~0.083, the noise
  term has std ~432, dwarfing the rank-1 mean shift (~15). The K-sum
  signal is the *noise correlation between W' and δ_X*, not the
  product of their per-row/col aggregates. Useful negative result: no
  Freivalds-free residual model exists in this regime. Filed at
  `reports/r11a_val_rank1_fit.json`.

- 2026-05-13 — **Empirical diagnosis of X/W asymmetry (root cause: int8
  range utilization).** Computed per-cell statistics on one held-out
  prompt (4.4M cells, blocks {0,11,23}). Per-term std:
    r        : std=347.6, abs_mean=269.7
    X'·δ_W   : std=113.4, abs_mean= 85.7
    W'·δ_X   : std=328.3, abs_mean=253.2
    δ_X·δ_W  : std=  2.9, abs_mean=  2.2
  **W'·δ_X has 8.4× the variance of X'·δ_W.** Correlations with r:
  corr(r, W'·δ_X)=0.945, corr(r, X'·δ_W)=0.328, and the two terms are
  *orthogonal* (corr=0.002) — they're independent components of the
  total residual. Root cause: int8 range utilization differs.
  Per-row absolute K-sums:  Σ_k |X'[t,k]| ≈ 7580 vs Σ_k |W'[d,k]|
  ≈ 29915 — **W' uses the full int8 range 4× more fully than X'**.
  Under per-token absmax scaling, a single outlier in x[t,:] dominates
  s_X[t], compressing most other entries close to 0; per-row weight
  scaling is gentler because weight distributions are smoother. Result:
  X'·δ_W ≈ 4× smaller in magnitude than W'·δ_X for the same δ, and
  variance ratio (4²=16) approximately matches the observed std ratio
  (~8.4 — partial because |X'| and Var(X') don't scale identically).
  Per-family: corr(r, W'·δ_X) ranges 0.88 (o_proj) to 0.98 (down_proj);
  asymmetry is universal. δ_X/δ_W per-row means are symmetric (~8 each)
  — it is NOT a δ-bias asymmetry. **Implication for proof design**:
  for this quant scheme (per-token act, per-row weight) the W-side
  matmul dominates the residual model; the X-side term is mostly
  ignorable. A different quant scheme (per-block activation scaling
  or smaller activation outliers) would change the asymmetry.

- 2026-05-13 — **Asymmetry is cast-scheme dependent, not intrinsic.**
  Computed per-record |X'| sums under (current) per-token activation
  cast vs (hypothetical) per-block-of-32 activation cast on the same
  X_q tensors. Per-token Σ|X'|/T ≈ 5000–11000; per-block-of-32
  Σ|X'|/T ≈ 19000–137000 — a **3–14× recovery of int8 range**.
  Compared against per-row Σ|W'|/D ≈ 18000–112000, the per-block X'
  magnitudes match W' magnitudes. So under MXFP-style block
  activation scaling, both first-order terms (X'·δ_W and W'·δ_X)
  would have comparable variance and both would carry signal. The
  observed asymmetry is an *artifact of per-token absmax cast under
  activation outliers* (a single outlier in x[t,:] compresses the
  rest of the row toward 0). **Implication for proof systems**:
  - per-token-act cast (Qwen2.5-FP8-dynamic): 1–2 K-sum proof, W-side
    dominates, X-side can be dropped or sign-quantized cheaply.
  - per-block-act cast (MXFP*, smoothquant variants): full 2–3 K-sum
    proof needed; cost asymmetries from this work do not transfer.
  No actual cast-mode run since output-cast under per-block scaling
  has its own complication (same source-1 issue as FP4); this is a
  diagnostic-only result, but it cleanly explains the asymmetry.
