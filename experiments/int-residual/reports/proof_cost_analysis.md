# Proof-cost analysis for residual models

Baseline. The verifier already certifies the integer matmul `Y' = X' W'`
via Freivalds: pick a random vector `r ∈ F^D`, check `Y'·r =? X'·(W' r)`.
Cost is **two matvec multiplications** of length K (plus a vector-vector
multiplication) per check, so O(n²) where n is the dominant of {T, D, K}.
Repeated κ times for soundness 2⁻ᵏ.

Residual-model cost is added **on top** of this. We compare each model's
per-output-cell work and any extra commitments it introduces.

## Conventions

- T = output rows (tokens per matmul call)
- D = output columns (output features)
- K = inner dim (input features)
- N_cells = T × D
- "1 matvec" = the cost of multiplying a K-vector by a [D × K] matrix; in
  the proof, that's one extra Freivalds-style certification.
- "1 K-sum per cell" = N_cells × K extra products; equivalent to one
  extra matmul of cost identical to Y'.

## Models and their costs

| Model | Features | Per-cell ops | Extra commitments | Extra K-sums | Bit-exact (smoke) |
|---|---|---|---|---|---|
| R1 zero       | none                                | 0           | none      | 0  | 0.12% |
| R2 const      | global mean                         | 1 add       | 1 int     | 0  | 0.12% |
| R3 family     | per-family means                    | 1 lookup    | 8 ints    | 0  | 0.12% |
| R4 affine_Y'  | Y'                                  | 1 mul + 1 add | 2 floats | 0  | 0.13% |
| R5 (fam,blk,d)| per-(family, block, d) mean         | 1 lookup    | ~10k ints | 0  | 0.13% |
| R7 scale_lut  | s_X·s_W·Y'                          | 2 mul       | 1 float   | 0  | 0.13% |
| R7b           | affine in (s_X·s_W·Y', Y')          | 3 mul + 2 add | 3 floats | 0  | 0.12% |
| R8 sign_split | a + Σ X' over signs, …              | 5 mul + 4 add | 5 floats | 2 (sum_pos, sum_neg) | 0.12% |
| R9 outlier    | clip counts + absmax + Y' + s·Y'    | 6 mul + 5 add | 7 floats | 0  | 0.13% |
| R10 deltas    | per-row/col δ summaries + cross-prods | 10 mul + 9 add | ~10 floats | 0 (precomputed per-row/col)  | 0.11% |
| R11 mixed     | X'·δ_W, W'·δ_X, δ_X·δ_W              | 3 mul + 3 add | 4 floats | 3 | 13.1% |
| R11a mixed-2  | X'·δ_W, W'·δ_X                       | 2 mul + 2 add | 3 floats | 2 | 9.2% |
| R11_fixed     | X'·δ_W + W'·δ_X − δ_X·δ_W (no fit)   | 2 add        | 0 (rule) | 3 | 13.1% |
| R12 sign-δ    | X'·sign(δ_W), sign(δ_X)·W'           | 2 mul + 2 add | 3 floats | 2 (1-bit δ) | 0.25% |

## Interpretation

The cheap-feature ensemble (R1–R10) all sit at 0.1–0.2% bit-exact and
r_abs_mean ≈ 273 — i.e., they recover nothing of the residual structure
beyond rounding-zero base rate. This is the **null result for Luke's
LUT-on-cheap-features critique** specifically in the int8 cast regime:
no function of (Y', s_X, s_W, per-row/col operand statistics) predicts the
residual to better than the base rate, because the residual is dominated
by per-product rounding terms (δ_X, δ_W) that are **not summary statistics
of the operands** — they're the fractional parts after the cast, which
depend on the individual codepoints.

The exact decomposition

> r = X'·δ_W + δ_X·W' − δ_X·δ_W − rounding(Ỹ)

immediately drops the residual by 100× (R11/R11_fixed: r_abs_mean ≈ 2.78,
bit-exact ≈ 13%) at the cost of **3 extra Freivalds-checkable K-sum
operations per output cell** — i.e. ~3 extra matmuls of the same cost as
Y'. The remaining 2.78 floor is **fp32 accumulator drift in the captured
Y**; recomputing Y in fp64 from operands collapses r to ≤ 0.5 (the Ỹ
rounding floor).

`δ_X` and `δ_W` are deterministic from `X_q` and `W_q` via the cast rule;
the prover commits to them implicitly when committing to the operands.
So R11 adds no new commitment, just extra proof work.

## Cost trade-off summary

- **R11_fixed**: 3 extra matmuls, no learned params. The "proof-friendly"
  point. Closes ~100× of the residual exactly.
- **R11a**: 2 extra matmuls, +0.86 r_mean vs R11. Drops the second-order
  δ_X·δ_W term, which is bounded by K·0.25 in absolute value but
  empirically small (since δ_X and δ_W are weakly correlated).
- **R12**: 2 extra sign-matmuls (1-bit δ commit). r_abs_mean 136 — gives
  about 2× improvement over no correction but loses the magnitude
  information. Not worth it.

The 2.78-floor from fp32 accumulator drift is **independent of the
residual model**. To go below it, the verifier would need either: (a) the
production kernel to use a higher-precision accumulator (fp64, exact int);
or (b) a residual model that consumes the actual reduction trajectory of
the float kernel — i.e., partial-sum state — which is the source #6
"reduction order" feature in the planning doc, and is expensive to
commit.

## What R11_fixed does NOT cover

- **Saturation (clipping)** in the operand cast. The int8 cast clips to
  [-127, 127]. For Qwen2.5-0.5B-fp8E4M3 → int8 (β = 127/448 ≈ 0.283), the
  pre-cast values are in [-127, 127] by construction, so no clipping
  occurs. R11_fixed exactness depends on this. If we cast a *higher-range*
  float operand into a tighter int range, clipping introduces a bias term
  that R11_fixed does not capture (because δ_X = X_prime - X_exact ≠
  round(X_exact) - X_exact when clipped). For our setup, this is moot.

- **Block-scaling along k** (e.g. NVFP4, MXFP8). The teacher applies a
  separate scale per block of k, so the residual algebra picks up an
  additional Σ_b (s_X[b] s_W[b] − const) · partial_sum_b term. Source #1
  from the plan. Will need a separate study on FP4 teachers.

- **Output requantization**. We capture Y as the matmul accumulator
  output, before any next-layer fp8 quantization. If the production
  pipeline requantizes Y before the next layer, the residual we'd want is
  Y_quantized minus the int-matmul output, which is a different (larger)
  function. Separate experiment.

- **Bias add**. We subtract the bias at capture time so the matmul-only
  residual is isolated. Adding the bias back in the proof is a free
  per-(output) operation.

## Bottom line for the proof system

R11_fixed is the minimum-cost residual model that closes the gap modulo
fp32 reduction noise. It costs:

- **3 extra Freivalds-checkable K-sum matmuls** over (X', δ_W),
  (δ_X, W'), (δ_X, δ_W).
- 0 fitted parameters.
- 0 extra commitments beyond `δ_X = X_prime - X_exact` (deterministic).

For a workload that does N matmul calls, this triples the matmul work
in the proof. Whether this is acceptable depends on the proving system's
Freivalds-check amortization (the random vectors can be shared across
matmuls within a layer/block, reducing the constant factor).

The cheaper R11a (drop δ_X·δ_W) costs 2 extra matmuls and gives up about
30% in bit-exact rate (13.1 → 9.2%) and 30% in r_abs_mean (2.78 → 3.64).
This may be the right ergonomic trade for moderate-precision proofs.

## Held-out validation (r11a_val, 2026-05-13)

Same numbers replicate on a proper cross-prompt + cross-block split. Capture:
Qwen2.5-0.5B, 4 wikitext prompts × blocks {0, 11, 23} × T=128. Train on
prompts {0,1}, eval on {2,3}. ~9.2M val cells.

| Model      | abs_mean | p99 | worst | bit-exact | extra K-sums |
|---|---|---|---|---|---|
| R1         | 270.17   | 975 | 3071  | 0.13%     | 0 |
| R11        | 2.64     | 12  | 59    | 14.97%    | 3 |
| R11a       | 3.55     | 14  | 63    | 9.64%     | 2 |
| R11_fixed  | 2.64     | 12  | 59    | 14.98%    | 3 (no fit) |

R11_fixed and R11 are bit-identical (fit coefs (0.008, 1.000, 1.000, −0.999)
→ rounding to (0, 1, 1, −1) is exact). Held-out R11=2.64 vs same-prompt
smoke R11=2.78 → the gap is sample noise, not generalization failure.

### `down_proj` flips families

Per-family abs_mean on the val split:

| family | K     | R1 abs_mean | R11 abs_mean | R11a abs_mean |
|---|---|---|---|---|
| q      | 896   | 234 | 2.59 | 3.36 |
| k      | 896   | 245 | 2.92 | 3.60 |
| v      | 896   | 227 | 2.15 | 3.01 |
| o      | 896   | 243 | 2.75 | 3.45 |
| gate   | 896   | 261 | 2.90 | 3.61 |
| up     | 896   | 261 | 2.51 | 3.30 |
| down   | 4864  | **442** | **1.95** | **5.02** |

Under R1, `down_proj` is the worst family (K is 5.4× larger). Under R11
the algebra cancels K-scaling and `down_proj` is the *best*. Under R11a
it's *worst again* — the dropped δ_X·δ_W term is a K-sum whose
variance grows with K, so dropping it disproportionately hurts the
larger-K layer. Practical implication for the proof: keep δ_X·δ_W on MLP
`down_proj`, drop it elsewhere. That costs 3 extra matmuls on `down_proj`
(out of 7 matmul families per block) and 2 on the rest — a weighted
average closer to R11's accuracy without paying the full 3-matmul tax
everywhere.

### Hybrid R11 (`R11_hybrid_byK`)

Implemented as `model_R11_hybrid_byK` in `fit_residual_models.py`. Result
on the same held-out split:

| Model | extra K-sums (avg per family) | abs_mean | bit-exact |
|---|---|---|---|
| R11    | 3            | 2.64 | 14.97% |
| R11a   | 2            | 3.55 |  9.64% |
| Hybrid | (6·2+1·3)/7 ≈ 2.14 | 3.34 | 10.78% |

Hybrid recovers ~22% of the R11a→R11 gap for ~7% extra cost over R11a.

## X/W asymmetry and the cheap-1-matmul operating point (`r11a_val`)

A full Pareto sweep of the algebra-based variants:

| Model | Features | extra K-sums | abs_mean | bit-exact |
|---|---|---|---|---|
| R1 baseline    | none                              | 0          | 270.17 | 0.13% |
| R11b_X         | X'·δ_W only                       | 1          | 253.00 | 0.13% |
| R11b_W         | W'·δ_X only                       | 1          |  87.44 | 0.41% |
| R12            | sign(δ_W)·X' + sign(δ_X)·W'       | 2 sign     | 134.97 | 0.25% |
| R12a_X         | X'·δ_W + sign(δ_X)·W'             | 1 + 1 sign | 126.20 | 0.27% |
| R12a_W         | sign(δ_W)·X' + W'·δ_X             | 1 + 1 sign |  44.36 | 0.80% |
| R11a           | X'·δ_W + W'·δ_X                   | 2          |   3.55 | 9.64% |
| Hybrid         | R11 on down_proj, R11a elsewhere  | ~2.14      |   3.34 | 10.78% |
| R11            | full algebra                      | 3          |   2.64 | 14.97% |

**The W-side dominates.** Dropping or sign-quantizing the δ_W direction
(X'·δ_W) costs little; the same operation on the δ_X direction collapses
the model to baseline.

### Empirical root cause

Per-cell statistics on a held-out prompt (4.4M cells):

|  | std | abs_mean | corr(·, r) |
|---|---|---|---|
| r          | 347.6 | 269.7 | 1.000 |
| X'·δ_W     | 113.4 |  85.7 | 0.328 |
| W'·δ_X     | 328.3 | 253.2 | **0.945** |
| δ_X·δ_W    |   2.9 |   2.2 | −0.052 |

X'·δ_W and W'·δ_X are nearly orthogonal (corr = 0.002) — they are two
independent components of r. **W'·δ_X has 8.4× the variance of X'·δ_W**
and a 0.945 correlation with r, so it carries ≈94% of the residual mass.

The asymmetry is rooted in int8 range utilization, not in δ bias:

|  | Σ_k |·|, mean |
|---|---|
| Σ_k |X'[t,k]| | 7580 |
| Σ_k |W'[d,k]| | 29915 |

W' uses the int8 range 4× more fully than X' under the standard
per-token (act) / per-row (weight) cast. Under per-token absmax scaling,
a single outlier in `x[t,:]` dominates s_X[t] and compresses the rest of
the row toward 0; per-row weight scaling is gentler. δ_X and δ_W
per-row sums are symmetric (|Σδ_X|≈8 ≈ |Σδ_W|≈8) — it is NOT a δ-bias
asymmetry.

### Rank-1 hypothesis falsified

Tried decomposing W'·δ_X as a rank-1 outer product
`δ̄_X[t] · Σ_k W'[d,k]` (both factors are O(K) per-row/per-col commitments,
no extra K-sums per cell). Result: abs_mean=270.05, matches baseline R1
exactly. The W'·δ_X signal is genuinely per-cell — it is the K-sum
*correlation* `Σ_k W'[d,k]·δ_X[t,k]` minus its rank-1 mean, and that
correlation cannot be summarized by per-row × per-col products. Result
filed at `reports/r11a_val_rank1_fit.json`. So **no zero-K-sum residual
model exists** in this regime.

## FP4 status

The current capture/build pipeline assumes per-token / per-row scales
and silently misapplies them to FP4 (which uses block-of-32 scales along
K). On `r11a_val_fp4`: R1 abs_mean=1109, R11 abs_mean=1073 — algebra
fails (3% improvement vs 100× on FP8). R11 coefs (1.005, 0.977, −1.14)
show the first-order terms are *locally* correct, but the dominant
residual lives in block-scale variation along K — the source-1 problem
from the plan. Proper handling requires:

1. Capture: save per-block scales `s_X[T, K/32]`, `s_W[D, K/32]`.
2. Build: define `Y_int[t,d] = Σ_k X'·W'` (cast cancels block scales
   inside each block of 32), but `Y` itself is `Σ_b s_X[t,b]·s_W[d,b] ·
   block_b_partial_sum` — so output cast can't reduce to a single
   per-cell scale.
3. Residual decomposition needs a per-block correction term; the
   residual model must either commit to per-block partial sums or
   tolerate larger error.

Out of scope for this session.

## T-independence (`r11a_val_T512`)

R11 floor at block-0, T~155 = 2.40 vs all-blocks T=128 = 2.64. The gap is
block selection, not T scaling. The per-cell residual floor (2.4–2.8) is
intrinsic — driven by fp32 accumulator drift in the captured Y, bounded
per cell.
