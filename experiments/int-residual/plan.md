# int-residual: learning the integer residual of an FP8 matmul

## Strategic context

Prior experiments (int-vs-fp8-ref, int-emulates-fp) showed that we can build
a fully-int student whose **logits** match an FP8 production teacher
bit-exactly when the operand grid matches (per-token fp8 act + per-row fp8
weight), and within ~1pp on PPL otherwise. But the matmul kernel in those
runs is still bf16 `F.linear` over fp8-rounded operands — not an integer
GEMM. Daniel's recurring criticism stands: "is the matmul kernel int, or are
you committing int operands and running float ops?"

This experiment attacks the kernel question directly, but at a single-matmul
granularity. We are not building a whole int model. We capture the operands
and outputs of individual FP8 matmuls during a normal forward pass, then
offline compute the *clean integer* matmul `Y' = X'W'` from the same
operands and study the per-cell residual

    r = Y' - cast(Y)

where `Y` is the production FP8 output. We then try to predict `r` from
cheap summaries of the operands. If a tiny residual model gets us bit-exact
agreement on `cast(Y)`, we have a cheaply provable proxy for the FP8 matmul
(int matmul is cheap to certify via Freivalds; the residual model is small
by construction).

## Core experimental loop

1. Run an FP8 production model on wikitext prompts. Intercept selected
   matmul calls. For each one, save (X, W, Y, scales, kernel state).
2. Fix integer cast rules — these are not learned. Apply them to every
   record to get X', W', Y' (the clean int matmul) and the integer-coordinate
   teacher target `Ỹ = cast(Y)`. Compute residual `r = Y' - Ỹ`.
3. Per output cell (t, d), build features from the data the proof would be
   allowed to see (X', W', Y', any committed kernel state, derived
   statistics). Fit a residual model `r̂ = R(features)`.
4. Evaluate on held-out prompts. Report absolute and signed error
   percentiles for `r̂ - r`. Estimate proof cost.

Each output cell of every captured matmul is one training example. A single
record with `Y` of shape `[T, D]` contributes `T·D` examples.

## Format choices

Production teacher: `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` (FP8E4M3 weights +
per-row scale, per-token fp8 dynamic activation). This is the closest
realistic target: cheap to run on a laptop, fixed grid, no block scales
along k (so source #1 from the planning doc is *not* the dominant one and
sources #2-#6 are the interesting ones). Starting here lets us isolate the
codebook/cancellation/clipping/accumulator effects without simultaneously
fighting block-scale variation.

The cast we use to define Y':
- **operand cast**: int8 with per-row weight scale + per-token activation
  scale, matching the operand grid the teacher already lives on. So
  `W' = round(W_fp8 / s_W)` and `X' = round(X_fp8 / s_X)` are exact —
  the fp8 codebook values dequantize to a discrete set, and absorbing the
  scales gives integers in `[-127, 127]`.
- **output cast**: `Ỹ = round(Y / (s_W · s_X))`. The scale per output cell
  `(t, d)` is `s_W[d] · s_X[t]`. After this cast, both `Y'` and `Ỹ` live
  in the same integer coordinate so `r = Y' − Ỹ` is an integer scalar per
  cell.

Choosing the operand cast to *match* the teacher's grid means
*operand-level quantization error is zero*. So when r ≠ 0 it must be from:
- accumulator order / kernel reduction trajectory (small with fp32 accum),
- output-side cast rounding (the round in `Ỹ = round(...)`),
- per-output-row rescaling effects (the teacher's `s_W` is per-output-row,
  not per-output-cell), and
- whatever the FP8 kernel does differently from a plain `X_fp8 @ W_fp8.T`.

If r is large here, we'll know we have a kernel-level effect; if r is small
(near {-1, 0, +1}), the bound on a cheap rule-based residual model is
tight.

## Records

We start with one model. Records to capture:

- For each `nn.Linear` in the transformer (or its FP8 equivalent), at every
  forward pass on the chosen prompts:
  - `X_fp8` — the per-token fp8 activation (after dynamic quant)
  - `s_X` — the per-token activation scale (shape [T])
  - `W_fp8` — the fp8 weight (post-cast at load)
  - `s_W` — the per-row weight scale (shape [D])
  - `Y_acc` — the matmul output BEFORE bias add (the GEMM accumulator
    output). bf16 in `F.linear`-based recipes; fp32 in real Hopper kernels.
  - `bias` — kept fp32 in production; we store separately so the residual
    model doesn't have to learn it.
  - Layer name, position in transformer, matmul type (q/k/v/o, gate/up/down).

We deliberately skip lm_head (production convention).

## Residual model space

Hand-built per-cell predictors, each rejecting any tolerance / approx-equal
check (hard reject per plan):

R1. `r̂ = 0` — baseline. Tells us how big r already is.
R2. `r̂ = a` — global constant.
R3. `r̂ = a[matmul_family]` — one constant per (q/k/v/o/gate/up/down).
R4. `r̂ = a + b · Y'` — affine in Y' itself.
R5. `r̂ = a[d]` — per-output-column constant (allowed: `s_W[d]` is committed,
    so a function of d is committed too).
R6. `r̂ = a[d] + c[t]` — additive per-column + per-row corrections.
R7. `r̂ = round( g(s_X[t], s_W[d], Y') )` — small explicit formula in the
    operand scales (proof-cheap; just a few mults per cell).
R8. Lookup over (sign(Y'), |Y'| bucket, scale-product bucket).
R9. `r̂ = round(α · |Y'|)` style — quantization round drift model.
R10. Sign-split summaries on the dot product:
     features per cell = (P_t,d, N_t,d) where
     P_t,d = Σ_{k: X'_t,k · W'_d,k > 0} X'_t,k · W'_d,k
     N_t,d = Σ_{k: X'_t,k · W'_d,k < 0} |X'_t,k · W'_d,k|
     Note: these require Σ work along k per cell, but they're integer sums
     that can themselves be Freivalds-checked, so they don't blow up proof
     cost. Trial a linear model on (P, N, Y'=P-N).

For each one: train on train-prompts, eval on val-prompts. Report:
- abs error mean / p50 / p99 / p99.9 / worst
- signed error mean / p1 / p99
- fraction of cells with r̂ - r = 0 (bit-exact rate)
broken out per matmul family.

## Proof-cost framing

The point of separating Y' from the residual model is that Y' is provable
in O(n²) via Freivalds. So we score the residual model by counting the per-
output-cell ops it requires:

- Constant / per-family / per-row / per-col: 0–1 lookups per cell.
- Affine in Y': 1 mult + 1 add.
- Affine in operand scales: 2-3 mults (s_X[t] is committed once per row of
  the batch; s_W[d] once per column).
- Sign-split P,N: requires committing P and N at proof time; each can be
  Freivalds-checked from the operand sign-masks, cost ≈ another O(n²)
  per matmul. Per-cell residual evaluation is then 2 mults + 1 add.
- Per-cell MLP: rejected if it touches accumulator state or per-product
  values; allowed if it consumes only committed per-cell features. Counts
  scale by hidden dim².

We do not optimize this; we just report it qualitatively per row.

## Phases

### Phase 0: smoke test
- Qwen2.5-0.5B fp8 dynamic on 4 wikitext prompts × 64 tokens, CPU.
- Capture matmul records for one transformer block (q/k/v/o/gate/up/down).
- Build residual dataset, verify r is plausible (mostly small integers).
- Fit R1-R5 and dump a results table.

Goal: end-to-end loop works.

### Phase 1: real data
- 50 wikitext prompts × 256 tokens, all transformer blocks, Qwen2.5-0.5B.
- Train/val split at prompt level.
- All residual models R1-R10. Per-family and pooled.

Goal: first real numbers in the deliverable table.

### Phase 2: vary the cast
- Vary output cast (round vs round-to-even, vs truncate). Check whether
  the residual disappears when we move to a different rounding rule.
- Try larger operand cast (int16 instead of int8) to see if r is
  fundamentally bounded by output-cast rounding or by something else.

### Phase 3: another teacher
- Run on Qwen2.5-7B fp8 dynamic to see whether the same residual model
  family extrapolates.
- Optionally on a block-fp8 model (Qwen3-8B) — that introduces source #1
  (block-scale-along-k) and is the harder case.

## Out of scope

- Building the actual SNARK / arithmetic circuit.
- Training a learned residual model with many parameters.
- Whole-model PPL evaluation — the per-matmul residual is the unit of
  study here, not end-to-end loss.
- Block-fp8 / NVFP4 teachers (Phase 3 is optional stretch).

## Files touched

- `experiments/int-residual/scripts/capture_records.py` — run model + dump
  records.
- `experiments/int-residual/scripts/build_residuals.py` — operand cast +
  Y' computation + residual dataset construction.
- `experiments/int-residual/scripts/fit_residual_models.py` — fit and
  evaluate R1–R10.
- `experiments/int-residual/data/*.pt` — captured records, train/val.
- `experiments/int-residual/reports/results-2026-05-13.md` — first table.
- No changes to `src/difr_expt/*` planned; we read from existing modules
  and operate on captured tensors.
