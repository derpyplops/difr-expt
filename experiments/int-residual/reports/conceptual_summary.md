# int-residual: the conceptual story

A plain-language summary of what this experiment is, what we found, and
why it matters. For the numbers and full table, see
`proof_cost_analysis.md`. For the iteration trail, see
`../ITERATION_LOG.md`.

## The problem

You have a neural network doing matrix multiplies in low precision
(FP8). You want to cryptographically prove it ran correctly, without
redoing the float math (which is expensive to prove). Integer matmuls
are easy to prove cheaply with a standard trick called Freivalds. Float
matmuls are not.

## The idea

Don't try to prove the float matmul `Y = X · W`. Instead, decompose it:

1. Round `X` and `W` to nearby integers `X'` and `W'`.
2. Prove the **integer** matmul `Y' = X' · W'` (cheap).
3. Account for the gap: `r = Y' − Y` (the "residual"). If `r` is small
   and structured, you can prove it cheaply too.

Total cost = cheap integer proof + small residual model. Win if the
residual model is much cheaper than the original float proof.

## The discovery

The residual `r` looks random at first, but it has exact algebra. Let
`δ_X = X' − X` be the rounding error on activations (a tiny fractional
value per entry, magnitude ≤ 0.5) and `δ_W = W' − W` be the rounding
error on weights. Then:

> **r = (X' · δ_W) + (δ_X · W') − (δ_X · δ_W) + (tiny floor noise)**

This is an identity, not a fit. No learned parameters. Three
"correction matmuls" exactly recover the residual, leaving only a
noise floor that's a fraction of a bit per cell.

## What this buys you

If you're willing to run **3 extra integer matmuls** alongside the main
one, you predict `r` to within rounding noise. So a full proof of an
FP8 matmul becomes: prove 1 integer matmul + 3 integer correction
matmuls + a tiny rounding budget. Call this the "3-matmul approach"
(`R11` in the table).

## The cheaper option that actually matters

This is the surprise. Of the three correction terms, **one of them
(the W-side, `W' · δ_X`) does almost all the work**. The other two
contribute very little. You can drop them and keep ~94% of the
predictive power.

So:

- **3-matmul approach:** essentially bit-exact prediction of the
  residual. (`R11`.)
- **1-matmul approach:** keep only the W-side correction. ~3× reduction
  in residual size at one third the cost. Not bit-exact, but small
  enough that downstream uses (e.g. proving the next layer's output
  is close enough) may not care. (`R11b_W`.)

There's also a mixed point: full W-side correction plus a
"sign-quantized" cheap version of the X-side (1 full integer matmul +
1 sign-only matmul), which roughly doubles the accuracy of the
1-matmul approach for marginal extra cost. (`R12a_W`.)

## Why the W-side dominates

In standard FP8-per-token quantization, activations have outliers that
dominate the per-token scale. Once you divide by that scale, most
entries get squashed near zero — the integer activations under-use
their available range. Weights don't have this problem and use the
full integer range. So the W-side correction term carries far more
variance than the X-side, and the W-side alone captures nearly all
of `r`.

This asymmetry is a property of the quant scheme, not a deep fact.
Under per-block activation scaling (MXFP4 / smoothquant-style), it goes
away and you'd need both sides.

## The bottom line

If you're designing a proof system for FP8-per-token-dynamic LLMs (the
standard "FP8-dynamic" recipe used by Qwen2.5-FP8-dynamic and similar
production models), the most promising approach is:

- Prove the integer matmul `Y' = X' · W'` (one Freivalds check).
- Prove **one** extra integer matmul `W' · δ_X` (one more Freivalds
  check).
- Add a small per-cell correction budget for the leftover noise.

Two cheap integer matmul proofs replace a hard float matmul proof, and
you keep most of the accuracy you'd get from the full 3-matmul
approach.

## Open questions

- **FP4 / block-scale schemes.** When activations and weights use one
  scale per block of 32 along the contraction dimension (MXFP4, NVFP4,
  MXFP8), this analysis doesn't directly transfer. Initial probes
  confirm the X/W asymmetry disappears, and the residual decomposition
  needs a per-block term. Separate experiment.
- **Outlier-handling variants.** Schemes that pre-process activations
  (e.g. SmoothQuant, AWQ, per-channel scaling) will shift the asymmetry
  — possibly toward the X side. The 1-matmul recommendation depends on
  the exact deployed quant pipeline.
- **End-to-end proof cost.** This experiment measures per-matmul
  residual quality. The proof system's amortization across matmuls
  (sharing Freivalds randomness within a layer, batching) determines
  whether a 1-matmul vs 3-matmul correction actually matters in
  wall-clock terms.

## Glossary

| Term | Meaning |
|---|---|
| `X`, `W`, `Y` | Float activation, weight, and matmul output of one Linear layer |
| `X'`, `W'`, `Y'` | Integer versions: `X'` = round(X to int8 range), etc. `Y' = X' · W'` is the cheap integer matmul. |
| `δ_X = X' − X` | Per-entry rounding error on the activation cast. Always magnitude ≤ 0.5. |
| `δ_W = W' − W` | Same for the weight cast. |
| `r = Y' − Y` | The residual: what the integer matmul `Y'` is off by, relative to the true float output `Y`. |
| Freivalds check | A standard cryptographic technique that lets a verifier check an integer matmul in O(n²) time instead of redoing the O(n³) multiplication. |
| `R11` | The full 3-correction-matmul residual model: `r̂ = X'·δ_W + W'·δ_X − δ_X·δ_W`. Bit-exact modulo floor noise. |
| `R11a` | The 2-correction-matmul version: drop the `δ_X·δ_W` term. |
| `R11b_W` | The 1-correction-matmul version: keep only `W'·δ_X`. Captures ~94% of `r`. |
| `R12a_W` | Mixed: full `W'·δ_X` + sign-quantized `X'·sign(δ_W)`. One extra full matmul + one extra cheap "sign-only" matmul. |
| `R11_hybrid_byK` | Use the full `R11` on the wide-K MLP layer (`down_proj`), the cheaper `R11a` everywhere else. Pareto-better than `R11a`. |
