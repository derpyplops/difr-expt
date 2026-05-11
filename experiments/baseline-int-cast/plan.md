# Baseline: int-cast vs float DiFR divergence

## Goal
Measure how close an int-arithmetic version of a model is to its float reference,
using the Token-DiFR metric from the paper (`docs/2511.20621.pdf`). This is the
de-risking experiment for whether ZK-proof-compatible (integer) inference can
replace float inference without meaningful behavioral drift.

**Pass criteria:** top-1 token match ≥ 99.9%, stretch 99.99%.
Luke's prior naive run on Qwen2.5-0.5B reported ~99%; that number is a
last-token-only aggregation over a small sample (eval was 8 prompts × last
position = 8 binary observations). To make our numbers directly comparable
to his AND meaningfully measured, we report **both aggregations**: all positions
(strict, 10k+ observations) and last-only (matches Luke). See report.

Luke also claimed matmul divergence was ~0 and the residual lived in norms;
that's a per-layer attribution claim we have not yet verified here.

## Models (in priority order)
1. **Llama-3.1-8B-Instruct** — paper's primary, bf16 reference. ✅ done.
2. Qwen3-8B — same scale, different arch family. ✅ done.
3. Qwen2.5-0.5B — small + matches Luke's repo for direct comparison. ✅ done.
4. Qwen3-30B-A3B — paper's MoE stretch model. ⏸ deferred: needs sequential ref↔int
   loading to fit in 80 GB.

All four appear in the DiFR paper's eval suite (§5) or Luke's repo.

## Int conversion (full: weights + activations)
Quantize **both** weights and activations to int. Skip the weights-only halfway
step — that would measure something weaker than what ZKP-verified inference
actually needs.

Per-row symmetric for weights, per-token dynamic symmetric for activations:
```
# weights, once at load time
sw[i]    = max(|W[i, :]|) / qmax                  # per output channel
W_int    = round(W / sw).clamp(-qmax, qmax)       # stored

# activations, every forward
sa[t]    = max(|x[t, :]|) / qmax                  # per token
x_int    = round(x / sa).clamp(-qmax, qmax)

# matmul (target: int32 × int32 → int64 accumulate → float dequant)
y[t, i]  = sa[t] * sw[i] * sum_k W_int[i, k] * x_int[t, k]
```

**Effective bit width: 16**, not 32. This is forced by the int64 overflow
ceiling for the matmul accumulator: at hidden dim 14336 (Llama 3.1 8B MLP),
two 16-bit ints multiplied give a 30-bit value, and 14336 of those summed gives
~2⁴⁴ — well inside int64. Going to int32 effective width would risk overflow.
(Matches Luke's choice in `int-model-approximation`.)

**Compute path** (this is a measurement deliberate choice):
The dequant-then-float-matmul `(W_int * sw) @ (x_int * sa).T` is mathematically
equal to true int matmul plus float dequant *in exact arithmetic*, and differs
only by float reduction order in practice — which is the same kind of noise the
reference itself accumulates. We use it for the baseline so we can run on GPU
at normal speed. A `--true-int-matmul` flag (CPU fallback à la Luke's repo,
~2-3h per model) is available for verification but not the default.

**Out of scope for quantization**: RMSNorm, RoPE, softmax remain float. Same
as Luke's repo. Full ZKP eventually needs these too, but they're a separate
research thread; the matmul-quantization question is the one we're de-risking
this round.

## Dataset
- 100 prompts (current; want 1000+ for tail metrics) from Salesforce/wikitext
  (`wikitext-103-raw-v1`, train split). Switched from UltraChat because lmsys/
  UltraChat are gated and wikitext is what Luke's repo uses, giving direct parity.
- Teacher-forced over the prompt tokens (no generation), so per-token logits
  come straight from one forward pass per prompt. Faster than autoregressive
  generation and gives strictly more measurement positions.
- Reference: bf16 weights, fixed seed 42. Sampling parameters (T, top-p, top-k)
  enter the harness only via the Gumbel-margin metric where T = 1.0.

## Metrics (recorded per token, reported under TWO aggregations)
Every metric is reported under:
- **all positions** — strict; mean over every position in every prompt (~15k
  observations per model).
- **last position only** — last token of each prompt only (matches the
  aggregation in Luke's repo). Lets us reconcile directly with his numbers.

| metric | what it measures | aggregation reported |
|---|---|---|
| top-1 argmax match | does the int model pick the same token? | mean (both) |
| top-5 logit set overlap | overlap of top-5 candidate sets | mean (both) |
| logit L2 (full vocab) | unscaled raw divergence; training signal | mean, p50, p99, max (both) |
| KL(ref ‖ int) | distribution divergence | mean, p50, p99, max (both) |
| Token-DiFR margin (paper Eq. 1, Δ_max clipped) | post-Gumbel verification-relevant gap; this is the metric a verifier sees | mean, p50, p99, max (both) |

Per-layer logit-L2 attribution (one number per transformer block, via forward
hooks) is planned but not yet implemented. The idea: cache reference activations
after each block, then re-run with int-cast block-by-block to localise where
divergence enters (Luke's claim is norms; matmuls were ~0).

## Implementation
- Engine: raw HF transformers + custom forward, not vLLM. vLLM's CUDA graphs
  make it hard to swap Linear modules cleanly, and we don't need throughput.
- Layout:
  ```
  src/difr_expt/
    int_cast.py        # IntLinear + patch_model_int_cast + true_int_matmul toggle
    metrics.py         # top-1, top-5 overlap, logit L2, KL, post-Gumbel margin
    run_baseline.py    # teacher-forced eval, dual aggregation, JSON dump
  experiments/baseline-int-cast/
    plan.md
    EXPERIMENT_LOG.md
    data/{model}.json
    reports/results-YYYY-MM-DD.md
  tests/
    test_int_cast.py   # 10 cases
    test_metrics.py    # 8 cases
  ```
- Implementation references: paper §4.2 Eq. (1) for the Gumbel margin
  (reimplemented in metrics.py rather than importing adamkarvonen/difr — small
  surface, easier to test). Luke's `int-model-approximation` for the int conversion
  scheme and the int64-overflow-driven 16-bit cap.

## Compute
Single H100 80GB on vast, standard PyTorch image. Llama 8B bf16 fits in ~17 GB;
running reference + int-cast side by side stays under 40 GB. Plan: stock
`pytorch/pytorch:2.5.x-cuda12.4-cudnn9-devel` image launched with `--ssh`.

Run-time estimate: ~30 min/model for 1k prompts × 256 tokens at HF speed (no
batching), so ~1.5 h across the three models. Leaves time inside the 10 h window
for iteration.

## Out of scope for this baseline
- Training the int model to close the gap (separate experiment, after we have
  the baseline numbers).
- Quantizing the KV cache.
- Activation-DiFR (paper §4.3) — Token-DiFR only here.
- vLLM integration.
- Replay-server / ZKP cryptographic plumbing.

## Resolved decisions (originally open questions)
- **Starting point**: built our own harness rather than forking Luke's repo —
  smaller surface, easier to extend for dual aggregation. Used adamkarvonen/difr
  as a reference for the paper's Token-DiFR math, did not import it directly.
- **LM head scale**: per-row, same as every other Linear. No evidence that
  per-channel for the LM head specifically would help; deferred unless we hit a
  surprise in per-layer attribution.
- **DiFR margin implementation**: reimplemented in `metrics.py`, with unit tests
  validating the zero-margin-when-logits-equal and positive-margin-on-disagreement
  cases. Faster than wiring up vLLM.
