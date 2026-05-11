# Training: close the int-cast → reference gap

## Goal
Train the int-cast model's weight shadows (with STE) to push behavioral
divergence from the bf16 reference below the ZKP-acceptable threshold,
without retraining the model's actual capability.

**Pass criteria (in priority order):**
1. all-positions top-1 ≥ **99.9%**, stretch 99.99% (paper bar).
2. Token-DiFR Gumbel margin p99 ≤ 1e-3 (currently ~1e-2 at p99).
3. KL p99 ≤ 1e-4 (currently 3–7e-3).

Starting point per the baseline (see `../baseline-int-cast/reports/results-2026-05-11.md`):
Qwen2.5-0.5B all-positions top-1 = 0.9788, Gumbel margin mean 1.1e-3.

## Starting model
**Qwen2.5-0.5B first.** Rationale:
- ~10× faster iteration than the 8B models — full train + eval in <1 h.
- Direct apples-to-apples with Luke's `int-model-approximation` repo, so
  the result reconciles with prior work.
- Fastest failure signal: if Luke's "residual is in the norms, not the
  matmuls" claim is true, Linear-only training plateaus short of 99.9%
  within 20–30 min, and we pivot rather than burn an 8B run.

Graduate to Qwen3-8B and Llama-3.1-8B-Instruct only after the recipe
is dialed on 0.5B.

## What's trainable, what isn't
- **Trainable**: a fp32 *float shadow* `W_fp` per IntLinear, initialised
  from the original linear's weight. We do NOT train the int tensor
  directly; the int values fall out of `quantize_per_row(W_fp)` each
  step via STE.
- **Frozen**: RMSNorm, RoPE, lm_head bias (if any), embedding. Same
  scope as Luke. This is deliberate: any norm/embedding edit changes
  representation capacity, which is a different experiment.
- **STE**: `W_int_fake = (round(W_fp / s) * s).clamp(...)` with
  `W_int_fake = W_fp + (W_int_fake - W_fp).detach()` so gradients flow
  through `W_fp` as if the quant was identity. Activation quant uses
  the same STE trick at every forward pass.

## Loss
Pure logit L2 between teacher (bf16 reference, frozen) and student
(int-cast with float shadows), averaged over every position:

```
L = mean_t || logits_ref(x)[t] - logits_int(x)[t] ||_2^2
```

Mean reduction over both position and batch; no scaling by vocab. We
start with this single term — no per-matmul auxiliary loss (Luke uses
one; see "Differences from Luke" below).

**If logit-L2 alone plateaus before 99.9%**, fallback: add the
per-matmul normalized-MSE term from Luke's repo. That gives the
optimizer a localized signal at every Linear instead of a single
end-to-end gradient.

## Data
- `Salesforce/wikitext`, `wikitext-103-raw-v1`, train split (same as
  baseline and Luke).
- Filter to `len(text) >= 100` chars, tokenize to 512 with truncation.
- 10k prompts total → ~5M tokens of supervision signal. Cached on disk
  as a single `prompts.pt` so the dataloader is just an index shuffle.
- Eval set: held-out 200 prompts from train (different shuffle seed)
  + the 100 prompts used in the baseline (for direct comparison).

## Optimization
| | value |
|---|---|
| optimizer | AdamW, β=(0.9, 0.999), wd=0 on shadows |
| LR | sweep {1e-7, 1e-6, 1e-5} in three parallel runs |
| warmup | linear, 100 steps |
| schedule | cosine to 10% of peak over the full budget |
| batch | 4 prompts × 512 tokens = 2048 tokens/step |
| grad accum | 1 (no AMP — shadows are fp32 already) |
| clip | global-norm 1.0 |
| steps | 5000 (≈30–45 min on H100 for 0.5B) |
| seed | 42 |

## Eval cadence
- Every 250 steps: run `compare()` from `run_baseline.py` on the 100
  baseline prompts. Log top-1 (both aggregations), Gumbel margin mean
  and p99, KL p99, logit-L2 mean.
- Every 1000 steps: full eval (1000 prompts) + write a checkpoint
  (`step{N}.pt` containing each IntLinear's `W_fp` and the optimizer
  state).
- Early stop: 3 consecutive eval points with no improvement on
  all-positions top-1 → save final, stop.

## Schedule / budget
- One H100 80 GB; bf16 teacher + fp32 student shadows + optimizer state
  for 0.5B comfortably fits (≈8 GB used).
- ~30–45 min per LR for the 5k-step run. Three LRs in series ≈ 2 h.
- Pad budget to 3 h for cache prep + eval + checkpoint I/O. Inside
  the same 10 h compute window the baseline used.
- If 0.5B clears 99.9%, repeat on Qwen3-8B and Llama-3.1-8B (estimated
  3–5 h each — same step count, slower forward).

## Differences from Luke's training (`int-model-approximation`)
| | Luke | Us |
|---|---|---|
| primary loss | logit L2 + per-matmul normalized MSE | logit L2 only (Luke's aux is fallback) |
| positions | last token of each prompt only | every position |
| LR | 1e-7 | sweep {1e-7, 1e-6, 1e-5} |
| data scale | hundreds of prompts | 10k prompts |
| eval | last-token top-1 / KL on 8 prompts | dual-aggregation top-1, Gumbel margin, KL on 100–1000 prompts |
| trainable | float shadows of every Linear (same) | same |
| frozen | norms, embeddings (same) | same |

So: same fundamental approach (float shadow + STE on Linears, frozen
norms), simpler loss to start, more data, broader LR sweep, and a
much stricter eval. The per-matmul aux is the obvious thing to add
back if pure logit-L2 plateaus — it's a useful localizer signal.

## Implementation
Layout (sibling to baseline):
```
src/difr_expt/
  int_cast.py        # extend IntLinear with `W_fp: nn.Parameter` and STE forward
  train.py           # main loop, optimizer, eval cadence, checkpoint I/O
  cache_prompts.py   # one-shot wikitext tokenize → prompts.pt
experiments/train-int-cast/
  plan.md
  EXPERIMENT_LOG.md
  scripts/run_lr_sweep.sh
  data/{lr}/{step}.pt
  reports/results-YYYY-MM-DD.md
tests/
  test_train_step.py  # gradient flows through STE, fp shadow updates, eval works
```

Key code changes from current `int_cast.py`:
1. `IntLinear.__init__` gains `self.weight_fp = nn.Parameter(...)`
   (initialised from the float linear's weight). The `weight_int` /
   `weight_scale` buffers become *derived* — computed fresh in
   `forward()` via STE — rather than fixed at construction.
2. `from_linear` keeps the same signature but populates `weight_fp`.
3. Activation path keeps STE: `x_q = quantize_per_token(x); x_ste = x + (x_q.float() * s_x - x).detach()`.
4. Eval mode: set `self.training = False`, and switch the forward to
   the existing inference path (no STE) so the saved checkpoint runs
   in pure-int mode without modification.

## Risks / what could go wrong
- **Plateau short of 99.9%** (most likely). Luke saw ~10% reduction in
  logit L2 over his run, not a 100× one. Suggests residual error
  lives in the un-quantized parts (norms/RoPE). Pivot: either (a) add
  per-matmul aux, (b) try training norm scale parameters too, or
  (c) bite the bullet and quantize norms.
- **Catastrophic forgetting on capability tasks**. Logit-L2 to ref
  shouldn't change capability if it converges to zero, but a partially
  trained student could be worse than the untrained one on real tasks.
  Mitigation: pre-register a quick lm-eval-harness sanity check (e.g.
  ARC-easy 10-shot) on the final checkpoint, not as a training metric.
- **Quant scales going off-distribution as W_fp drifts**. The
  per-row absmax scale `sw[i] = max(|W_fp[i,:]|)/qmax` is recomputed
  every step, so a single outlier weight can blow up the scale and
  destroy precision on the other 14k values in the row. Mitigation:
  log per-layer `max(|W_fp|)` over training; if outliers appear, clip
  or add a small weight-decay-toward-grid term.
- **bf16 teacher non-determinism**. The reference is bf16 + cuDNN
  algorithms, so teacher logits aren't bit-stable across runs. Cache
  them once and read from disk for the whole sweep — otherwise the
  loss signal is noisy at the 1e-4 level we care about.

## Out of scope
- Quantizing norms / RoPE / softmax (separate research thread; flagged
  as the likely culprit for residual divergence, but addressed by a
  different intervention).
- Activation-DiFR training (paper §4.3) — Token-DiFR only here.
- KV cache quantization.
- Multi-GPU training. 0.5B and 8B both fit on one H100.
- vLLM integration; we're still on raw HF for clean module swap.
