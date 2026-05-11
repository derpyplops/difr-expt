# Baseline: int-cast vs float DiFR divergence

## Goal
Measure how close an int-arithmetic version of a model is to its float reference,
using the Token-DiFR metric from the paper (`docs/2511.20621.pdf`). This is the
de-risking experiment for whether ZK-proof-compatible (integer) inference can
replace float inference without meaningful behavioral drift.

**Pass criteria:** top-1 token match ≥ 99.9%, stretch 99.99%.
Luke's prior naive run hit ~99% without any training, with divergence concentrated
in normalization layers (matmul divergence was ~0). This baseline confirms that
number on the paper's models and gives us per-layer attribution for any later
training run.

## Models (in priority order)
1. **Llama-3.1-8B-Instruct** — start here, smallest + most documented.
2. Qwen3-8B — next, same scale, different arch family.
3. Qwen3-30B-A3B — stretch, MoE, paper's hardest case.

All three appear in the DiFR paper's eval suite (§5).

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
- 1k prompts from UltraChat-200k (paper §5.1 uses UltraChat). Prefill the prompt,
  generate up to 256 tokens. ~256k decode tokens total — well above the
  "few thousand" needed for DiFR to saturate at AUC ≥ 0.999 per Table 2.
- Reference sampling config (paper §5.1): bf16 weights, T=1.0, top-p=0.95,
  top-k=50, fixed PRNG seed.

## Metrics (recorded per token, aggregated per run)
| metric | aggregation | target |
|---|---|---|
| top-1 argmax match | mean | ≥ 0.999 |
| Token-DiFR margin (paper Eq. 1, Δ_max clipped at 99.9%ile) | mean | as low as possible |
| logit L2 (full vocab) | mean | reported, training signal |
| top-5 logit set overlap | mean | sanity check |
| per-layer logit-L2 contribution | one number per block | locate divergence (matmul vs norm vs residual accumulation) |

Per-layer attribution uses forward hooks: cache reference activations after each
transformer block, then re-run with int-cast layer-by-layer to see where
divergence enters.

## Implementation
- Engine: raw HF transformers + custom forward, not vLLM. vLLM's CUDA graphs
  make it hard to swap Linear modules cleanly, and we don't need throughput at
  this scale.
- Repo layout (created in this project):
  ```
  src/
    int_cast.py        # IntLinear module + model patcher
    metrics.py         # token match, DiFR margin, L2
    run_baseline.py    # main eval loop
  results/
    {model}_{date}.json
  ```
- Reuse what's in Luke's repo where possible (Daniel/Luke to share link;
  if not available within ~1 h, build from scratch — the surface area is small).
- Reference for the DiFR margin definition:
  paper §4.2 Eq. (1), Δ_max clipping per §4.2 pseudocode.

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
- Quantizing activations or KV cache.
- Activation-DiFR (paper §4.3) — Token-DiFR only here.
- vLLM integration.
- Replay-server / ZKP cryptographic plumbing.

## Open questions (answer before kicking off)
- Use Luke's existing repo as starting point, or rebuild? (waiting on link)
- Per-row vs per-channel scale for the LM head specifically (vocab dim is huge,
  per-row might be excessive — test both if time permits).
- Whether to use the differ codebase's own DiFR margin implementation
  (github.com/adamkarvonen/difr) or reimplement. Prefer using it for parity.
