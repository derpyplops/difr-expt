# Full int model: multi-approach exploration

## Goal

Produce a fully ZKP-compatible integer model — every op in the forward pass representable as integer arithmetic with public scales — that hits **top-1 ≥ 99.9%** against the fp32 reference on at least Qwen2.5-0.5B, and ideally on Qwen3-8B and Llama-3.1-8B-Instruct.

Starting state (already done): the matmul portion. Every `nn.Linear` is replaced with `IntLinear` at b=24, validated bit-exact (top-1 = 1.0000) against fp32 reference. The float-equivalent forward path is mathematically identical to literal int execution; the literal int path was validated on Qwen2.5-0.5B (current run pending).

What's still float in our forward:
- **RMSNorm**: `y = x · γ · (mean(x²) + ε)^(-1/2)` — has sqrt and division.
- **Softmax**: `exp(x_i) / Σ exp(x_j)` — has exp and division.
- **SiLU / SwiGLU**: `x · sigmoid(x)` — has exp.
- **Attention matmuls**: `Q @ K.T` and `P @ V`, computed as float `@` operators inside HF's attention modules (not `nn.Linear`, so our `IntLinear` patch missed them).
- **RoPE rotation**: cos/sin × position. Trig values are precomputable (public tables); the rotation is just multiply + add and is easy to int.

## Pass criteria

1. **Primary**: all-positions top-1 ≥ 99.9% vs fp32 reference.
2. **Secondary**: Gumbel-margin p99 ≤ 1e-3 (DiFR verification-relevant).
3. **Validation**: at least Qwen2.5-0.5B run end-to-end with literal int execution (`--true-int-matmul`-style for all ops) and confirmed to match the float-equivalent measurement within ULP noise.

Stretch: 99.99% top-1. Architecture coverage: all three target models (Qwen2.5-0.5B, Qwen3-8B, Llama-3.1-8B-Instruct).

## Approaches to try

Five approaches arranged from cheapest-likeliest-to-work to most-expensive-most-flexible. We'll fan out to multiple boxes; A/E run in parallel with B/C since they don't share state.

### Approach A — Pure int approximation, no training
Replace each non-matmul op with an int-friendly version. Measure accuracy without changing any weights.

- **Softmax**: subtract max → 1024-entry `exp` lookup → integer sum → Newton-Raphson reciprocal.
- **RMSNorm**: square + sum (int) → Newton-Raphson `1/sqrt` with LUT seed → multiply by quantized `γ`.
- **SiLU**: 4096-entry `sigmoid` lookup → integer multiply.
- **Attention matmuls**: wrap `Q @ K.T` and `P @ V` as `IntMatmul` modules (no learned params; per-token activation quant on both sides + int matmul + dequant via per-token scales).
- **RoPE**: precompute cos/sin per position as fp32 tables, quantize, then rotation is a 2×2 int matmul per dim.

This is the most informative first measurement. If the model survives, we don't need any training.

Expected effort: ~half a day to a day of code (HF model surgery is the bulk). Compute: ~$5 across all three models for eval-only runs.

### Approach B — Train γ and Linear biases to compensate (the hybrid)
Build on A. Freeze `IntLinear` matmuls (already exact). Make RMSNorm `γ` and Linear biases trainable (~1M params for 8B). Train with logit-L2 distillation vs fp32 reference.

Rationale: each replaced op introduces small structured error; γ and biases are the small upstream lever that can compensate.

Expected effort: builds directly on A. Compute: ~$10 for 0.5B, ~$30 each for 8Bs.

### Approach C — Full QAT including the int approximations
Like B, but also unfreeze the `IntLinear` `weight_fp` shadows (fp32 STE training of weight values). Every approximation in the forward has STE backward. Train with logit-L2 + per-matmul aux loss (Luke-style).

Rationale: if γ + biases are insufficient degrees of freedom, the matmul weights themselves get a tiny adjustment to absorb the residual approximation error.

Expected effort: forks our existing `train.py` (already supports IntLinear QAT). Compute: ~$15 for 0.5B, ~$45 each for 8Bs.

### Approach D — Higher-precision non-matmul approximations
Same as A but with more aggressive approximations:
- Softmax lookup at 16k or 65k entries (vs 1k in A).
- RMSNorm with 3 Newton-Raphson iterations (vs 2).
- Softmax bins by `max(x)` range (separate exp tables per bin).
- Higher-precision attention matmul accumulator (int128 emulated via int64 split).

Test which knob matters most. If A almost works but is borderline, this is the cheapest way to push it over the line without training.

Expected effort: parameter sweep over A's approximations. Compute: ~$10.

### Approach E — Mixed precision per op
Keep softmax inputs at higher bit width (b=30+) where the ops are most sensitive; keep MLP intermediates at b=24. Track which ops contribute most to logit error by ablating one at a time (replace just softmax with int, keep others float; then add RMSNorm; etc.).

Rationale: the bit-width sweep on matmul showed that 8B models don't need b≥30 there; maybe one or two specific non-matmul ops are the limiting factor and warrant the precision bump.

Expected effort: small extension of A. Compute: ~$10 across the ablation matrix.

### Out of scope for this round
- **Architecture surgery** (replace SiLU with ReLU, softmax with sparsemax, RMSNorm with unit-scale): would require pretraining from scratch or extensive fine-tuning. Different research thread.
- **Activation-DiFR** (paper §4.3): stricter activation-level verification. Same framework, different metric. Tackle once Token-DiFR passes.
- **vLLM integration**: production engineering, orthogonal.
- **Building the actual ZK circuit**: compilation to Halo2/Plonky/etc. This experiment provides the spec; the circuit work is its own project.

## Models

Priority order:
1. **Qwen2.5-0.5B** — primary iteration target. All five approaches run on this first.
2. **Llama-3.1-8B-Instruct** — validation for winning approach.
3. **Qwen3-8B** — confirm architecture portability for winning approach.

Each model has its own HF attention/MLP/norm classes (`LlamaAttention`, `Qwen2Attention`, `Qwen3Attention`, etc.); the model-surgery code has to handle each separately. Expect to debug per arch.

## Parallelization plan

Spin up **up to three vast nodes**. Each independent approach goes to a separate node where helpful. Cheap H100 SXM (~$3.55/h) is the default; fall back to A100 SXM4 (~$1.57/h) for the smaller experiments.

```
Box X (H100 SXM, primary): 0.5B exploration
   ├── Phase 0: code + tests for int_ops.py
   ├── Approach A on 0.5B
   ├── (Phase A result → branch decision)
   ├── Approach B on 0.5B (if A < 99.9%)
   └── Approach C on 0.5B (if B insufficient)

Box Y (H100 SXM or H200, 8B validation): runs once winning approach is identified
   ├── Apply winning approach to Llama-3.1-8B
   └── Apply winning approach to Qwen3-8B (sequentially or on separate box)

Box Z (A100 SXM4, parallel ablation): runs in parallel with X
   ├── Approach D (lookup-table refinement) on 0.5B
   └── Approach E (mixed precision ablation) on 0.5B
```

Total parallel compute capacity: 2-3 nodes. Box Z is purely supplemental — it explores variations of A in parallel so we have data ready to inform B/C if needed.

Subagent strategy: each box gets its own subagent with explicit scope. They share local code via rsync but their boxes are independent. Coordination is by code commits to local + the eventual aggregated report.

## Phased schedule

### Phase 0 — Implementation (Box X, ~4-8 h of subagent work, ~$15)
1. Build `src/difr_expt/int_ops.py` with the int approximations: `IntRMSNorm`, `int_softmax`, `int_silu`, `IntMatmul`.
2. STE wrappers in `src/difr_expt/int_ops_ste.py` for backward through lookups + rounds.
3. HF model surgery in `src/difr_expt/patch_hf_model.py`: walker that swaps `LlamaRMSNorm` → `IntRMSNorm`, patches attention's softmax, replaces `SiLU` in MLP, replaces `Q@K.T` / `P@V` with `IntMatmul`. Handle each arch family.
4. Tests in `tests/test_int_ops.py`: per-op numerical accuracy < 0.5% relative on random inputs; HF surgery produces a working forward.
5. Extend `run_baseline.py` with `--int-nonmatmul` flag wiring the swapped modules.

### Phase 1 — Approach A measurement on 0.5B (Box X, ~30 min, ~$2)
Run baseline with `--int-nonmatmul` at b=24 fp32-ref. Measure all metrics.

Decision tree:
- **Top-1 ≥ 99.9%**: declare done; skip to Phase 4 (validation) and Phase 5 (8B).
- **Top-1 in [98%, 99.9%]**: Approach B is likely sufficient. Skip C. Proceed to Phase 2.
- **Top-1 in [90%, 98%]**: both B and C are worth running in parallel. Plus Approach D in parallel.
- **Top-1 < 90%**: approximations are too lossy. Run Approach D first (better approximations) before any training.

### Phase 2 — Approach B on 0.5B (Box X, ~1 h, ~$4)
Fork `train.py` → `train_nonmatmul.py`. Freeze `IntLinear` matmuls. Train norm γs + Linear biases with logit-L2 against fp32 reference. ~5,000 steps, batch=4, lr=1e-5.

Eval. Decision:
- **≥ 99.9%**: proceed to Phase 4-5.
- **In [98%, 99.9%]**: try Approach C (also unfreeze IntLinear shadows).
- **< 98%**: revisit approximations (Phase D in parallel).

### Phase 3 — Approach D and E in parallel (Box Z, ~2 h, ~$4)
Run in parallel with Phase 1 or 2 — Box Z is independent of Box X.

- **D**: sweep softmax LUT size {1024, 4096, 16384, 65536}, RMSNorm NR iterations {2, 3, 4}.
- **E**: ablate each int approximation: int-only-softmax, int-only-norm, int-only-silu, int-only-attn — measure each contribution to top-1 loss separately.

Outcome: a table of "marginal cost of integerizing each op." Tells us which ops most need higher precision or training-based compensation.

### Phase 4 — End-to-end literal int execution on 0.5B (Box X, ~3-4 h, ~$15)
Once a winning approach is identified, run the *entire* forward in literal integer mode: `IntLinear._true_int_path` for matmuls + the int approximations for everything else (no float fallback anywhere except for public scales).

This is slow (CPU int matmul throughout) but is the actual proof that the integer-arithmetic-only forward produces the trained accuracy. ~3-4 h for full 100-prompt eval.

Expected: matches the float-equivalent measurement within fp32 reduction-order noise (~1e-5 logit diff). If it doesn't match, we have a bug in the int approximation that the float-equivalent path is silently hiding.

### Phase 5 — Validate on 8B models (Box Y, ~6-8 h total, ~$30)
Apply the winning recipe to Llama-3.1-8B and Qwen3-8B. Run baseline eval first (no training); train γ+biases if needed.

May need to do per-architecture HF surgery debugging. Allocate ~2 h per model for "make it run" before measuring.

### Phase 6 — Aggregate & write up (~1 h, ~$0)
Collate results from all approaches. Build a comparison table. Write `reports/results-YYYY-MM-DD.md` summarizing the arc, including which approach won and how close it came on each model.

## Decision tree summary

```
Phase 1 (Approach A)
├── ≥ 99.9% on 0.5B  → Phase 4 (validate end-to-end) → Phase 5 (8B) → done
├── 98-99.9%        → Phase 2 (Approach B) → re-eval
│                       ├── ≥ 99.9%   → Phase 4 → Phase 5 → done
│                       └── < 99.9%   → Phase 2 with Approach C
├── 90-98%          → Phase 2 + Phase 3 in parallel
└── < 90%           → Phase 3 first (better approximations), then re-try Phase 1
```

## Implementation details by op

### IntRMSNorm
```python
class IntRMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6, lut_seed_bits=10):
        self.gamma = nn.Parameter(torch.ones(hidden_dim))     # trainable
        # invsqrt LUT: 1024-entry table indexed by top 10 bits of mantissa
        self.register_buffer("invsqrt_lut", ...)
        self.eps = eps

    def forward(self, x):
        # x: [..., hidden]
        x_int, x_scale = quantize_per_token(x, bits=24)
        s_int = (x_int.int64() ** 2).sum(-1, keepdim=True)    # int64
        s_dequant = s_int.float() * x_scale**2 / hidden_dim + eps
        # Newton-Raphson invsqrt:
        r0 = self.invsqrt_lut[bit_extract(s_dequant)]          # lookup seed
        r1 = r0 * (1.5 - 0.5 * s_dequant * r0**2)              # iterate
        r2 = r1 * (1.5 - 0.5 * s_dequant * r1**2)
        # Apply gamma and inv-sqrt:
        out_int = round(x_int * gamma_int * r2 * combined_scale)
        return out_int * combined_scale
```

### int_softmax
```python
def int_softmax(x):
    # x: [..., n], int
    m_int = x.amax(-1, keepdim=True)
    shifted = x - m_int                                       # all ≤ 0
    # 1024-entry exp LUT, range [-100, 0] mapped to 0..1023:
    idx = clamp((shifted / shift_step).to(int), 0, 1023)
    e_int = exp_lut[idx]                                       # int
    s_int = e_int.sum(-1, keepdim=True)
    # NR reciprocal:
    r = newton_reciprocal(s_int, iterations=2)
    return e_int * r                                           # int probabilities
```

### int_silu
```python
def int_silu(x):
    # x: [..., n], int
    idx = clamp((x / sigmoid_step + 2048).to(int), 0, 4095)
    sig_int = sigmoid_lut[idx]
    return x * sig_int                                         # int multiply
```

### IntMatmul (for attention Q@K.T and P@V)
```python
def int_matmul(a, b):
    a_int, a_scale = quantize_per_token(a, bits=24)
    b_int, b_scale = quantize_per_token(b.transpose(-1,-2), bits=24)
    int_prod = (a_int.int64() @ b_int.int64().transpose(-1,-2))
    return int_prod.float() * a_scale * b_scale.transpose(-1,-2)
```

## Risk catalog

| risk | likelihood | mitigation |
|---|---|---|
| HF model surgery breaks per-architecture forward | high | unit-test each model's forward against the unmodified model; debug per arch |
| Softmax LUT too coarse → attention degrades | medium | start with 4096 entries; bump if Phase D shows it matters |
| RMSNorm Newton-Raphson diverges on outliers | low | clamp `r_n` between iterations; 3 iterations as fallback |
| γ + biases not enough DoF for Approach B | medium | Approach C unfreezes IntLinear shadows |
| Cumulative drift across 32 layers | medium | logit-L2 over all positions provides strong training signal |
| STE backward through lookup is too noisy | medium-low | larger batch + grad accumulation; or smooth-interpolated LUT |
| Per-channel γ can't compensate position-dependent error | medium | add learnable per-layer scalar offsets as fallback |
| 8B models hit a wall the 0.5B didn't | medium | the matmul work showed 8B was the hard case; train longer if needed |
| ZK circuit cost of int operations is prohibitive | unknown | out of scope for this experiment; this is for the engineering follow-up |

## Compute budget

Phased:
- Phase 0 (impl): $15
- Phase 1 (A on 0.5B): $2
- Phase 2 (B on 0.5B): $4
- Phase 3 (D + E parallel): $4
- Phase 4 (literal int 0.5B): $15
- Phase 5 (8B validation): $30
- Buffer: $10

**Total estimated: $80.**

Three boxes in parallel for ~24-48 h of total walltime. Each box budget cap: $30.

## Reproducibility

Same conventions as prior experiments:
- All JSONs in `experiments/train-nonmatmul-int/data/`.
- One ≤200-word log entry per session in `EXPERIMENT_LOG.md`.
- Code synced local ↔ remote via rsync.
- Each box's invocation script in `scripts/`.
- Final aggregate report in `reports/results-YYYY-MM-DD.md`.
- Code modules: `src/difr_expt/int_ops.py`, `int_ops_ste.py`, `patch_hf_model.py`, `train_nonmatmul.py`.

## What success looks like

A table:

| model | approach used | top-1 (all) | logit-L2 mean | margin p99 | end-to-end int-validated |
|---|---|---|---|---|---|

Plus a 1-paragraph explanation of which approximation choices and which training (if any) got us there, and any remaining ZK-engineering work (circuit compilation, replay-server protocol) that's still ahead.
