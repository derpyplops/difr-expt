# Plan: execute the "int proof model emulates fp production model" experiment

Status: draft 2026-05-12. Executes the experiment described in the proposal
doc (`<doc>` in the conversation). Supersedes the broader 5-condition
sweep in `fp-emulation-published-teachers.md` — that plan answered a
slightly different question (cross-model generalization with logit-only
loss); this plan answers the doc's headline question directly.

## What changes from the in-flight setup

- **Drop the 5-condition cross**. Doc says "use one small-to-medium model
  where iteration time is low." We focus on **Qwen2.5-0.5B + published fp8
  teacher** (RedHatAI/Qwen2.5-0.5B-FP8-dynamic). All prompts already cached,
  ~10 min per run, low-risk.
- **Add per-matmul loss term**. Doc's main loss is per-matmul, not just
  logits. We currently train on logits only. ~50 LOC change to add hooks.
- **Add per-matmul measurement to eval**. Currently not measured at all.
  Required for the doc's "per-matmul L1/L2 over training" plots.
- **Switch init to cast-from-teacher**. Resolves the ambiguity in the doc's
  "Naive int32" row — `Mᵢ_int_naive(x)` is what you get when student weights
  *are* the teacher's quantized weights cast to int24. Cleaner experimental
  story: cast handles weight-side noise for free; training handles
  activation-side noise.
- **Terminate the 8B runs** in flight. Not needed for this experiment.
  Save the vast.ai instance (already up); reuse for the smaller sweep.

## Decisions on ambiguous points from the doc

| Doc says | Our concrete choice | Reason |
|---|---|---|
| int32 | **int24** | int24×int24 = int48 fits in int64 for matmul accumulation; matches our existing IntLinear; equivalent for emulation (both vastly exceed fp32 precision) |
| "Naive int32 baseline" | **Cast teacher's fp8 weights → int24** | Otherwise the baseline measures fp32-vs-fp8 gap, not what a proof-model deployment would actually start with |
| "Train weights only" | **`weight_fp` shadows only**, no γ / bias / LUT updates | Matches doc literally |
| "Train weights + scales" | **`weight_fp` + per-row scales as a separate fp32 param** | Doc lists this as a variant; currently scale is derived from absmax. ~10 LOC to make scales their own parameter group |
| "Per-matmul loss" | **L2 per linear, sum across all matmuls** | Doc says L1 or L2; L2 has smoother gradients |
| lm_head | **Skip** (matches teacher convention) | Already settled — keep `--no-int-lm-head` default |
| Embedding | **Keep bf16** | Lookup is outside the int circuit in ZK-ML proofs; out of scope for this experiment |
| Activation quantization | **Same as today: per-token absmax to int24** | Required for the int student to be ZK-proveable; documented in the report |
| LUTs (softmax, SiLU) | **Variant 6: include or exclude** | Adds a "non-matmul ops trainable" variant beyond the doc's table |

## The 6 variants to run

The doc proposes 5 variants. We run those plus one extra:

| # | Variant | Init | Trainable | Loss | Notes |
|---|---|---|---|---|---|
| 1 | **Naive** | teacher-cast | none | n/a | Step-0 eval only; no training |
| 2 | **Weights only** | teacher-cast | `weight_fp` | logit (KL + L2) | Doc's "Train weights only" |
| 3 | **Weights + scales** | teacher-cast | `weight_fp` + per-row scale | logit | Doc's "Train weights + scales" |
| 4 | **Per-matmul loss only** | teacher-cast | `weight_fp` | Σ per-matmul L2, no logit term | Doc's "Per-matmul loss only" |
| 5 | **Per-matmul + logits** | teacher-cast | `weight_fp` | Σ per-matmul L2 + logit (KL + L2) | Doc's "Per-matmul + logits loss" |
| 6 | **+ trainable non-matmul ops** | teacher-cast | `weight_fp` + γ + bias + LUTs | Same as #5 | Tests whether γ/LUT training adds anything on top of the per-matmul recipe |

Variant 1 is free (no training, just eval). Variants 2-6 are 500 training steps each, ~10 min on H100. Total: ~50 min for the full sweep + ~5 min for variant 1.

## Code changes needed

### 1. Cast-from-teacher init in `build_models()` (~30 LOC)

When `--init-from-teacher` is set:
- After loading teacher and base, walk the teacher's `Linear` modules and extract their effective weights (dequantize from the published fp8 storage on the fly).
- Use those as the `weight_fp` init for the corresponding student IntLinear (instead of the fp32 base weights).
- Verify shape/name match before the swap.

### 2. Per-matmul loss term in `train_emulate.py` (~50 LOC)

- Register forward hooks on every patched `IntLinear` in the student and the
  corresponding `nn.Linear` / `LowPrecisionLinear` in the teacher.
- Hooks capture (output) per layer; pair them by module-path name.
- Each train step: forward both, get per-layer output pairs, compute Σᵢ L2.
- Add to total loss with a configurable weight `--matmul-loss-weight`
  (0.0 = logit-only, 1.0 = balanced with logit term, >1.0 = per-matmul dominant).

### 3. Per-matmul L2 / L1 reporting in eval (~30 LOC)

- Same hooks as #2, used in inference mode during eval.
- Compute per-layer L2 / L1 over a fixed prompt batch.
- Log to `metrics.jsonl` under `matmul/<layer_name>/l2` keys.

### 4. CLI flags

```
--init-from-teacher           # cast teacher weights → student weight_fp
--matmul-loss-weight FLOAT    # 0.0 disables, default 1.0
--matmul-loss-norm {l1,l2}    # default l2
--logit-loss-weight FLOAT     # default 1.0; set 0 for variant 4
--trainable-scales            # promote per-row scale to its own param
                              # (currently derived); for variant 3
--no-trainable-gamma-bias     # for variants 2-5
```

Existing `--no-trainable-luts` already handles the LUT toggle.

## Metrics to log per eval step

Existing (keep):
- `student_vs_teacher/{top1, top5, kl_p99, logit_l2_p99, margin_p99}`
- `loss` (decomposed: `loss/logit_kl`, `loss/logit_mse`, `loss/matmul_l2`)

New:
- `matmul/<layer>/l2` — per-layer L2 between student and teacher matmul output
- `matmul/<layer>/l1` — per-layer L1
- `matmul/aggregate/{mean_l2, max_l2, sum_l2}` — across all layers
- Optionally `matmul/per_block/{embed, attn_q, attn_k, ..., mlp_down}` —
  grouped by role within a transformer block

## Eval set

Reuse the 100 wikitext-103 prompts already cached at
`experiments/fp-emulation/data/prompts_qwen25.pt` (80 train / 20 eval split,
matching existing convention).

## Plots / deliverable (matches the doc's "Deliverable" section)

After all 6 variants complete, write `experiments/int-emulates-fp/reports/results-2026-05-12.md`:

1. **Description** of float teacher (RedHatAI Qwen2.5-0.5B-FP8-dynamic) and
   int24 student (matmul + per-token act quant + RMSNorm/softmax/SiLU LUTs).
2. **Exact int24 / fixed-point scheme**: per-row absmax weight scale,
   per-token absmax activation scale, integer accumulation in fp32 (matches
   production GEMM kernel convention).
3. **Logit L1/L2 over training** (line plot, all 6 variants overlaid).
4. **Per-matmul L1/L2 over training** (heatmap over layers, one panel per
   variant; or one line per layer-group).
5. **Naive vs trained**: bar chart comparing variant 1's pre-train numbers
   against variant 2-6's post-train numbers.
6. **Headline readout** (1-2 paragraphs): does training materially close
   the gap? Where does the residual divergence concentrate?

## Phases

| Phase | What | Time |
|---|---|---|
| 0 | Terminate 8B sweep, save GPU instance | 1 min |
| 1 | Implement cast-from-teacher init | ~30 min |
| 2 | Implement per-matmul loss + measurement | ~45 min |
| 3 | Implement trainable-scales flag (variant 3) | ~15 min |
| 4 | Smoke-test variant 1 (Naive) — should match doc's expected baseline | 5 min |
| 5 | Run variants 2-6 sequentially | ~50 min |
| 6 | Pull results, generate plots, write report | ~1 hr |

Total: ~3 hr clock time. Total GPU spend: ~$3.

## Negative-result paths the doc anticipates

If we get the doc's negative result, the residual divergence is likely to
land in one of these buckets:

- **fp8 activation quant in deep layers**: noise compounds across 24 layers;
  the int student can't easily reproduce the *specific* per-token rounding
  pattern from fp8 activations because its activation quant is structurally
  different (int24 has finer granularity).
- **Attention softmax**: small differences in attention probability
  distributions can swing top-1 even when other metrics look good.
- **lm_head**: kept fp32 in both teacher and student so this *shouldn't* be
  a contributor; if it is, our skip-lm_head choice was wrong.

The per-matmul L2 trace tells us which.

## Out of scope (deliberately)

- 8B / Llama runs — separate plan, not what this experiment is asking.
- NVFP4 / fp4 teachers — same.
- Actual ZK circuit construction.
- Embedding quantization — outside the int circuit per ZK-ML conventions.
- Hyper-parameter sweep — single LR/warmup pair (1e-5/20), same as Run 1
  recipe which we already validated stable on Qwen2.5-0.5B.

## Files touched

- `src/difr_expt/train_emulate.py` — add hooks, init flag, loss decomposition
- `experiments/int-emulates-fp/plan.md` — operational (created in Phase 1)
- `experiments/int-emulates-fp/EXPERIMENT_LOG.md` — append-only log
- `experiments/int-emulates-fp/scripts/run_variants.sh` — orchestrator
- `experiments/int-emulates-fp/reports/results-2026-05-12.md` — final write-up
- `experiments/int-emulates-fp/figures/` — generated plots

## Open questions for sign-off

1. **Confirm: Qwen2.5-0.5B is the right model** vs. e.g. Qwen3-1.8B if we want
   "small-to-medium." 0.5B is fastest; 1.8B is closer to "production scale."
2. **Confirm: int24 vs int32.** The doc says int32; we use int24. Equivalent
   for emulation but the report should explain why we picked one.
3. **Confirm: terminate the in-flight 8B runs.** Run 2 (Qwen3-8B fp8 with the
   working low-LR recipe) is mid-training, ~15 min from completing. Could
   let it finish and have the result as a side-experiment, or kill now to
   save $1 of GPU.
