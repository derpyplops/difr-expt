# layer-harness: per-named-module L2 error of the integerized model

## Why

Luke (Slack, 2026-05-14): "we want to see the error of the integerized
version of every layer ... not just the matmuls for evaluation of the
loss". Ultimate target is logits, but the harness reports per-layer error
so we can localize where the int approximations cost us accuracy.

Prior work in this repo measures end-to-end logit divergence
(`run_baseline.py`) and per-matmul residuals (`int-residual`). Neither
gives a layer-by-layer view of where the int-cast error originates and
how it compounds through the network.

**Scope (2026-05-15 onward): real hardware FP8 only.** The fake-quant
emulation modes were removed after we confirmed that emulation
substantially underestimates the deployed quantization error
(emulated 16-bit-int: top-1 = 1.0; real FP8: top-1 ≈ 0.91 on the
same checkpoint family). The harness now requires a Hopper-class
GPU (SM_89+) and refuses to run otherwise — `ScaledMmProbe` aborts
the run if `torch._scaled_mm` is never called during the first
forward.

## What the harness does

For one prompt:

1. Load the model twice: `M_float` (untouched bf16 teacher) and
   `M_int` (a real-FP8 student loaded from a pre-quantized checkpoint
   via HF + compressed_tensors, with the lazy-decompress hook
   stripped and every FP8 nn.Linear swapped to an `FP8Linear` that
   dispatches through `torch._scaled_mm`).
2. Register kwarg-aware forward hooks on **every named submodule** of
   both models. Hooks capture `(args, kwargs, output)` keyed by qualified
   name.
3. Run a forward pass through each model on the same input ids.
4. For each named module `m` that both models share:
   - **propagated** L2: `||out_int[m] − out_float[m]||₂` along the
     last (feature) dim. This is the deployed error and includes
     compounding from upstream int approximations.
   - **isolated** L2: feed `M_float`'s captured `(args, kwargs)` for
     `m` into `M_int`'s copy of `m`, diff against `M_float`'s captured
     output. This isolates `m`'s own contribution. Skipped (without
     erroring the run) when the call raises — typically modules whose
     forward mutates external state (kv-cache update inside attention).
5. Final logits: `logit_l2`, `kl_div_ref_to_cand`, `top1_match`,
   `topk_overlap` — same metrics as `run_baseline.py`.

Aggregations per (block_idx, module_class, family) and per (mean, p50,
p99, worst) reported in the table. We tag each module with:
- `block` — transformer block index parsed from name (`layers.{i}.…`),
  or `-1` for `model.embed_tokens` / `model.norm` / `lm_head`.
- `family` — `q/k/v/o/gate/up/down` for the seven Linears in a block;
  `attn`, `mlp`, `rmsnorm`, `silu`, `softmax`, `rotary`, `block`,
  `embed`, `head`, `final_norm` for everything else.

## Output shape

`experiments/layer-harness/reports/results-<date>.md` —
- summary header (model, n_prompts, max_len, int cfg flags),
- top-line: aggregate logit L2 / KL / top-1 match,
- table: row per qualified module name, columns =
  {block, family, shape, prop_mean, prop_p99, prop_worst, iso_mean,
  iso_p99, iso_worst}.

Sister file `results-<date>.json` for downstream plotting.

## Scope of this experiment

- **In**: smoke run (2 prompts × 64 tok, CPU) → first real run (50
  prompts × 256 tok, GPU) on Qwen2.5-0.5B with the default
  `IntOpsConfig`. Full-int (matmul + nonmatmul). Side-by-side
  propagated + isolated.
- **Out** (later, only if useful):
  - sweeping `weight_bits` / `activation_bits` / softmax LUT size etc.,
  - other models (Qwen2.5-7B, Llama-3.1-8B),
  - figures (one PR review at a time — start with the table).

## Files

- `src/difr_expt/run_harness.py` — the CLI script (reusable; the
  baseline experiments will share it).
- `experiments/layer-harness/scripts/` — kept empty for now; if we add
  experiment-specific glue (sweep driver, plot script) it goes here.
- `experiments/layer-harness/reports/results-*.md` — results.

No changes to `int_cast.py` / `patch_hf_model.py` / `int_ops.py`; the
harness only consumes their public surfaces.
