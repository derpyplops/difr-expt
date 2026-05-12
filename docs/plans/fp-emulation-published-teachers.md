# Plan: int24 student emulates published fp8/fp4 teachers (3 models, 5 runs)

Status: draft 2026-05-12. Supersedes the single-model fp4-emulation plan
(`experiments/fp4-emulation/plan.md`) which used an in-house fake-quant teacher
on Qwen2.5-0.5B and OOM'd before training started.

## Strategic shift

Prior `train-nonmatmul-int` work showed int24 vs fp32 reference is uninteresting
(top-1 ≈ 0.999 untrained — both formats have enormous precision headroom). The
fp4-emulation experiment swapped the teacher to a fake-quantized fp4 model to
introduce real noise, but the teacher was still our own construction, leaving
the obvious objection: *"is that what labs actually deploy?"*

The cleanest answer is to use **published, publicly released fp8 and fp4
checkpoints** as the teacher. Every measurement then says "the int24 student
can emulate Qwen3-8B-FP8 *as deployed*" rather than "as we approximated it."

This also retires the MXFP4-vs-NVFP4 ambiguity in our own `fp_quant.py`: the
published checkpoints settle the format question by being what they are.

## Scope: 5 training runs

| Model | fp8 teacher | fp4 teacher |
|---|---|---|
| Qwen2.5-0.5B | `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` | — (no published fp4 variant at this size) |
| Qwen3-8B | `Qwen/Qwen3-8B-FP8` (official) | `nvidia/Qwen3-8B-NVFP4` |
| Llama-3.1-8B-Instruct | `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` | `nvidia/Llama-3.1-8B-Instruct-NVFP4` |

Student in all 5 runs: the same int24 patching stack (`patch_model_int_cast` +
`patch_model_int_nonmatmul`, optionally `patch_model_int_embedding`). All three
base architectures are RMSNorm (Qwen2.5 / Qwen3 / Llama-3.1), so the int
patches apply cleanly.

### Teacher choice notes

- **fp8**: per-row dynamic e4m3 is the de facto deployed format. Qwen ships
  their own official `-FP8` checkpoints for the 8B model; for the 0.5B we use
  RedHat's because Qwen didn't publish one at that size. All three teachers
  are W8A8 dynamic.
- **fp4**: we standardize on **NVFP4** (NVIDIA's format: block-16 with an e4m3
  scale per block + a per-tensor fp32 scale on top), not MXFP4. NVIDIA ships
  NVFP4 checkpoints for both 8B models; their NVFP4 is the format actually
  deployed on Blackwell tensor cores. MXFP4 variants exist (ISTA-DASLab) but
  are kept as a possible ablation, not a primary axis.
- **No fp4 row for 0.5B**: nobody has published a fp4 Qwen2.5-0.5B (too small
  for fp4 memory savings to matter). Either we accept this and report fp4
  only for the 8B pair, or we fall back to our own fake-quant for 0.5B,
  breaking the "deployed checkpoints" methodology. **Recommendation**: leave
  0.5B out of fp4. If someone asks "what about 0.5B fp4?", we have our prior
  fake-quant numbers ready.

## Code changes

### New: published-checkpoint teacher path in `train_emulate.py`

Currently `build_models()` always calls `patch_model_low_precision()` on a copy
of the fp32 base. Add a `--teacher-source published|fake_quant` flag:

- `published`: load the teacher via `AutoModelForCausalLM.from_pretrained(
  teacher_id, quantization_config=CompressedTensorsConfig(...))` for fp8, and
  the `fp_quant` HF integration for NVFP4. Teacher returns fp32-cast logits
  via the library's built-in dequant during forward.
- `fake_quant`: current behavior (kept for ablation and as a fallback if a
  published checkpoint has loader issues).

Required dependencies: `compressed-tensors` (for fp8) and `fp_quant` (for
NVFP4) — both pip-installable. Verify against torch version in `pyproject.toml`.

### Add `--no-fp32-ref` plumbing if not already wired

The argparse flag exists but verify `build_models(keep_fp32_ref=False)`
actually frees the base. Needed for 8B runs to fit in 80 GiB VRAM.

### Optional: NVFP4 in our own `fp_quant.py`

For the fake-quant ablation we'd want NVFP4 support too. ~30 LOC:
`fake_quantize_nvfp4_blocksym(x, block_size=16, scale_dtype=torch.float8_e4m3fn)`,
plus per-tensor fp32 scale. Lower priority — only build if the published-loader
path turns out to be flaky.

## Compute & memory analysis

### Qwen2.5-0.5B

Comfortably fits anywhere. CPU works (~3 GiB), H100 turns 500 steps into ~25
minutes. Use H100 for consistency with the 8B runs and to validate the
multi-model harness on a small target first.

### 8B models (Qwen3-8B, Llama-3.1-8B-Instruct)

Naive memory budget at fp32:
- Teacher (fp8 or NVFP4, dequant on forward): ~8 GiB weights + activations
- Student int32 buffers (8B params): 32 GiB
- Student fp32 weight shadows (trainable): 32 GiB
- AdamW state on shadows (m + v): 64 GiB
- fp32 reference: 32 GiB
- **Total: ~170 GiB** — overflows H100 80 GiB by 2×.

Trim plan to fit on a single H100 80GB:
1. `--no-fp32-ref` → save 32 GiB → 138 GiB
2. bf16 student shadows → save 16 GiB → 122 GiB
3. 8-bit AdamW (bitsandbytes) → save ~48 GiB → 74 GiB
4. Gradient checkpointing → reduces activation peak, buys headroom
5. **Total: ~74 GiB**, fits

If still tight, fallback options in order of preference:
- LoRA adapters on matmul shadows (drop trainable params 100×; preserves
  qualitative "weight-side correction" capability)
- Two-H100 setup (data parallel or just bigger box)
- Drop matmul shadows entirely and train only γ+bias+LUTs (changes the
  experiment — reserve as ablation only)

### GPU schedule estimate

| Run | Steps | Est. time on H100 |
|---|---|---|
| Qwen2.5-0.5B fp8 | 500 | ~25 min |
| Qwen3-8B fp8 | 500 | ~4 hr |
| Qwen3-8B NVFP4 | 500 | ~4 hr |
| Llama-3.1-8B-Instruct fp8 | 500 | ~4 hr |
| Llama-3.1-8B-Instruct NVFP4 | 500 | ~4 hr |
| **Total** | | **~17 hr** |

Plus pre-flight (smoke, baseline-only eval pass for all 5): ~1 hr. Plus the
inevitable debug cycles on the first 8B run: budget 2 GPU-days total.

Use vast.ai per existing memory; one H100 SXM offer at ~$3/hr ≈ $150 total.

## Phases

### Phase 0 — Smoke (Qwen2.5-0.5B fp8 only)

- Goal: verify the new `published`-teacher loader works end-to-end.
- 5 steps, batch 1, 2 prompts. Confirm:
  - Teacher loads without dtype/architecture errors
  - Student forward emits logits of matching vocab size
  - Loss is finite and decreasing
  - All 3 eval pairings (student_vs_teacher, student_vs_ref, teacher_vs_ref)
    produce non-NaN scalars
- Output: `experiments/fp-emulation/data/smoke_qwen05_fp8/`

### Phase 1 — Untrained baselines (all 5 conditions, zero training)

- For each (model, precision) pair, run a single eval pass at step 0.
- Records the "untrained" row of the final table — the pre-training Gumbel
  margin between int24 student and the published fp8/fp4 teacher.
- Cheap (~5–10 min per condition on H100, ~30 min total).
- Independent of training; informative on its own.
- Output: `experiments/fp-emulation/data/baseline_<model>_<precision>/`

This phase has standalone value: it answers "how big is the gap before any
training?" — which is the comparator the trained student has to beat.

### Phase 2 — Training runs (5 conditions, sequential)

Per-run config:
- 500 steps (revisit after first run if loss is still falling steeply)
- Batch 2, prompts ~256 tokens
- LRs: 1e-5 (matmul shadows), 1e-3 (LUTs), 1e-4 (γ+bias) — same as
  fp4-emulation plan
- Cosine warmup 20, plateau patience 5
- 8-bit AdamW for 8B runs; standard AdamW for 0.5B
- `--no-fp32-ref` for 8B runs; keep fp32 ref for 0.5B (cheap, useful sanity row)
- Save `best.pt` on `student_vs_teacher/top1`

Order them smallest-first so any harness issue surfaces on a cheap run:
1. Qwen2.5-0.5B fp8
2. Qwen3-8B fp8
3. Qwen3-8B NVFP4
4. Llama-3.1-8B-Instruct fp8
5. Llama-3.1-8B-Instruct NVFP4

Output: `experiments/fp-emulation/data/<model>_<precision>/`

### Phase 3 — Final eval & report

For each run: reload `best.pt`, run on the 100-prompt held-out set, log
pre/post for all three pairings.

Write `experiments/fp-emulation/reports/results-2026-05-1?.md` with one
table (the headline):

| Model | Teacher | top-1 pre/post | top-5 pre/post | KL p99 pre/post | Gumbel margin p99 pre/post | logit-L2 p99 pre/post | teacher_vs_ref noise floor |

Plus a one-paragraph takeaway per row and an overall summary: does training
narrow the int24-vs-fp4/fp8 gap to within `1 - teacher_vs_ref/top1`? (i.e.,
does the student get as close to the teacher as the teacher gets to fp32?)

## Sanity checks before kicking off

1. **Vocab parity**: verify all three teachers share vocab with their fp32 base.
   Compressed-tensors loaders preserve the tokenizer config but worth
   checking — vocab mismatch silently truncates logits in `evaluate()`.
2. **Architecture parity**: confirm `patch_model_int_nonmatmul` returns
   non-zero counts for each of the three bases. The fp4-emulation smoke on
   tiny-gpt2 silently no-op'd — don't repeat that.
3. **Teacher determinism**: re-run the teacher on the same prompt twice;
   logits should be bit-identical. CompressedTensorsConfig's per-token
   dynamic act quant should still be deterministic given fixed inputs, but
   verify before relying on it as a stable training target.
4. **Fake-quant vs published sanity**: on the 0.5B model, compare
   `fake_quantize_fp8` logits against the RedHatAI checkpoint's logits on the
   same 20 prompts. If `logit_l2 < 0.01` and `top-1 ≈ 1.0`, our fake-quant
   was a faithful stand-in (relevant for future fp4 work on sizes where no
   published checkpoint exists). If they diverge, we now know to trust the
   published version.

## Open questions for owner sign-off

1. **5 runs, published teachers** — confirm scope. Alternative: 3 runs
   (fp8 only) first, fp4 in a follow-up wave.
2. **Prompt source**: reuse the 8 cached `qwen_prompts.pt`, or grab ~100
   from wikitext-103 / c4 / OpenWebMath?
   - Recommendation: wikitext-103 val split, 100 prompts, ~256 tok each.
3. **Same 500-step budget across all models** vs scale with size?
   - Recommendation: same 500 everywhere, simplifies comparison.
4. **Single output directory** (`experiments/fp-emulation/`) or one per
   precision (`experiments/fp8-emulation/`, `experiments/fp4-emulation/`)?
   - Recommendation: single dir, subfolders per `(model, precision)`. The
   existing `experiments/fp4-emulation/` from the prior aborted run can be
   archived (preserves logs without polluting the new tree).

## Out of scope

- MXFP4 teachers (kept available as ablation if NVFP4 results raise questions)
- Activation-DiFR or any non-token-distribution metric
- Building actual ZK circuits
- Model sizes beyond 8B
- Quantization-aware training of the teacher (we use deployed checkpoints
  as-is; the ISTA-DASLab QAT-NVFP4 sweep is interesting comparison data but
  not within this plan's training budget)

## Files

- `docs/plans/fp-emulation-published-teachers.md` — this doc
- `src/difr_expt/train_emulate.py` — add `--teacher-source` flag and
  published-checkpoint loader (~50 LOC)
- `src/difr_expt/fp_quant.py` — unchanged unless we add NVFP4 fake-quant
- `experiments/fp-emulation/plan.md` — operational plan (commands, config
  files); created in Phase 0
- `experiments/fp-emulation/EXPERIMENT_LOG.md` — append-only log per project
  convention
- `experiments/fp-emulation/scripts/` — per-run shell harnesses
- `experiments/fp-emulation/data/<model>_<precision>/` — outputs
- `experiments/fp-emulation/reports/results-2026-05-1?.md` — final write-up
