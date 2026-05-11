# fp4/fp8 emulation: int proof model as student of a low-precision production model

## Strategic context

Prior experiments (`train-nonmatmul-int`) measured int24 vs fp32 reference and got top-1 ≈ 0.999 — basically indistinguishable. That tells us very little about DiFR's real value, because fp32 and int24 are both extremely high precision; the work of "emulating" the reference is trivial.

**The real production setting**: labs deploy LLMs at fp4 or fp8 for inference. That low-precision model is what users actually see and what a verifier must check. fp4/fp8 introduces significant noise — Gumbel-margin-style divergence vs a full-precision reference on the order of 70–90% of sampled tokens.

The DiFR pitch is: prove the int model executes correctly in ZK, and trust that the int model matches the production fp4/fp8 model **because we trained it to**.

## Question

How small can we make the divergence between an int24 student and an fp4/fp8 teacher, given that we can train the student? Concretely:

- **Top-1 match** between int24 student and fp4/fp8 teacher on held-out prompts
- **Gumbel margin** (the DiFR-relevant metric) between them
- **KL** and **logit-L2** for completeness

Compare to:

- **Untrained baseline**: int24 student vs fp4/fp8 teacher with default initialization (matmul weights = teacher's fp32 weights, default LUTs)
- **Untrained int vs fp32**: how small the divergence is when the teacher is the high-precision reference

## Approach

1. **fp4/fp8 quantizer (`src/difr_expt/fp_quant.py`)** — a small library that fake-quantizes any nn.Linear to:
   - **fp8 e4m3**: round each weight (and activation) to the nearest fp8 representable value using torch's native `torch.float8_e4m3fn` dtype. Block-wise absmax scaling per row of 128.
   - **fp4 (MXFP4 / E2M1)**: block-wise quant to the 16 E2M1 representable values `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` with one fp32 absmax scale per block of 32.

   Activations get the same fake-quant treatment per-token. Both ops are STE-friendly (rounding error feeds back as identity).

2. **Production-model patch (`patch_model_low_precision`)** — walk model, replace `nn.Linear` with `LowPrecisionLinear` that fake-quantizes weight + activation each forward. Skip lm_head by default (a common production convention).

3. **Student model** — the existing int24 model from `patch_model_int_cast` + `patch_model_int_nonmatmul`, with trainable matmul shadows + γ + biases + LUT entries. No changes to the int infrastructure; we just retarget the loss.

4. **Training loop (`src/difr_expt/train_emulate.py`)** — fork of `train_nonmatmul.py`:
   - Teacher = fp4/fp8 quantized HF model (frozen)
   - Student = int24 model (trainable params as above)
   - Loss = `KL(softmax(teacher_logits/T) || softmax(student_logits/T)) + aux * MSE(teacher_logits, student_logits)`
   - Also logs the auxiliary "untrained int vs fp32" divergence at eval so we can see how much harder fp4 emulation is than fp32 matching.

5. **Eval harness (`src/difr_expt/eval_emulate.py`)** — runs three pairings on a held-out prompt set:
   - `int_student` vs `fp4/fp8_teacher`  (the headline)
   - `int_student` vs `fp32_ref`         (sanity)
   - `fp4/fp8_teacher` vs `fp32_ref`     (how much noise fp4/fp8 itself injects)

## Smoke test (Phase 0)

- Local CPU, Qwen2.5-0.5B (already cached at `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B`).
  We use Qwen2.5-0.5B and not `sshleifer/tiny-gpt2` because the int patches target Qwen/Llama/Mistral RMSNorm-architectures only — GPT-2 would silently skip every int op.
- 2 prompts × 32 tokens, 20 training steps. Confirm:
  - Untrained int student diverges noticeably from fp4 teacher (>0% of positions mismatched)
  - One training step improves at least one metric (loss goes down)
  - All three eval pairings produce non-NaN scalars

## Full run (Phase 1)

- Qwen2.5-0.5B on CPU still works; if it's too slow, use the remote GPU box.
- 100 prompts (eval), ~256 tokens each. ~500–1000 training steps. lr ≈ 1e-5 for matmul shadows + 1e-3 for γ/biases/LUTs.
- Two configurations: fp8 e4m3 teacher and fp4 (E2M1 block) teacher.

## Reporting (Phase 4)

Write `reports/results-2026-05-11.md` with one before/after table per teacher precision. Include columns: top-1, top-5, KL p99, Gumbel margin p99, logit-L2 p99, and the int-vs-fp32 sanity row.

## Out of scope

- Activation-DiFR (separate metric).
- Building the actual ZK circuit.
- Larger models (Qwen3-8B, Llama-3.1-8B) — once the recipe works on 0.5B we'd port the winning approach.

## Files touched

- `src/difr_expt/fp_quant.py` — new
- `src/difr_expt/train_emulate.py` — new (fork of train_nonmatmul.py)
- `src/difr_expt/eval_emulate.py` — new
- `experiments/fp4-emulation/scripts/run_smoke.sh` — smoke harness
- `experiments/fp4-emulation/scripts/run_phase1.sh` — main run harness
- `experiments/fp4-emulation/reports/results-2026-05-11.md` — write-up
