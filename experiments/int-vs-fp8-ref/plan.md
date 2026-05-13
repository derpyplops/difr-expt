# Int-cast student vs FP8 production teacher

## Goal
Measure top-1 + Gumbel margin of a true-int student (int weights + int
activations + int RMSNorm/softmax/SiLU/attention matmuls) against the
**actual FP8 production model**, not the fp32-dequant or bf16-dequant
reference used in prior experiments.

This is the comparison Luke's idea actually asks for. Prior int-cast
experiments (`baseline-int-cast`, `train-int-cast`, `train-nonmatmul-int`)
all evaluated against a deepcopy of the same fp32/bf16 base model with
nn.Linear unmodified — never against the production fp8 model with
FP8Linear/CompressedLinear modules.

## Pass criteria
1. all-positions top-1 ≥ 0.99 vs fp8 production teacher
2. Gumbel margin p99 ≤ 1e-2
3. KL p99 ≤ 1e-3

Stretch: 99.9% top-1.

## Approaches (parallel)

### A. No-train cast (Luke's exact idea — cheapest)
- Teacher: published fp8 (`RedHatAI/Qwen2.5-0.5B-FP8-dynamic` / Llama
  equivalent / `Qwen/Qwen3-8B-FP8`).
- Student: bf16 base, dequantized fp8 weights copied in
  (`init_from_teacher=True` from build_models), then patched with
  IntLinear (w24, a24) + int non-matmul + IntEmbedding. **No training.**
- Question: does the int cast preserve top-1 vs the fp8 teacher?

### B. Approach C training (if A's gap > 0.5%)
- Same setup as A, plus train matmul weight shadows + γ + biases +
  LUT entries against the fp8 teacher's logits (logit-L2 loss).
- lr=1e-7, 2k steps, plateau-stop, eval-every-250.
- Smallest model first (0.5B) for cheap iteration.

### C. Luke's "fp32 train then cast" variant
- Train an fp32 student (no fake quant) on the fp8 teacher's logits.
- At end, cast to int24 and re-eval.
- Different from B: B uses STE fake-quant during training (so student
  learns to be robust to int noise); C trains in fp32 and accepts cast
  noise at the end. Tests whether the cast is "lossy enough to matter."

## Models
1. Qwen2.5-0.5B — `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` (per-row fp8)
2. Llama-3.1-8B-Instruct — `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`
3. Qwen3-8B — `Qwen/Qwen3-8B-FP8` (block fp8, harder — leave for last)

## Eval set
Same 100 wikitext-103-raw-v1 prompts × 512 tokens as prior baselines, so
results are directly comparable to the no-train fp32-ref numbers
(0.99928 / 0.99889 / 0.99826).

## Compute
Lambda H100 SXM5 (vast out of credit). ~$2/hr. Budget: ≤$10 for A,
≤$30 if B/C kick in.

## Code surface
- Reuse `build_models()` from `src/difr_expt/train_emulate.py` for
  teacher/student construction with `init_from_teacher=True`,
  `int_nonmatmul_bitexact=False`, `teacher_source="published"`.
- Reuse `evaluate()` from same file for headline metrics.
- New driver: `scripts/run_int_vs_fp8.py` — thin wrapper, no new logic.
