# Experiment log: layer-harness
Started 2026-05-14. See experiments/layer-harness/plan.md.
- 2026-05-14: Scaffold + smoke — wrote `src/difr_expt/run_harness.py`
  (kwarg-aware forward hooks on every named submodule of a float teacher
  and a full-int student built via `patch_model_int_nonmatmul` +
  `patch_model_int_cast`). Per module: propagated L2 = ||int.out − float.out||
  along the last dim, isolated L2 = ||int_mod(float.args) − float.out||
  with kv-cache stripped from kwargs (so attention forwards don't mutate
  cache state during isolation). Final logit metrics: L2, KL, top-1,
  top-5. Smoke on Qwen2.5-0.5B (2 wikitext prompts × 48 tok, CPU,
  default IntOpsConfig: 16-bit weight/act, 24-bit rmsnorm/attn/rope,
  softmax LUT 1024, silu LUT 4096): logit_l2=2.22 mean, KL=3.1e-5, top-1
  and top-5 = 1.000. Caught a real bug on the first run — the iso call
  was re-firing the int model's forward hooks and overwriting
  `cap_int[name]` for every inner submodule it visited, silently turning
  subsequent prop_l2 readings into iso_l2 readings; visible because
  `self_attn.q_proj` had `prop == iso` exactly. Fix: remove all hooks
  before the per-module iso loop (commit-pending; one block in
  `run_one_prompt`). Output table written to
  `reports/smoke-2026-05-14.{md,json}` — 317 modules per prompt.
  **Surprise**: for deep blocks, propagated L2 of self_attn is much
  smaller than isolated (block 5: prop=0.005, iso=1.57; growing with
  depth). Triangle inequality says prop ≥ iso − ||input drift||, so
  this only works if int_attn(int_h) − float_attn(int_h) is itself
  small — i.e. the int approximations are well-behaved on inputs that
  have already been cast through the int pipeline, but produce bigger
  per-cell drift on the cleaner float inputs (plausible if int_matmul
  clamps to int range and the float-input regime trips that clamp).
  Worth a closer look on a real-size run.
  Next: real run (50 prompts × 256 tok) on a GPU box; this CPU session
  doesn't have CUDA.
- 2026-05-14: Coverage gap closed — softmax, Q@K.T, P@V now measured.
  v1 of the harness covered every `nn.Module` (RMSNorm/Linear/SiLU/the
  whole attention block) but missed the three int approximations that
  live inline inside the int patcher's attention closure
  (`_int_qk_matmul`, `_int_pv_matmul`, `_int_softmax`). The float model
  computes the same ops as inline `torch.matmul` / `F.softmax` — not
  modules — so the harness's by-name matching had nothing to diff
  against. Added `src/difr_expt/harness_attn_wrap.py`:
  `FloatMatmul` / `FloatSoftmax` wrap the inline calls as modules;
  `wrap_attention_forward_float` replaces every float attention's
  forward with a float-eager closure that routes the matmuls and the
  softmax through those modules — math bit-identical to HF's
  `eager_attention_forward` in transformers 4.57.3 (we verified the
  source). `rename_int_attn_submodules` then renames the int side's
  `_int_*` to `_*` so the harness picks them up automatically as
  common submodules (safe because the int patcher's new_forward
  captures the modules as closure locals, not by attribute lookup).
  `prepare_models_for_harness` glues it together: deepcopy float →
  build int, wrap float, patch int, rename. Force eager dispatch
  on both. Smoke re-run on Qwen2.5-0.5B (2 prompts × 48 tok) now
  shows 389 modules per prompt (was 317; +72 = 24 blocks × 3 new
  rows), logit_l2 essentially unchanged (2.617/1.816 — proving the
  float wrapper is numerically faithful). By-family rollup of the
  new rows: `qk_matmul` prop=0.30 / iso=2e-4 (intrinsic int_matmul
  error is small, propagated big because Q,K themselves have drifted
  upstream); `pv_matmul` prop=6e-3 / iso=7e-7 (smallest intrinsic);
  `softmax` prop=4e-4 / iso=5e-6 (LUT-1024 is fine). RoPE skipped:
  the int patcher attaches `IntRopeApply` as `_int_rope` but
  `new_forward` calls the float `apply_rotary_pos_emb` directly, so
  RoPE isn't actually integerized in this codebase — no per-op error
  to measure. Documented in the harness_attn_wrap docstring.
- 2026-05-15: Real FP8 path on H200 — the emulated harness measures
  fake-quant in fp32, not the deployed FP8 GEMM. Wired a real path:
  new `--student fp8-hw` mode that loads a pre-quantized checkpoint
  (`RedHatAI/Qwen2.5-0.5B-FP8-dynamic`) via HF + `compressed_tensors`
  on an H200 (SM_90, $4.13/hr on vast.ai). Added a `ScaledMmProbe`
  context manager that monkey-patches `torch._scaled_mm` and counts
  invocations during the first forward — refuses to proceed if the
  count is zero. **The probe immediately caught that the default HF
  integration is itself an emulation**: at first forward
  `compressed_tensors` runs a `decompress` hook that restores FP8 →
  bf16 in memory, then `F.linear` runs plain bf16 — zero
  `_scaled_mm` calls. The "FP8 model" is really a bf16 model whose
  weights were rounded through FP8 at load time. Fix: new
  `src/difr_expt/fp8_hw_linear.py` with `FP8Linear` (forward does
  per-token dynamic FP8 quant + `torch._scaled_mm(x_fp8, w_fp8.t(),
  scale_a, scale_b, out_dtype=bf16)`) and
  `replace_compressed_linears_with_fp8` that strips the decompress
  hook and swaps every FP8 nn.Linear for an FP8Linear before the
  hook can fire. Two real-FP8 GEMM gotchas, both fixed: (1)
  `_scaled_mm` requires column-major `b` (stride(0)==1); `.t()` gives
  exactly that, but `.contiguous()` after `.t()` forces row-major and
  the kernel rejects it. (2) `weight_scale` from the checkpoint is
  fp16; promoted to fp32 to give the fused multiply headroom. Probe
  on the working run: 840 `_scaled_mm` calls during prompt 0 (= 168
  Linears × ~5 incl. iso re-calls) ✓. **Results, 16 wikitext prompts
  × ~190 avg tokens, Qwen2.5-0.5B bf16 teacher vs FP8-dynamic
  student**: logit_l2 mean ≈ 103, KL ≈ 0.029, top-1 ≈ 0.908
  (range 0.83–0.95), top-5 ≈ 0.911. Vastly larger than the
  emulated path's 16-bit-int student (logit_l2≈2, top-1=1.0) — real
  FP8 (~4 effective mantissa bits) is genuinely much lossier than
  what the emulated harness was measuring, and the harness can now
  surface that gap. Per-op isolated L2: `qk_matmul / pv_matmul /
  softmax / silu` all = 0 (FP8-dynamic doesn't quantize these — both
  sides run identical `torch.matmul` / `F.softmax`); `q_proj / k_proj
  / v_proj / o_proj / gate / up / down` show structured FP8
  per-channel quantization error compounding through the residual
  stream. Reports: `reports/fp8-hw-2026-05-15.{md,json}` (4×128)
  and `reports/fp8-hw-n16-2026-05-15.{md,json}` (16×~190). H200
  destroyed after the run.
