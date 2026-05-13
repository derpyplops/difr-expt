# Experiment log: int-residual
Started 2026-05-13. See experiments/int-residual/plan.md.
- 2026-05-13: Scaffold + smoke + R11 result — built `capture_records.py` /
  `build_residuals.py` / `fit_residual_models.py` for fake-quant FP8E4M3
  on Qwen2.5-0.5B CPU. After R1–R9 cluster at abs_mean≈273 / bit-exact ≈ 0.1%,
  derived the exact algebra `r = X'·δ_W + W'·δ_X − δ_X·δ_W − round(Ỹ)` and
  implemented R10 (cheap per-row/col δ summaries) and R11 (per-cell K-sum
  mixed products). R10 doesn't help; **R11 (3 extra Freivalds-checkable
  K-sums) crushes residual to abs_mean=2.78, bit-exact=13.1%** on smoke;
  fitted coefs land at (0, 1, 1, −1) confirming algebra. Remaining floor
  is fp32 accumulator drift in captured Y. R11_fixed (no-fit unit-coef
  rule) is bit-identical to R11. See `ITERATION_LOG.md` for the iteration
  trail and `reports/proof_cost_analysis.md` for cost framing.
- 2026-05-13: R11a validation, hybrid, FP4, larger T — captured
  `r11a_val/` (4 prompts × blocks {0,11,23} × T=128) and ran proper
  held-out fit (train={0,1}, val={2,3}): **R11=2.64, R11a=3.55,
  R11_fixed=2.64** (algebra-exact, no fit needed). Discovered K-asymmetry:
  under R1 `down_proj` (K=4864) is worst, under R11 it's best, under R11a
  it's worst again — δ_X·δ_W is a K-sum whose variance scales with K,
  so dropping it disproportionately hurts wide-K layers. Added
  `model_R11_hybrid_byK` (R11 on `down_proj`, R11a elsewhere): abs_mean=3.34
  at ~2.14-matmul cost vs R11a's 2.0 and R11's 3.0. Tried FP4 with the
  existing int8 pipeline (which assumes per-token/per-row scales): R1=1109,
  R11=1073 — algebra fails, R11 coefs (1.005, 0.977, −1.14) show the
  first-order terms are still locally right but the dominant residual lives
  in block-scale-along-K variation (source #1 from the plan). FP4 needs a
  capture+build refactor to expose per-block scales; out of scope this
  session. Larger T capture (block-0 only, max_len=512, actual lengths
  166–229) gave R11=2.40 — the residual floor is intrinsic per-cell and
  doesn't grow with T. Files: `reports/r11a_val_fit.json`,
  `reports/r11a_val_hybrid_fit.json`, `reports/r11a_val_fp4_fit.json`,
  `reports/r11a_val_T512_fit.json`; proof-cost analysis updated with held-
  out numbers and the K-asymmetry observation.
