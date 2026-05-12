# Experiment: int24 student emulates published fp8/fp4 teachers

Operational plan. Strategic plan: `docs/plans/fp-emulation-published-teachers.md`.

## Conditions

| Run | Base model | Teacher source | Notes |
|---|---|---|---|
| 1 | `Qwen/Qwen2.5-0.5B` | `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` | smallest, runs first |
| 2 | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B-FP8` (official) | needs 8-bit AdamW + grad ckpt |
| 3 | `Qwen/Qwen3-8B` | `nvidia/Qwen3-8B-NVFP4` | same |
| 4 | `meta-llama/Llama-3.1-8B-Instruct` | `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` | gated; needs HF token |
| 5 | `meta-llama/Llama-3.1-8B-Instruct` | `nvidia/Llama-3.1-8B-Instruct-NVFP4` | gated; needs HF token |

## GPU setup (vast.ai H100 80GB, one-shot)

Per `~/.claude/CLAUDE.md` vast.ai notes. After SSH:

```bash
# 1. Fix CUDA driver symlinks (per CLAUDE.md)
cd /usr/lib/x86_64-linux-gnu
for f in libcuda libcudadebugger libnvidia-ml libnvidia-nvvm libnvidia-ptxjitcompiler \
         libnvcuvid libnvidia-opencl libnvidia-cfg libnvidia-gpucomp \
         libnvidia-opticalflow libnvidia-sandboxutils; do
    src=$(ls ${f}.so.[0-9]*.[0-9]*.[0-9]* 2>/dev/null | head -1)
    rm -f "${f}.so.1" "${f}.so"
    [ -n "$src" ] && ln -sf "$src" "${f}.so.1" && ln -sf "$src" "${f}.so"
done
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

# 2. Install missing python deps (per user direction: pip on running container)
pip install --break-system-packages compressed-tensors fp_quant bitsandbytes==0.46.1

# 3. Verify CUDA
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expect: NVIDIA H100 ... (9, 0)

# 4. HF login (Llama-3.1-8B-Instruct is gated)
huggingface-cli login --token "$HF_TOKEN"
```

## Phase 0 — smoke (~5 min)

```bash
python3 -m difr_expt.train_emulate \
  --model Qwen/Qwen2.5-0.5B \
  --teacher-source published \
  --teacher-id RedHatAI/Qwen2.5-0.5B-FP8-dynamic \
  --prompts experiments/fp-emulation/data/prompts_qwen25.pt \
  --out experiments/fp-emulation/data/smoke_qwen25_fp8 \
  --steps 5 --batch 1 --eval-every 5 --warmup 2 \
  --eval-n-prompts 4 --eval-max-positions 256 \
  --dtype bfloat16
```

Verify: loss finite + decreasing; non-NaN scalars in `summary.json`; teacher
loaded without compressed-tensors errors.

## Phase 1 — untrained baselines (zero-train, all 5 conditions, ~30 min)

Run training script with `--steps 0` (or invoke an eval-only entry point).
For now use 1-step run with `--no-trainable-matmul-weights --no-trainable-luts`
to effectively do "untrained eval only" — the pre-train eval gets logged.

Actually simpler: just rely on `pre` row of each Phase-2 run's summary.json
(it's logged at step 0 before training).

## Phase 2 — training runs (sequential, ~17 hr)

Common args: `--steps 500 --batch 2 --eval-every 100 --warmup 20 --grad-clip 1.0
--temperature 1.0 --eval-n-prompts 20 --eval-max-positions 50000 --dtype bfloat16
--seed 42`.

### Run 1: Qwen2.5-0.5B fp8 (~25 min)

```bash
python3 -m difr_expt.train_emulate \
  --model Qwen/Qwen2.5-0.5B \
  --teacher-source published \
  --teacher-id RedHatAI/Qwen2.5-0.5B-FP8-dynamic \
  --prompts experiments/fp-emulation/data/prompts_qwen25.pt \
  --out experiments/fp-emulation/data/qwen25_fp8 \
  --steps 500 --batch 2 --eval-every 100 --warmup 20 \
  --dtype bfloat16
```

### Run 2: Qwen3-8B fp8 (~4 hr, needs 8bit AdamW + grad ckpt)

```bash
python3 -m difr_expt.train_emulate \
  --model Qwen/Qwen3-8B \
  --teacher-source published \
  --teacher-id Qwen/Qwen3-8B-FP8 \
  --prompts experiments/fp-emulation/data/prompts_qwen3.pt \
  --out experiments/fp-emulation/data/qwen3_8b_fp8 \
  --steps 500 --batch 2 --eval-every 100 --warmup 20 \
  --dtype bfloat16 --use-8bit-adamw --grad-checkpointing --no-fp32-ref
```

### Run 3: Qwen3-8B NVFP4 (~4 hr)

```bash
python3 -m difr_expt.train_emulate \
  --model Qwen/Qwen3-8B \
  --teacher-source published \
  --teacher-id nvidia/Qwen3-8B-NVFP4 \
  --prompts experiments/fp-emulation/data/prompts_qwen3.pt \
  --out experiments/fp-emulation/data/qwen3_8b_nvfp4 \
  --steps 500 --batch 2 --eval-every 100 --warmup 20 \
  --dtype bfloat16 --use-8bit-adamw --grad-checkpointing --no-fp32-ref
```

### Run 4: Llama-3.1-8B-Instruct fp8 (~4 hr)

```bash
python3 -m difr_expt.train_emulate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --teacher-source published \
  --teacher-id RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
  --prompts experiments/fp-emulation/data/prompts_llama31.pt \
  --out experiments/fp-emulation/data/llama31_8b_fp8 \
  --steps 500 --batch 2 --eval-every 100 --warmup 20 \
  --dtype bfloat16 --use-8bit-adamw --grad-checkpointing --no-fp32-ref
```

### Run 5: Llama-3.1-8B-Instruct NVFP4 (~4 hr)

```bash
python3 -m difr_expt.train_emulate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --teacher-source published \
  --teacher-id nvidia/Llama-3.1-8B-Instruct-NVFP4 \
  --prompts experiments/fp-emulation/data/prompts_llama31.pt \
  --out experiments/fp-emulation/data/llama31_8b_nvfp4 \
  --steps 500 --batch 2 --eval-every 100 --warmup 20 \
  --dtype bfloat16 --use-8bit-adamw --grad-checkpointing --no-fp32-ref
```

## Phase 3 — report

After all 5 runs complete:
1. scp `experiments/fp-emulation/data/` back to local
2. Write `reports/results-2026-05-12.md` with headline table:
   - Model | Teacher | top-1 pre/post | KL p99 pre/post | Gumbel margin p99 pre/post | logit-L2 p99 pre/post | teacher_vs_ref noise floor
3. Terminate vast.ai instance

## Prompts

3 separate prompt caches (different tokenizers):

```bash
python3 -m difr_expt.cache_prompts \
  --tokenizer Qwen/Qwen2.5-0.5B --n-prompts 100 --max-len 256 \
  --out experiments/fp-emulation/data/prompts_qwen25.pt

python3 -m difr_expt.cache_prompts \
  --tokenizer Qwen/Qwen3-8B --n-prompts 100 --max-len 256 \
  --out experiments/fp-emulation/data/prompts_qwen3.pt

python3 -m difr_expt.cache_prompts \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct --n-prompts 100 --max-len 256 \
  --out experiments/fp-emulation/data/prompts_llama31.pt
```
