#!/bin/bash
# Phase 1: Approach A measurement on Qwen2.5-0.5B.
# Run inside the rented box from /root/difr-expt with venv activated.

set -e
cd /root/difr-expt
mkdir -p experiments/train-nonmatmul-int/data

# Sanity step: --n-prompts 1 to print replacement counts.
echo "=== sanity: 1-prompt run to verify patcher fires ==="
python -m difr_expt.run_baseline \
  --model Qwen/Qwen2.5-0.5B \
  --dtype float32 \
  --n-prompts 1 --max-len 128 \
  --weight-bits 24 --activation-bits 24 \
  --matmul-dtype auto \
  --int-nonmatmul \
  --out experiments/train-nonmatmul-int/data/qwen25_0p5b_PHASE1_sanity.json 2>&1 | tee /tmp/phase1_sanity.log

echo "=== full Phase 1 run: 100 prompts ==="
python -m difr_expt.run_baseline \
  --model Qwen/Qwen2.5-0.5B \
  --dtype float32 \
  --n-prompts 100 --max-len 512 \
  --weight-bits 24 --activation-bits 24 \
  --matmul-dtype auto \
  --int-nonmatmul \
  --out experiments/train-nonmatmul-int/data/qwen25_0p5b_PHASE1_A.json 2>&1 | tee /tmp/phase1_a.log
