#!/usr/bin/env bash
# Phase 0: 5-step smoke on Qwen2.5-0.5B published fp8 teacher.
# Validates the published-teacher loader path on the smallest model.
set -euo pipefail

cd "$(dirname "$0")/../../.."

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu}
PY=${PYTHON:-python3}

$PY -m difr_expt.train_emulate \
    --model Qwen/Qwen2.5-0.5B \
    --teacher-source published \
    --teacher-id RedHatAI/Qwen2.5-0.5B-FP8-dynamic \
    --prompts experiments/fp-emulation/data/prompts_qwen25.pt \
    --out experiments/fp-emulation/data/smoke_qwen25_fp8 \
    --steps 5 --batch 1 --eval-every 5 --warmup 2 \
    --eval-n-prompts 4 --eval-max-positions 256 \
    --plateau-patience 99 \
    --dtype bfloat16

echo "=== Smoke done; check summary.json ==="
cat experiments/fp-emulation/data/smoke_qwen25_fp8/summary.json
