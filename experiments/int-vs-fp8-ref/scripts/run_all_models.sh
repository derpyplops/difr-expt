#!/bin/bash
# Run the int-cast vs fp8 production teacher eval on all three models.
# Sequential on one H100 (each model uses ~60-80GB peak).
#
# Usage: bash run_all_models.sh [out_dir]
#   defaults: out_dir = experiments/int-vs-fp8-ref/data
set -euo pipefail

OUT_DIR="${1:-experiments/int-vs-fp8-ref/data}"
mkdir -p "$OUT_DIR"

PY="python -u experiments/int-vs-fp8-ref/scripts/run_int_vs_fp8.py"

echo "=== Qwen2.5-0.5B (per-row fp8) ==="
$PY \
  --base-model Qwen/Qwen2.5-0.5B \
  --teacher-id RedHatAI/Qwen2.5-0.5B-FP8-dynamic \
  --dtype bfloat16 \
  --weight-bits 24 --activation-bits 24 \
  --rmsnorm-bits 24 --attn-matmul-bits 24 \
  --softmax-lut-size 4096 --silu-lut-size 4096 \
  --int-embedding --embedding-bits 24 \
  --n-prompts 100 --max-len 512 \
  --out "$OUT_DIR/qwen25_0p5b.jsonl" 2>&1 | tee "$OUT_DIR/qwen25_0p5b.log"

echo "=== Llama-3.1-8B-Instruct (per-row fp8) ==="
$PY \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --teacher-id RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
  --dtype bfloat16 \
  --weight-bits 24 --activation-bits 24 \
  --rmsnorm-bits 24 --attn-matmul-bits 24 \
  --softmax-lut-size 4096 --silu-lut-size 4096 \
  --int-embedding --embedding-bits 24 \
  --n-prompts 100 --max-len 512 \
  --out "$OUT_DIR/llama31_8b.jsonl" 2>&1 | tee "$OUT_DIR/llama31_8b.log"

echo "=== Qwen3-8B (block fp8 — hardest) ==="
$PY \
  --base-model Qwen/Qwen3-8B \
  --teacher-id Qwen/Qwen3-8B-FP8 \
  --dtype bfloat16 \
  --weight-bits 24 --activation-bits 24 \
  --rmsnorm-bits 24 --attn-matmul-bits 24 \
  --softmax-lut-size 4096 --silu-lut-size 4096 \
  --int-embedding --embedding-bits 24 \
  --n-prompts 100 --max-len 512 \
  --out "$OUT_DIR/qwen3_8b.jsonl" 2>&1 | tee "$OUT_DIR/qwen3_8b.log"

echo "=== done. summary: ==="
for f in "$OUT_DIR"/*.jsonl; do
  echo "--- $f"
  python3 -c "
import json,sys
with open('$f') as g:
    d = json.loads(g.readline())
keys = ['student_vs_teacher/top1','student_vs_teacher/margin_mean','student_vs_teacher/margin_p99',
        'student_vs_teacher/kl_p99','student_vs_teacher/n_positions']
for k in keys:
    v = d.get(k)
    print(f'  {k}: {v}')
"
done
