#!/usr/bin/env bash
# Phase A: bit-width sweep with the corrected bf16 / F.linear path.
# Usage: run_phaseA.sh <model_alias> <model_path> <out_prefix>
# Example: run_phaseA.sh qwen25_0p5b Qwen/Qwen2.5-0.5B qwen25_0p5b
set -e
ALIAS=$1
MODEL=$2
PREFIX=$3
cd /root/difr-expt
mkdir -p experiments/baseline-int-cast/data logs

# All bits we want, including the out-of-budget references (26, 28).
for B in 16 18 20 22 24 26 28; do
  OUT="experiments/baseline-int-cast/data/${PREFIX}_b${B}_mmbf16_flin.json"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT exists"
    continue
  fi
  echo "[run] $ALIAS b=$B -> $OUT"
  PYTHONPATH=src python3 -m difr_expt.run_baseline \
    --model "$MODEL" \
    --dtype bfloat16 \
    --n-prompts 100 \
    --max-len 512 \
    --weight-bits $B --activation-bits $B \
    --matmul-dtype bf16 \
    --out "$OUT" \
    > "logs/${PREFIX}_b${B}.log" 2>&1
done
echo "done"
