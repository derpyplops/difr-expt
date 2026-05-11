#!/usr/bin/env bash
# Phase C - Diagnostic: weight-only PTQ at finer bit-widths to find the
# threshold at which bf16-cast becomes bit-exact.
# Usage: run_phaseC_woff.sh <model_alias> <model_path>
set -e
ALIAS=$1
MODEL=$2
cd /root/difr-expt
mkdir -p experiments/baseline-int-cast/data logs

for B in 25 26 27 28 29 30; do
  OUT="experiments/baseline-int-cast/data/${ALIAS}_w${B}_aoff.json"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT exists"
    continue
  fi
  echo "[run] ${ALIAS} w=$B a=0 -> $OUT"
  PYTHONPATH=src python3 -m difr_expt.run_baseline \
    --model "$MODEL" \
    --dtype bfloat16 \
    --n-prompts 100 \
    --max-len 512 \
    --weight-bits $B --activation-bits 0 \
    --matmul-dtype bf16 \
    --out "$OUT" \
    > "logs/${ALIAS}_w${B}_aoff.log" 2>&1
done
echo "done"
