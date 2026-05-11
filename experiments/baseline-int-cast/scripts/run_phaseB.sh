#!/usr/bin/env bash
# Phase B: smarter quant at the int64-safe ceiling (b=24).
# Variants: per-group symmetric, asymmetric, mixed precision.
# Usage: run_phaseB.sh <model_alias> <model_path>
set -e
ALIAS=$1
MODEL=$2
cd /root/difr-expt
mkdir -p experiments/baseline-int-cast/data logs

run_one() {
  local TAG=$1; shift
  local OUT="experiments/baseline-int-cast/data/${ALIAS}_${TAG}.json"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT exists"
    return
  fi
  echo "[run] ${ALIAS} ${TAG}"
  PYTHONPATH=src python3 -m difr_expt.run_baseline \
    --model "$MODEL" \
    --dtype bfloat16 \
    --n-prompts 100 \
    --max-len 512 \
    --matmul-dtype bf16 \
    --out "$OUT" "$@" \
    > "logs/${ALIAS}_${TAG}.log" 2>&1
}

# Per-group symmetric at b=24
run_one "b24_grp128_sym" --weight-bits 24 --activation-bits 24 --quant-scheme per_group_sym --group-size 128
run_one "b24_grp64_sym"  --weight-bits 24 --activation-bits 24 --quant-scheme per_group_sym --group-size 64
run_one "b24_grp32_sym"  --weight-bits 24 --activation-bits 24 --quant-scheme per_group_sym --group-size 32

# Asymmetric (full, sym groups can't bring zero-point so a separate run)
run_one "b24_asym" --weight-bits 24 --activation-bits 24 --quant-scheme asymmetric

# Mixed precision: keep most layers at b=24 but bump LM head + first/last blocks
run_one "b24_nolm" --weight-bits 24 --activation-bits 24 --skip-lm-head

# Asymmetric weights b=24 activations sym b=24 - same as asym (above).

# w=20, a=28 (higher activation precision, weight savings) — product is 48 bits, int64 safe
run_one "w20_a28" --weight-bits 20 --activation-bits 28

# w=28, a=20 (higher weight precision)
run_one "w28_a20" --weight-bits 28 --activation-bits 20

echo "done"
