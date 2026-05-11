#!/usr/bin/env bash
# Phase C — Per-group weight quant at fine granularity, weight-only and full.
# Tests if per-group weight quant brings w=24 / w=22 / w=20 weight-only into
# the 99.9% regime. If yes, then w=a=24 with per-group should clear the
# int64-safe bar.
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

# Weight-only at b=24 with per-group sym at multiple grp sizes (drives weight quant error to ~0)
run_one "w24_aoff_grp128" --weight-bits 24 --activation-bits 0 --quant-scheme per_group_sym --group-size 128
run_one "w24_aoff_grp64"  --weight-bits 24 --activation-bits 0 --quant-scheme per_group_sym --group-size 64
run_one "w24_aoff_grp32"  --weight-bits 24 --activation-bits 0 --quant-scheme per_group_sym --group-size 32
run_one "w24_aoff_grp16"  --weight-bits 24 --activation-bits 0 --quant-scheme per_group_sym --group-size 16

# w=20 weight-only group sweep — closer to int64 budget
run_one "w20_aoff_grp32" --weight-bits 20 --activation-bits 0 --quant-scheme per_group_sym --group-size 32
run_one "w20_aoff_grp16" --weight-bits 20 --activation-bits 0 --quant-scheme per_group_sym --group-size 16

# w=16 weight-only group sweep — smallest reasonable weight
run_one "w16_aoff_grp32" --weight-bits 16 --activation-bits 0 --quant-scheme per_group_sym --group-size 32
run_one "w16_aoff_grp16" --weight-bits 16 --activation-bits 0 --quant-scheme per_group_sym --group-size 16

echo "done"
