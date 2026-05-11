#!/usr/bin/env bash
# Phase C — Per-group both w and a, fine groups (g=8, g=16, g=4) at b=24.
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

run_one "b24_grp16_both" --weight-bits 24 --activation-bits 24 --quant-scheme per_group_sym --group-size 16
run_one "b24_grp8_both"  --weight-bits 24 --activation-bits 24 --quant-scheme per_group_sym --group-size 8
run_one "b24_grp4_both"  --weight-bits 24 --activation-bits 24 --quant-scheme per_group_sym --group-size 4

# Also try smaller bits + fine groups
run_one "b20_grp16_both" --weight-bits 20 --activation-bits 20 --quant-scheme per_group_sym --group-size 16
run_one "b20_grp8_both"  --weight-bits 20 --activation-bits 20 --quant-scheme per_group_sym --group-size 8

echo "done"
