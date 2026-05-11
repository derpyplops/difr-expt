#!/usr/bin/env bash
# Phase C — SmoothQuant rescaling and activation outlier clipping.
# All variants run at int64-safe widths only.
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

# Baseline reminder: w=a=24 sym alone gives 0.985-0.986 top-1

# SmoothQuant at w=a=24, several alpha values
run_one "b24_sq_a05" --weight-bits 24 --activation-bits 24 --smoothquant-alpha 0.5
run_one "b24_sq_a06" --weight-bits 24 --activation-bits 24 --smoothquant-alpha 0.6
run_one "b24_sq_a07" --weight-bits 24 --activation-bits 24 --smoothquant-alpha 0.7
run_one "b24_sq_a08" --weight-bits 24 --activation-bits 24 --smoothquant-alpha 0.8
run_one "b24_sq_a09" --weight-bits 24 --activation-bits 24 --smoothquant-alpha 0.9

# Clip variant
run_one "b24_clip999" --weight-bits 24 --activation-bits 24 --act-clip-quantile 0.999
run_one "b24_clip9999" --weight-bits 24 --activation-bits 24 --act-clip-quantile 0.9999

# Combine smoothquant + clip
run_one "b24_sq_a05_clip999" --weight-bits 24 --activation-bits 24 --smoothquant-alpha 0.5 --act-clip-quantile 0.999

# SmoothQuant at smaller bits (b=20) to check budget headroom
run_one "b20_sq_a05" --weight-bits 20 --activation-bits 20 --smoothquant-alpha 0.5

# SmoothQuant + per-group sym
run_one "b24_grp32_sq05" --weight-bits 24 --activation-bits 24 --quant-scheme per_group_sym --group-size 32 --smoothquant-alpha 0.5

echo "done"
