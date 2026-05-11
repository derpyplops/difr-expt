#!/usr/bin/env bash
# Phase C — Mixed precision w/a, int64 safe (w+a ≤ 49 for dim 14336).
# We learned weight quant dominates the error, activations don't matter much.
# So bump w, keep a modest.
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

# int64 budget: w*a-products + log2(dim). dim 14336 -> 14 bits. So w+a ≤ 49.
# Try a few from the Pareto frontier.
run_one "w28_a20" --weight-bits 28 --activation-bits 20
run_one "w28_a21" --weight-bits 28 --activation-bits 21
run_one "w29_a20" --weight-bits 29 --activation-bits 20
run_one "w30_a18" --weight-bits 30 --activation-bits 18
run_one "w30_a19" --weight-bits 30 --activation-bits 19
# Bonus: w=27 (just below the 99.9 threshold weight-only) with high a
run_one "w27_a22" --weight-bits 27 --activation-bits 22

echo "done"
