#!/usr/bin/env bash
# Phase 2: 5 training runs, sequential, smallest-first.
# Each run logs metrics.jsonl + saves best.pt; pre-train eval is the Phase 1 baseline.
#
# Usage:
#   ./run_phase2_training.sh                # all 5 runs
#   ./run_phase2_training.sh 1 2            # just runs 1 and 2 (by index)
set -euo pipefail

cd "$(dirname "$0")/../../.."

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu}
PY=${PYTHON:-python3}

# Run config: (base_model, teacher_id, prompts_file, out_subdir, extra_flags)
# LR-luts 1e-5 (NOT the 1e-3 default — that destroyed the softmax LUT in
# the first attempt, collapsing top-1 from 0.91 to 0.008 by step 100).
# 8B runs use 5x lower LRs + 2.5x longer warmup + tighter grad clip — at the
# original 1e-5 LR + 8-bit AdamW + warmup=20, Qwen3-8B diverged at warmup
# peak (loss 0.018 → 3.77 between step 1 and step 20).
SMALL="--lr 1e-5 --lr-luts 1e-5 --lr-gamma-bias 1e-4 --warmup 20 --grad-clip 1.0"
BIG="--lr 2e-6 --lr-luts 2e-6 --lr-gamma-bias 2e-5 --warmup 50 --grad-clip 0.5"

RUNS=(
    "Qwen/Qwen2.5-0.5B|RedHatAI/Qwen2.5-0.5B-FP8-dynamic|prompts_qwen25.pt|qwen25_fp8|$SMALL"
    "Qwen/Qwen3-8B|Qwen/Qwen3-8B-FP8|prompts_qwen3.pt|qwen3_8b_fp8|$BIG --use-8bit-adamw --grad-checkpointing --no-fp32-ref"
    "Qwen/Qwen3-8B|nvidia/Qwen3-8B-NVFP4|prompts_qwen3.pt|qwen3_8b_nvfp4|$BIG --use-8bit-adamw --grad-checkpointing --no-fp32-ref"
    "meta-llama/Llama-3.1-8B-Instruct|RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic|prompts_llama31.pt|llama31_8b_fp8|$BIG --use-8bit-adamw --grad-checkpointing --no-fp32-ref"
    "meta-llama/Llama-3.1-8B-Instruct|nvidia/Llama-3.1-8B-Instruct-NVFP4|prompts_llama31.pt|llama31_8b_nvfp4|$BIG --use-8bit-adamw --grad-checkpointing --no-fp32-ref"
)

INDEXES=("$@")
if [ ${#INDEXES[@]} -eq 0 ]; then
    INDEXES=(1 2 3 4 5)
fi

NTFY_TOPIC=${NTFY_TOPIC:-claude-jon-alerts}
# Use python urllib so this works on the Nix container too (no curl).
notify() {
    python3 -c "
import urllib.request, sys
req = urllib.request.Request(
    'https://ntfy.sh/$NTFY_TOPIC',
    data=sys.argv[2].encode(),
    headers={'Title': sys.argv[1], 'Tags': 'gpu'},
)
try:
    urllib.request.urlopen(req, timeout=10)
except Exception:
    pass
" "$1" "$2"
}

for idx in "${INDEXES[@]}"; do
    i=$((idx - 1))
    IFS='|' read -r model teacher prompts outdir extra <<< "${RUNS[$i]}"
    echo "=== Run $idx/5: $model <- $teacher ==="
    notify "Run $idx/5 starting" "$model <- $teacher"
    t0=$(date +%s)
    set +e
    $PY -m difr_expt.train_emulate \
        --model "$model" \
        --teacher-source published \
        --teacher-id "$teacher" \
        --prompts "experiments/fp-emulation/data/$prompts" \
        --out "experiments/fp-emulation/data/$outdir" \
        --steps 500 --batch 2 --eval-every 100 --warmup 20 \
        --plateau-patience 5 \
        --dtype bfloat16 \
        $extra 2>&1 | tee "experiments/fp-emulation/data/${outdir}.log"
    rc=$?
    set -e
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
        notify "Run $idx/5 FAILED (rc=$rc)" "$outdir after ${dt}s — check log"
        echo "Run $idx failed; stopping the sequence."
        exit $rc
    fi
    notify "Run $idx/5 done in ${dt}s" "$(python3 -c "import json; s=json.load(open('experiments/fp-emulation/data/$outdir/summary.json')); print(f\"pre top1={s['pre']['student_vs_teacher/top1']:.4f}  post top1={s['post']['student_vs_teacher/top1']:.4f}\")" 2>/dev/null || echo 'no summary')"
    # Reclaim disk: pretrain.pt (initial weights, same for every run) and
    # final.pt (last step, redundant with best.pt) can be huge on 8B (~32GB each).
    rm -f "experiments/fp-emulation/data/$outdir/pretrain.pt"
    rm -f "experiments/fp-emulation/data/$outdir/final.pt"
done

echo "=== All requested runs complete ==="
notify "All training runs complete" "Ready for Phase 3 report"
