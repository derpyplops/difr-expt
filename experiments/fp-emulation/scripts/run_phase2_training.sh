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
RUNS=(
    "Qwen/Qwen2.5-0.5B|RedHatAI/Qwen2.5-0.5B-FP8-dynamic|prompts_qwen25.pt|qwen25_fp8|"
    "Qwen/Qwen3-8B|Qwen/Qwen3-8B-FP8|prompts_qwen3.pt|qwen3_8b_fp8|--use-8bit-adamw --grad-checkpointing --no-fp32-ref"
    "Qwen/Qwen3-8B|nvidia/Qwen3-8B-NVFP4|prompts_qwen3.pt|qwen3_8b_nvfp4|--use-8bit-adamw --grad-checkpointing --no-fp32-ref"
    "meta-llama/Llama-3.1-8B-Instruct|RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic|prompts_llama31.pt|llama31_8b_fp8|--use-8bit-adamw --grad-checkpointing --no-fp32-ref"
    "meta-llama/Llama-3.1-8B-Instruct|nvidia/Llama-3.1-8B-Instruct-NVFP4|prompts_llama31.pt|llama31_8b_nvfp4|--use-8bit-adamw --grad-checkpointing --no-fp32-ref"
)

INDEXES=("$@")
if [ ${#INDEXES[@]} -eq 0 ]; then
    INDEXES=(1 2 3 4 5)
fi

NTFY_TOPIC=${NTFY_TOPIC:-claude-jon-alerts}
notify() {
    curl -s -d "$2" -H "Title: $1" -H "Tags: gpu" "ntfy.sh/$NTFY_TOPIC" >/dev/null || true
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
done

echo "=== All requested runs complete ==="
notify "All training runs complete" "Ready for Phase 3 report"
