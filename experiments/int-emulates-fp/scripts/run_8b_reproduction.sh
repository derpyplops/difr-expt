#!/usr/bin/env bash
# Reproduce int24-vs-fp8 results on 8B (Qwen3-8B + Llama-3.1-8B-Instruct).
# Runs serially on one H100 80GB. Each model: diag + V1 + V13 + V2.
set -euo pipefail

cd "$(dirname "$0")/../../.."
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu}
export HF_HOME=${HF_HOME:-/root/hf-cache}
export HF_TOKEN=${HF_TOKEN:-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}

PY=${PYTHON:-python3}
DATA=experiments/int-emulates-fp/data
SCRIPTS=experiments/int-emulates-fp/scripts
PROMPTS_DIR=experiments/fp-emulation/data

# 8B-specific training config (mirrors run_phase2_training.sh BIG profile).
# --skip-pretrain-checkpoint --skip-final-checkpoint: 8B fp32 shadows = 28 GB each,
# disk on the vast container is 150 GB total (~76 GB free); writing both
# alongside best.pt would blow the disk. We keep only best.pt.
BIG_TRAIN="--lr 2e-6 --lr-luts 2e-6 --lr-gamma-bias 2e-5 --warmup 50 --grad-clip 0.5 \
           --use-8bit-adamw --grad-checkpointing --no-fp32-ref --no-matmul-hooks \
           --init-from-teacher --no-trainable-gamma-bias --no-trainable-luts \
           --dtype bfloat16 --batch 1 --steps 500 --eval-every 100 --plateau-patience 5 \
           --skip-pretrain-checkpoint --skip-final-checkpoint \
           --teacher-source published"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

NTFY_TOPIC=${NTFY_TOPIC:-claude-jon-alerts}
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

run_model() {
    local short=$1     # qwen3_8b or llama31_8b
    local base=$2      # HF base id
    local teacher=$3   # HF teacher id
    local prompts=$4   # prompts file name

    echo "=========================================="
    echo "=== $short: base=$base teacher=$teacher"
    echo "=========================================="

    # ===== 1. base-vs-teacher diagnostic =====
    local diag_out="$DATA/${short}_base_vs_teacher.json"
    echo "--- [$short] diag base-vs-teacher ---"
    notify "${short} diag starting" "base-vs-teacher"
    $PY $SCRIPTS/diag_base_vs_teacher.py \
        --model "$base" \
        --teacher-id "$teacher" \
        --prompts "$PROMPTS_DIR/$prompts" \
        --n-prompts 20 --dtype bfloat16 \
        2>&1 | tee "$DATA/${short}_base_vs_teacher.log"

    # Parse the top1/kl numbers from the log into a JSON for upload.
    $PY -c "
import re, json, sys
with open('$DATA/${short}_base_vs_teacher.log') as f:
    log = f.read()
m_t1 = re.search(r'top1 \(base-with-cast vs teacher\) = ([\d.]+)', log)
m_klm = re.search(r'kl_mean=([\d\.eE+-]+)', log)
m_klp = re.search(r'kl_p99=([\d\.eE+-]+)', log)
m_n = re.search(r'n_positions=(\d+)', log)
out = {
    'model': '$base',
    'teacher': '$teacher',
    'top1_base_vs_teacher': float(m_t1.group(1)) if m_t1 else None,
    'kl_mean': float(m_klm.group(1)) if m_klm else None,
    'kl_p99': float(m_klp.group(1)) if m_klp else None,
    'n_positions': int(m_n.group(1)) if m_n else None,
}
with open('$diag_out', 'w') as f:
    json.dump(out, f, indent=2)
print('wrote', '$diag_out', '->', out)
"
    notify "${short} diag done" "$(python3 -c "import json; d=json.load(open('$diag_out')); print(f'top1={d[\"top1_base_vs_teacher\"]:.4f}')")"

    # ===== 2. V1 naive (steps=0) =====
    local v1_dir="$DATA/${short}_v1_naive"
    echo "--- [$short] V1 naive ---"
    notify "${short} V1 starting" ""
    rm -rf "$v1_dir"
    $PY -m difr_expt.train_emulate \
        --model "$base" --teacher-source published --teacher-id "$teacher" \
        --prompts "$PROMPTS_DIR/$prompts" \
        --out "$v1_dir" \
        --steps 0 --init-from-teacher --no-fp32-ref \
        --skip-pretrain-checkpoint --skip-final-checkpoint \
        --dtype bfloat16 \
        --no-trainable-gamma-bias --no-trainable-luts \
        2>&1 | tee "$v1_dir.log"
    # Sanity: strip anything that leaked through.
    rm -f "$v1_dir/pretrain.pt" "$v1_dir/final.pt" "$v1_dir/best.pt"
    notify "${short} V1 done" "$(python3 -c "import json; s=json.load(open('$v1_dir/summary.json')); print(f'top1={s[\"pre\"][\"student_vs_teacher/top1\"]:.4f}')" 2>/dev/null || echo 'no summary')"

    # ===== 3. V13 high-bits (bits=31, steps=0) =====
    local v13_dir="$DATA/${short}_v13_high_bits"
    echo "--- [$short] V13 high-bits ---"
    notify "${short} V13 starting" ""
    rm -rf "$v13_dir"
    $PY -m difr_expt.train_emulate \
        --model "$base" --teacher-source published --teacher-id "$teacher" \
        --prompts "$PROMPTS_DIR/$prompts" \
        --out "$v13_dir" \
        --steps 0 --init-from-teacher --no-fp32-ref \
        --skip-pretrain-checkpoint --skip-final-checkpoint \
        --weight-bits 31 --activation-bits 31 --attn-matmul-bits 31 --rmsnorm-bits 31 \
        --dtype bfloat16 \
        --no-trainable-gamma-bias --no-trainable-luts \
        2>&1 | tee "$v13_dir.log"
    rm -f "$v13_dir/pretrain.pt" "$v13_dir/final.pt" "$v13_dir/best.pt"
    notify "${short} V13 done" "$(python3 -c "import json; s=json.load(open('$v13_dir/summary.json')); print(f'top1={s[\"pre\"][\"student_vs_teacher/top1\"]:.4f}')" 2>/dev/null || echo 'no summary')"

    # ===== 4. V2 weights-only training =====
    local v2_dir="$DATA/${short}_v2_weights_only"
    echo "--- [$short] V2 weights-only training (500 steps) ---"
    notify "${short} V2 training starting" "500 steps, ~30-60 min"
    rm -rf "$v2_dir"
    t0=$(date +%s)
    $PY -m difr_expt.train_emulate \
        --model "$base" --teacher-source published --teacher-id "$teacher" \
        --prompts "$PROMPTS_DIR/$prompts" \
        --out "$v2_dir" \
        $BIG_TRAIN \
        2>&1 | tee "$v2_dir.log"
    dt=$(( $(date +%s) - t0 ))
    rm -f "$v2_dir/pretrain.pt" "$v2_dir/final.pt"
    notify "${short} V2 done in ${dt}s" "$(python3 -c "import json; s=json.load(open('$v2_dir/summary.json')); print(f'pre={s[\"pre\"][\"student_vs_teacher/top1\"]:.4f} post={s[\"post\"][\"student_vs_teacher/top1\"]:.4f} best_step={s[\"best_step\"]}')" 2>/dev/null || echo 'no summary')"

    echo "=========================================="
    echo "=== $short DONE"
    echo "=========================================="
}

# Allow targeting a single model from CLI: "qwen3" or "llama31"
ONLY=${1:-}

if [ -z "$ONLY" ] || [ "$ONLY" = "qwen3" ]; then
    run_model qwen3_8b "Qwen/Qwen3-8B" "Qwen/Qwen3-8B-FP8" "prompts_qwen3.pt"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = "llama31" ]; then
    run_model llama31_8b "meta-llama/Llama-3.1-8B-Instruct" "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic" "prompts_llama31.pt"
fi

echo "=== All requested 8B runs complete ==="
notify "8B reproduction complete" "Ready for analysis"
