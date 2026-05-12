#!/usr/bin/env bash
# Cache wikitext-103 prompts for all three base tokenizers.
# 100 prompts × max 256 tokens. The training script reserves 20 for eval and
# uses the rest for training. Idempotent: existing .pt files are overwritten.
set -euo pipefail

cd "$(dirname "$0")/../../.."  # project root

PY=${PYTHON:-python3}
mkdir -p experiments/fp-emulation/data

for triple in \
    "Qwen/Qwen2.5-0.5B prompts_qwen25.pt" \
    "Qwen/Qwen3-8B prompts_qwen3.pt" \
    "meta-llama/Llama-3.1-8B-Instruct prompts_llama31.pt"; do
    tok=$(echo "$triple" | awk '{print $1}')
    out=$(echo "$triple" | awk '{print $2}')
    echo "=== Caching prompts for $tok -> $out ==="
    $PY -m difr_expt.cache_prompts \
        --tokenizer "$tok" \
        --n-prompts 100 \
        --max-len 256 \
        --out "experiments/fp-emulation/data/$out"
done

echo "=== All prompt caches written ==="
ls -lh experiments/fp-emulation/data/prompts_*.pt
