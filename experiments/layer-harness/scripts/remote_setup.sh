#!/usr/bin/env bash
# Bring up the difr-expt env on a fresh vast.ai pytorch container so we can
# run the layer-harness with `--student fp8-hw`. Idempotent — re-running is
# safe.
#
# Expectations:
#   - PyTorch image (e.g. pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel) already
#     has python3, pip, torch, CUDA. nvidia-smi works (this is NOT the
#     Nix-only deterministic-serving image — that path needs the libcuda
#     symlinks dance).
#   - The repo has been rsync'd to $REPO (default /workspace/difr-expt).
#
# Run with:
#   bash /workspace/difr-expt/experiments/layer-harness/scripts/remote_setup.sh
set -euo pipefail

REPO=${REPO:-/workspace/difr-expt}
cd "$REPO"

echo "[remote_setup] python: $(python3 --version)"
echo "[remote_setup] torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "[remote_setup] cuda: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
echo "[remote_setup] gpu: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))')"

# Install the project in editable mode + the compressed_tensors dep needed
# for the real-FP8 student. We skip the [gpu] extras (vllm/xformers/
# bitsandbytes) because the harness doesn't use them.
pip install --no-cache-dir -e . compressed-tensors accelerate datasets tqdm

# Sanity: real _scaled_mm is present on this torch build.
python3 - <<'PY'
import torch
assert hasattr(torch, "_scaled_mm"), "torch._scaled_mm missing — need torch ≥ 2.3"
caps = torch.cuda.get_device_capability(0)
assert caps >= (8, 9), f"GPU SM_{caps[0]}{caps[1]} too old for FP8 (need SM_89+)"
print(f"[remote_setup] _scaled_mm available, GPU SM_{caps[0]}{caps[1]} ✓")
PY

echo "[remote_setup] done. Ready for: python -m difr_expt.run_harness --student fp8-hw ..."
