#!/usr/bin/env bash
# Run inside the vast.ai H100 container as root, immediately after SSH.
# Fixes CUDA driver symlinks (the Nix image doesn't ship them as torch expects)
# and installs the python deps that the project image doesn't carry.
set -euo pipefail

echo "=== CUDA driver symlinks ==="
cd /usr/lib/x86_64-linux-gnu
for f in libcuda libcudadebugger libnvidia-ml libnvidia-nvvm libnvidia-ptxjitcompiler \
         libnvcuvid libnvidia-opencl libnvidia-cfg libnvidia-gpucomp \
         libnvidia-opticalflow libnvidia-sandboxutils; do
    src=$(ls ${f}.so.[0-9]*.[0-9]*.[0-9]* 2>/dev/null | head -1 || true)
    rm -f "${f}.so.1" "${f}.so"
    if [ -n "$src" ]; then
        ln -sf "$src" "${f}.so.1"
        ln -sf "$src" "${f}.so"
        echo "  $f -> $src"
    fi
done
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

echo "=== Python dep install ==="
# --break-system-packages because the Nix container has a system Python with no
# user-writable site-packages convention. Fine for an ephemeral instance.
pip install --break-system-packages \
    compressed-tensors fp_quant bitsandbytes==0.46.1 huggingface_hub

echo "=== Sanity ==="
cd ~
python3 -c "
import torch
print('torch:', torch.__version__)
print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('mem:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
import compressed_tensors; print('compressed_tensors:', compressed_tensors.__version__)
import fp_quant; print('fp_quant:', fp_quant.__version__)
import bitsandbytes; print('bitsandbytes:', bitsandbytes.__version__)
"

echo "=== Setup complete ==="
echo "Remember: export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu on every SSH session."
echo "If using Llama: huggingface-cli login --token \$HF_TOKEN"
