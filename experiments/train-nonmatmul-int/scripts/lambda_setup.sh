#!/bin/bash
# Run on Lambda Labs A100 box (ubuntu user, Ubuntu 22.04, pytorch preinstalled).
# Expects /home/ubuntu/difr-expt to already exist via scp + tar extract.

set -e
cd /home/ubuntu/difr-expt

# Install missing deps. Lambda's base image ships torch + tooling; we add
# transformers/accelerate/datasets pinned to pyproject versions.
pip install --quiet --upgrade pip
pip install --quiet \
  "transformers==4.57.3" \
  "accelerate>=1.0.1" \
  "datasets>=3.1.0" \
  "safetensors>=0.5.3" \
  "tqdm" \
  "numpy"

# Editable install of the project package
pip install --quiet -e .

# Smoke check
python -c "import torch, transformers, accelerate; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'tx', transformers.__version__)"
python -c "import torch; print('device', torch.cuda.get_device_name(0))"
