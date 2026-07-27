#!/bin/bash

set -e

cd "$(dirname "$0")"

conda create -n anyssr python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anyssr

pip install -r requirements.txt

pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE \
  pip install flash-attn==2.7.2.post1 --no-build-isolation

bash scripts/train_olora_permutation.sh
