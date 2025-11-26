#!/bin/bash

# Exit on error
set -e

echo "=== conda environment ==="
conda env create -f environments/default.yml
eval "$(conda shell.bash hook)"
conda activate sam3d-objects

echo "=== Setting up PyTorch/CUDA dependencies ==="
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

echo "=== Installing sam3d-objects and core dependencies ==="
pip install -e '.[dev]'

echo "=== Installing pytorch3d dependency ==="
pip install -e '.[p3d]'

echo "=== Setting up inference dependencies ==="
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

echo "=== Applying patches ==="
./patching/hydra

echo "=== Setup complete! ==="