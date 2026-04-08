#!/bin/bash
# Run this ONCE interactively on a login node to set up the verl conda env and clone the repo.
# Usage: bash scripts/setup_verl.sh
# Takes ~20-40 minutes due to flash-attn compilation.

set -e

module load Miniforge3/26.1.0-fasrc01
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Repo root: ${REPO_ROOT}"

# Step 1: Clone verl into repo root if not already present
if [ ! -d "${REPO_ROOT}/verl" ]; then
    echo "==> Cloning verl..."
    git clone https://github.com/volcengine/verl.git "${REPO_ROOT}/verl"
    cd "${REPO_ROOT}/verl"
    git reset --hard 083da9ab130efa2dc284eeb821a3edd6ce570fe3
    cd "${REPO_ROOT}"
else
    echo "==> verl directory already exists, skipping clone."
fi

# Step 2: Create conda env
if conda env list | grep -q "^verl "; then
    echo "==> conda env 'verl' already exists, skipping creation."
else
    echo "==> Creating conda env 'verl' with python 3.10..."
    conda create -n verl python=3.10 -y
fi

conda activate verl

# Step 3: Install dependencies
echo "==> Installing PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing transformers + vllm..."
pip install transformers==4.51.1 vllm==0.8.5.post1

echo "==> Installing flash-attn (this compiles and takes ~15-20 min)..."
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

echo "==> Installing verl in editable mode..."
cd "${REPO_ROOT}/verl"
pip3 install -e .

echo "==> Re-pinning transformers + vllm (verl may have upgraded them)..."
pip install transformers==4.51.1 vllm==0.8.5.post1

echo "==> Installing flashinfer..."
pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6/

echo ""
echo "============================================================"
echo "Setup complete! Test with:"
echo "  conda activate verl && python -c 'import verl; print(verl.__file__)'"
echo "============================================================"
