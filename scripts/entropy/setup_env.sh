#!/bin/bash
# scripts/entropy/setup_env.sh
# Setup conda environment on MIMUW entropy server
# Usage: bash scripts/entropy/setup_env.sh

set -euo pipefail

ENV_NAME="drugkit"
CONDA_BASE="${HOME}/miniconda3"

echo "=== DrugKit Environment Setup on Entropy ==="
echo "Server: $(hostname)"
echo "User: $(whoami)"
echo "Date: $(date)"

# Load conda if available
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "[ERROR] Conda not found. Install miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_BASE}"
    exit 1
fi

# Create or update environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Updating existing '${ENV_NAME}' environment..."
    conda env update -n "${ENV_NAME}" -f config/drugkit.yaml --prune
else
    echo "Creating '${ENV_NAME}' environment..."
    conda env create -f config/drugkit.yaml -n "${ENV_NAME}"
fi

conda activate "${ENV_NAME}"

# Install package in editable mode
pip install -e ".[dev]"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run tests: pytest tests/ -v"
