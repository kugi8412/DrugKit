#!/bin/bash
# scripts/entropy/run_active_learning.sh
# Run the active learning pipeline on MIMUW entropy server
# Usage: bash scripts/entropy/run_active_learning.sh [--gpu]
#
# Requirements:
#   - config.yaml in project root with active_learning section
#   - Docking grids JSON file
#   - Receptor .pdbqt files in data/ directory
#   - Pool CSV and seed CSV files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Activate environment
ENV_NAME="drugkit"
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
fi
conda activate "${ENV_NAME}"

# Parse arguments
USE_GPU=false
for arg in "$@"; do
    case $arg in
        --gpu) USE_GPU=true ;;
    esac
done

echo "=== DrugKit Active Learning Pipeline ==="
echo "Server: $(hostname)"
echo "Date: $(date)"

if [ "$USE_GPU" = true ]; then
    echo "GPU mode requested"
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  No NVIDIA GPUs found"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "CPU mode"
    export CUDA_VISIBLE_DEVICES=""
fi

# Ensure output directories exist
mkdir -p logs output/active_learning/poses

# Run
echo ""
echo "Starting active learning loop..."
python -m active_learning.run_active_learning 2>&1 | tee "logs/al_run_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== Pipeline complete ==="
echo "Results in: output/active_learning/"
