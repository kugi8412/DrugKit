#!/bin/bash
# scripts/entropy/run_comparison.sh
# Run the DrugKit vs Deep Docking comparison benchmark
# Usage: bash scripts/entropy/run_comparison.sh
#
# This runs the head-to-head comparison between:
#   - DrugKit: GINE Siamese RankNet (graph-based)
#   - Original DD: MLP + Morgan Fingerprints (1024-bit)
#
# Uses the ESOL dataset (1128 drug-like molecules) as benchmark

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

echo "=== DrugKit vs Deep Docking Protocol — Comparison Benchmark ==="
echo "Server: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python --version)"
echo ""

# Ensure ESOL dataset exists
if [ ! -f "testing_data/esol_filtered.csv" ]; then
    echo "Downloading ESOL dataset..."
    curl -s --max-time 30 -L \
        "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv" \
        -o /tmp/esol.csv
    python smiles_processing_comparison/prepare_dataset.py
fi

mkdir -p logs results

echo "Running comparison benchmark..."
echo ""

# Run the comparison test with verbose output
pytest tests/test_deep_docking_comparison.py -v -s \
    --tb=short \
    2>&1 | tee "logs/comparison_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== Benchmark complete ==="
echo "Logs saved to logs/"
