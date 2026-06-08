#!/bin/bash
# scripts/entropy/run_tests.sh
# Run the test suite on MIMUW entropy server
# Usage: bash scripts/entropy/run_tests.sh [--quick]

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

echo "=== DrugKit Test Suite ==="
echo "Server: $(hostname)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Parse arguments
QUICK=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
    esac
done

if [ "$QUICK" = true ]; then
    echo "Running quick tests (no integration)..."
    pytest tests/ -v --tb=short \
        -k "not test_full_active_learning_cycle" \
        --timeout=60 \
        2>&1 | tee "logs/test_results_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Running full test suite..."
    pytest tests/ -v --tb=short \
        --timeout=300 \
        2>&1 | tee "logs/test_results_$(date +%Y%m%d_%H%M%S).log"
fi

echo ""
echo "=== Tests complete ==="
