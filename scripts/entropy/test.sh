#!/bin/bash -l
#SBATCH --job-name=dk_tests
#SBATCH --qos=jgiezgala_common
#SBATCH --partition=common
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=0-02:00:00
#SBATCH --output="logs/slurm_tests_%j.out"
#SBATCH --error="logs/slurm_tests_%j.err"

# scripts/entropy/run_tests.sh

set -euo pipefail

mkdir -p logs

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
    srun pytest tests/ -v --tb=short \
        -k "not test_full_active_learning_cycle" \
        --timeout=60 \
        2>&1 | tee "logs/test_results_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Running full test suite..."
    srun pytest tests/ -v --tb=short \
        --timeout=300 \
        2>&1 | tee "logs/test_results_$(date +%Y%m%d_%H%M%S).log"
fi

echo ""
echo "=== Tests complete ==="
