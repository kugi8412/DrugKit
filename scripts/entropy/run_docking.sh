#!/bin/bash
# scripts/entropy/run_docking.sh
# Run docking pipeline (Smina or Vina) on MIMUW entropy server
# Usage: bash scripts/entropy/run_docking.sh [smina|vina] [--ncpu N]
#
# Prerequisites:
#   - Receptor .pdbqt files prepared
#   - Docking grids JSON generated (via pocket_logic)
#   - Known ligands CSV and candidates CSV

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
ENGINE="${1:-smina}"
NCPU="${NPROC:-$(nproc)}"

for arg in "$@"; do
    case $arg in
        --ncpu)
            shift
            NCPU="${1:-$(nproc)}"
            shift
            ;;
        smina|vina) ENGINE="$arg" ;;
    esac
done

echo "=== DrugKit Docking Pipeline ==="
echo "Engine: ${ENGINE}"
echo "CPUs: ${NCPU}"
echo "Server: $(hostname)"
echo ""

# Check engine is available
if ! command -v "${ENGINE}" &> /dev/null; then
    echo "[WARNING] ${ENGINE} not found in PATH. Checking conda environment..."
    SMINA_PATH="$(conda run -n ${ENV_NAME} which ${ENGINE} 2>/dev/null || true)"
    if [ -z "$SMINA_PATH" ]; then
        echo "[ERROR] ${ENGINE} not available. Install via: conda install -c bioconda ${ENGINE}"
        exit 1
    fi
fi

mkdir -p logs output/docking/poses

echo "Running ${ENGINE} docking pipeline..."
if [ "${ENGINE}" = "smina" ]; then
    python -c "
from docking_smina.docking_smina_r import main
main()
" 2>&1 | tee "logs/docking_${ENGINE}_$(date +%Y%m%d_%H%M%S).log"
else
    python -c "
from docking_vina.docking_vina_r import main
main()
" 2>&1 | tee "logs/docking_${ENGINE}_$(date +%Y%m%d_%H%M%S).log"
fi

echo ""
echo "=== Docking complete ==="
