#!/bin/bash
# scripts/entropy/submit_slurm.sh
# Submit a SLURM job on MIMUW entropy cluster
# Usage: bash scripts/entropy/submit_slurm.sh [test|al|docking]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_TYPE="${1:-test}"

case "$JOB_TYPE" in
    test)
        JOB_NAME="drugkit-tests"
        SCRIPT="${SCRIPT_DIR}/run_tests.sh"
        TIME="00:30:00"
        MEM="8G"
        CPUS=4
        GPU=""
        ;;
    al)
        JOB_NAME="drugkit-al"
        SCRIPT="${SCRIPT_DIR}/run_active_learning.sh --gpu"
        TIME="04:00:00"
        MEM="16G"
        CPUS=8
        GPU="#SBATCH --gres=gpu:1"
        ;;
    docking)
        JOB_NAME="drugkit-docking"
        SCRIPT="${SCRIPT_DIR}/run_docking.sh smina"
        TIME="08:00:00"
        MEM="32G"
        CPUS=16
        GPU=""
        ;;
    *)
        echo "Usage: $0 [test|al|docking]"
        exit 1
        ;;
esac

SLURM_SCRIPT=$(mktemp /tmp/drugkit_slurm_XXXXXX.sh)

cat > "${SLURM_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
${GPU}
#SBATCH --partition=common

cd ${PROJECT_ROOT}
mkdir -p logs

echo "=== SLURM Job: ${JOB_NAME} ==="
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node: \$(hostname)"
echo "Start: \$(date)"
echo ""

bash ${SCRIPT}

echo ""
echo "End: \$(date)"
EOF

echo "Submitting SLURM job: ${JOB_NAME}"
echo "  Script: ${SLURM_SCRIPT}"
echo "  Time: ${TIME}, CPUs: ${CPUS}, Mem: ${MEM}"

if command -v sbatch &> /dev/null; then
    sbatch "${SLURM_SCRIPT}"
else
    echo "[INFO] sbatch not available (not on cluster head node)."
    echo "Generated script at: ${SLURM_SCRIPT}"
    echo "Copy to entropy and run: sbatch ${SLURM_SCRIPT}"
fi
