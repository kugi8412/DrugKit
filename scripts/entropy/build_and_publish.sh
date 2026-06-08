#!/bin/bash
# scripts/entropy/build_and_publish.sh
# Build and publish DrugKit to TestPyPI
# Usage: bash scripts/entropy/build_and_publish.sh [--publish]
#
# Prerequisites:
#   - pip install build twine
#   - TestPyPI token configured in ~/.pypirc

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

echo "=== DrugKit Build & Publish ==="

# Install build tools
pip install --quiet build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build
echo "Building package..."
python -m build

echo ""
echo "Built artifacts:"
ls -la dist/

# Verify
echo ""
echo "Checking package..."
twine check dist/*

# Publish to TestPyPI if requested
PUBLISH=false
for arg in "$@"; do
    case $arg in
        --publish) PUBLISH=true ;;
    esac
done

if [ "$PUBLISH" = true ]; then
    echo ""
    echo "Publishing to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo ""
    echo "Package published! Install with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ drugkit"
else
    echo ""
    echo "Dry run complete. To publish, run:"
    echo "  bash scripts/entropy/build_and_publish.sh --publish"
    echo ""
    echo "First, configure ~/.pypirc with your TestPyPI token:"
    echo "  [testpypi]"
    echo "  repository = https://test.pypi.org/legacy/"
    echo "  username = __token__"
    echo "  password = pypi-..."
fi
