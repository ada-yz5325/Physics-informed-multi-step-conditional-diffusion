#!/usr/bin/env bash
set -e

# Portable environment helper for the public repository.
# Create the environment first with:
#   conda env create -f environment.yml
# Then activate it with:
#   conda activate ACDM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "DATA_ROOT:    $DATA_ROOT"
echo "Using python: $(which python)"
python --version
