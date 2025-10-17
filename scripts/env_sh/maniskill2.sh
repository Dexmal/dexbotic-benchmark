#!/bin/bash

# ManiSkill2 evaluation script
# Usage: bash scripts/env_sh/maniskill2.sh [config_file]

conda init && source activate
conda activate maniskill2_env

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default configuration file
DEFAULT_CONFIG="${PROJECT_ROOT}/evaluation/configs/maniskill2/example_maniskill2.yaml"

# Parse arguments
CONFIG_FILE="${1:-${DEFAULT_CONFIG}}"

# Validate configuration file
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Configuration file not found: ${CONFIG_FILE}"
    echo "Available configuration files:"
    find "${PROJECT_ROOT}/evaluation/configs/maniskill2" -name "*.yaml" -type f | sed 's|^|  |'
    exit 1
fi

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}/evaluation/evaluator:${PROJECT_ROOT}:${PYTHONPATH}"

# Run ManiSkill2 evaluation
echo "Starting ManiSkill2 evaluation with config: ${CONFIG_FILE}"
python3 "${PROJECT_ROOT}/evaluation/run_maniskill2_evaluation.py" --config "${CONFIG_FILE}"

echo "ManiSkill2 evaluation completed!"