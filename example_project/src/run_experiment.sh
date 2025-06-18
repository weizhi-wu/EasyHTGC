#!/bin/bash
# --- START OF FILE src/run_experiment.sh ---
#
# ==================================================================================
#   Shell Wrapper with Data Staging for HTCondor
# ==================================================================================

# --- 1. Script Configuration & Error Handling ---
set -e
set -o pipefail

# --- 2. User-Defined Variables ---
# IMPORTANT: This should match the name of the Conda environment you created.
# The SimpleHTGC server config will eventually control this.
CONDA_ENV_NAME="deepzero-env"
PYTHON_EXECUTABLE="main.py"
DATA_ARCHIVE="dataset.tar.gz"
CONDA_BASE_PATH="$HOME/miniconda3"

# --- 3. Argument Handling ---
if [ -z "$1" ]; then
    echo "ERROR: No experiment ID provided." >&2
    exit 1
fi
EXP_ID=$1

# --- 4. Data Staging to Local Scratch Disk (/tmp) ---
echo "=========================================================="
echo "Job starting on $(hostname) at $(date)"
echo "--- Staging data to local scratch disk ---"

SCRATCH_DIR="/tmp/${USER}_${EXP_ID}"
mkdir -p "${SCRATCH_DIR}"
echo "Scratch directory created at: ${SCRATCH_DIR}"

# CRITICAL: Guarantees cleanup of the scratch directory on any exit.
trap 'echo "--- Cleaning up scratch directory ---"; rm -rf "${SCRATCH_DIR}"' EXIT

# Unpack the dataset archive into the scratch directory.
tar -xzf "${DATA_ARCHIVE}" -C "${SCRATCH_DIR}"
echo "--- Dataset staged and cleanup trap registered ---"

# --- 5. Environment Setup ---
echo "--- Activating Conda Environment: ${CONDA_ENV_NAME} ---"
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
echo "--- Environment Activated ---"
echo "Experiment ID for this run: ${EXP_ID}"
echo "Data path for this run: ${SCRATCH_DIR}"
echo "=========================================================="

# --- 6. Main Execution ---
# Use `exec` to replace the shell process with the Python process.
exec python "${PYTHON_EXECUTABLE}" "${EXP_ID}" --data-path "${SCRATCH_DIR}"

# --- END OF FILE src/run_experiment.sh ---
