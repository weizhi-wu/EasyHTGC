#!/bin/bash
#
# ==================================================================================
#   Extensive Shell Wrapper with Data Staging for HTCondor
# ==================================================================================
#
# PURPOSE:
#   - Sets up the Conda environment.
#   - Stages compressed data to the node's fast local /tmp disk for high-speed I/O.
#   - Executes the main Python script, passing it the path to the staged data.
#   - Guarantees cleanup of the staged data, even if the job fails.
#

# --- 1. Script Configuration & Error Handling ---

# Exit immediately if any command fails. This is crucial for automation.
set -e
# Fail a pipeline if any command within it fails.
set -o pipefail

# --- 2. User-Defined Variables ---

# The name of your specific Miniconda environment.
CONDA_ENV_NAME="deepzero-env"

# The name of your main Python executable script.
PYTHON_EXECUTABLE="main.py"

# The name of the compressed data archive you are transferring.
# This MUST match the filename in your `transfer_input_files` list.
DATA_ARCHIVE="dataset.tar.gz"

# The base path to your Miniconda installation in your home directory.
CONDA_BASE_PATH="$HOME/miniconda3"

# --- 3. Argument Handling ---

# Check for the experiment ID passed from the .dag file.
if [ -z "$1" ]; then
    echo "ERROR: No experiment ID provided. This script requires one argument." >&2
    exit 1
fi
EXP_ID=$1

# --- 4. Data Staging to Local Scratch Disk (/tmp) ---

echo "=========================================================="
echo "Job starting on $(hostname) at $(date)"
echo "--- Staging data to local scratch disk ---"

# Create a unique directory path in /tmp. Using the username and experiment ID
# prevents collisions with other users' jobs.
SCRATCH_DIR="/tmp/${USER}_${EXP_ID}"
mkdir -p "${SCRATCH_DIR}"
echo "Scratch directory created at: ${SCRATCH_DIR}"

# --- THIS IS THE CRITICAL CLEANUP COMMAND ---
# `trap` registers a command to be run when the script receives a signal.
# `EXIT` means the command will run no matter how the script terminates:
# on success, on failure (due to `set -e`), or if cancelled by HTCondor.
# This ensures we NEVER leave leftover data on the node's scratch disk.
trap 'echo "--- Cleaning up scratch directory ---"; rm -rf "${SCRATCH_DIR}"' EXIT

# Unpack the dataset archive into our new scratch directory.
# The `-C` flag tells `tar` to change to that directory before extracting.
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
# Now, we run the Python script. We pass it TWO arguments:
# 1. The original experiment ID.
# 2. A named argument `--data-path` with the path to our high-speed local data.

echo "--- Starting Python Execution ---"
exec python "${PYTHON_EXECUTABLE}" "${EXP_ID}" --data-path "${SCRATCH_DIR}"

# The script exits here. The `trap` command will automatically run on exit.
