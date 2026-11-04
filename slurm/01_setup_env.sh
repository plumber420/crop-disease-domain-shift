#!/usr/bin/env bash
# ==============================================================================
# 01_setup_env.sh
# Environment Setup & Virtualenv Installation Script for KU Leuven VSC (wICE)
#
# Usage:
#   bash slurm/01_setup_env.sh
# ==============================================================================

set -eo pipefail

echo "======================================================================"
echo " [KU Leuven VSC] Initializing Deep Learning Python Environment"
echo "======================================================================"

# 1. Purge any conflicting user modules
module purge

# 2. Load cluster architecture and toolchains
# KU Leuven Tier-2 wICE cluster uses Lmod modules
if module avail cluster/wICE 2>&1 | grep -q "cluster/wICE"; then
    module load cluster/wICE
    echo "[INFO] Loaded cluster/wICE architecture."
fi

# Load compatible Python and CUDA modules
echo "[INFO] Loading compiler and Python toolchain modules..."
# Standard module configurations on VSC
if module avail Python/3.10.8-GCCcore-12.2.0 2>&1 | grep -q "Python/3.10.8"; then
    module load Python/3.10.8-GCCcore-12.2.0
    module load CUDA/12.1.1 || module load CUDA/11.8.0 || true
elif module avail Python/3.11 2>&1 | grep -q "Python/3.11"; then
    module load Python/3.11*
    module load CUDA/12* || true
else
    # Fallback to general Python and CUDA
    module load Python
    module load CUDA || true
fi

echo "[INFO] Active Modules:"
module list

# 3. Define Virtual Environment Location in $VSC_DATA (Persistent Storage)
# IMPORTANT: Never install heavy virtualenvs or PyTorch wheels into $VSC_HOME
# because of the strict 10GB inode and quota limit on VSC_HOME.
VENV_DIR="${VSC_DATA}/venvs/dl_plants_env"

echo "[INFO] Virtual Environment Target: ${VENV_DIR}"
mkdir -p "${VSC_DATA}/venvs"

if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating new virtualenv with --system-site-packages..."
    python3 -m venv "${VENV_DIR}" --system-site-packages
else
    echo "[INFO] Existing virtualenv found. Updating dependencies..."
fi

# 4. Activate Virtual Environment
source "${VENV_DIR}/bin/activate"

# 5. Upgrade pip, wheel, setuptools
pip install --upgrade pip setuptools wheel

# 6. Install PyTorch with CUDA acceleration (matching CUDA driver on VSC)
echo "[INFO] Installing PyTorch with CUDA 12.1 acceleration..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 7. Install Project Requirements
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[INFO] Installing project requirements from ${REPO_ROOT}/requirements.txt..."
pip install -r "${REPO_ROOT}/requirements.txt"

# 8. Verification and Hardware Sanity Check
echo "======================================================================"
echo " ENVIRONMENT SANITY CHECK"
echo "======================================================================"
python3 -c "
import sys
import torch
import torchvision
import timm
import sklearn

print(f'Python Executable: {sys.executable}')
print(f'Python Version:    {sys.version.split()[0]}')
print(f'PyTorch Version:   {torch.__version__}')
print(f'Torchvision:       {torchvision.__version__}')
print(f'timm Version:      {timm.__version__}')
print(f'Scikit-learn:      {sklearn.__version__}')
print(f'CUDA Available:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device Count: {torch.cuda.device_count()}')
    print(f'Device 0 Name:     {torch.cuda.get_device_name(0)}')
"
echo "======================================================================"
echo "[SUCCESS] Environment is ready for SLURM job submission!"
echo "To activate manually in an interactive shell: source ${VENV_DIR}/bin/activate"
echo "======================================================================"
