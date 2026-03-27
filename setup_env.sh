#!/bin/bash
# ==============================================================================
# LittleBit - Environment Setup Script
# ==============================================================================
# Usage: bash setup_env.sh [--cuda-version cu124|cu126]
#
# Prerequisites:
#   - Python 3.10+
#   - NVIDIA GPU with CUDA 12.4+ (check: nvidia-smi)
#   - System MPI library (e.g., sudo apt install libopenmpi-dev)
# ==============================================================================

set -e

# --- Configuration ---
VENV_DIR="venv"
PYTHON=${PYTHON:-python3}
CUDA_VERSION="cu124"  # default: cu124 (compatible with CUDA 12.4~12.6)

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash setup_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda-version VERSION  CUDA version for PyTorch (default: cu124)"
            echo "                          Supported: cu124, cu126"
            echo "  --python PATH           Python executable (default: python3)"
            echo "  --venv-dir DIR          Virtual environment directory (default: venv)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo " LittleBit Environment Setup"
echo "=============================================="
echo " Python:       ${PYTHON}"
echo " CUDA version: ${CUDA_VERSION}"
echo " Venv dir:     ${VENV_DIR}"
echo "=============================================="

# --- Step 0: Check prerequisites ---
echo ""
echo "[0/4] Checking prerequisites..."

if ! command -v ${PYTHON} &> /dev/null; then
    echo "ERROR: ${PYTHON} not found. Please install Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$(${PYTHON} --version 2>&1 | awk '{print $2}')
echo "  Python version: ${PYTHON_VERSION}"

if command -v nvidia-smi &> /dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  NVIDIA driver CUDA: ${DRIVER_CUDA}"
else
    echo "  WARNING: nvidia-smi not found. GPU support may not work."
fi

# Check MPI
if ! command -v mpirun &> /dev/null && ! dpkg -l | grep -q libopenmpi-dev 2>/dev/null; then
    echo "  WARNING: MPI not found. mpi4py installation may fail."
    echo "  Fix: sudo apt install libopenmpi-dev"
fi

# --- Step 1: Create virtual environment ---
echo ""
echo "[1/4] Creating virtual environment in '${VENV_DIR}'..."

if [ -d "${VENV_DIR}" ]; then
    echo "  Virtual environment already exists. Skipping creation."
else
    ${PYTHON} -m venv ${VENV_DIR}
    echo "  Created."
fi

source ${VENV_DIR}/bin/activate
echo "  Activated: $(which python)"

# --- Step 2: Upgrade pip ---
echo ""
echo "[2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- Step 3: Install PyTorch ---
echo ""
echo "[3/4] Installing PyTorch (${CUDA_VERSION})..."

TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_VERSION}"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url ${TORCH_INDEX_URL}

# --- Step 4: Install remaining dependencies ---
echo ""
echo "[4/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Step 5: Install EAGLE as editable package ---
echo ""
echo "[5/5] Installing EAGLE package (editable)..."
if [ -d "EAGLE" ]; then
    pip install -e EAGLE/
    echo "  EAGLE installed."
else
    echo "  EAGLE directory not found. Skipping."
fi

# --- Verification ---
echo ""
echo "=============================================="
echo " Verification"
echo "=============================================="

python -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:  {torch.version.cuda}')
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')

import transformers
print(f'  Transformers:  {transformers.__version__}')

import accelerate
print(f'  Accelerate:    {accelerate.__version__}')

import deepspeed
print(f'  DeepSpeed:     {deepspeed.__version__}')
"

echo ""
echo "=============================================="
echo " Setup complete!"
echo " Activate with: source ${VENV_DIR}/bin/activate"
echo "=============================================="
