#!/bin/bash
# Setup script for gmr conda environment
# Used for: Multi-camera capture, SMPL-X fitting, GMR retargeting
#
# Usage:
#   bash setup_gmr_env.sh
#
# This script will:
# 1. Create gmr conda environment with Python 3.10
# 2. Install all pip dependencies from requirements_gmr.txt
# 3. Install conda-forge dependencies
# 4. Clone and install GMR (if not already present)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Setting up gmr conda environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Create environment if it doesn't exist
if conda env list | grep -q "^gmr "; then
    echo "gmr environment already exists"
else
    echo "Creating gmr environment with Python 3.10..."
    conda create -n gmr python=3.10 -y
fi

# Activate environment
echo "Activating gmr environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gmr

# Install pip dependencies
echo "Installing pip dependencies..."
pip install -r "$SCRIPT_DIR/requirements_gmr.txt"

# Install conda-forge dependencies
echo "Installing conda-forge dependencies..."
conda install -c conda-forge libstdcxx-ng -y

# Clone and install GMR if not present
GMR_DIR="$PROJECT_DIR/GMR"
if [ -d "$GMR_DIR" ]; then
    echo "GMR already cloned at $GMR_DIR"
else
    echo "Cloning GMR..."
    cd "$PROJECT_DIR"
    git clone https://github.com/YanjieZe/GMR.git
fi

# Install GMR
echo "Installing GMR..."
cd "$GMR_DIR"
pip install -e .

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate gmr"
echo ""
echo "To verify installation:"
echo "  python -c \"import cv2; import mediapipe; import numpy; print('Core deps OK')\""
echo ""
