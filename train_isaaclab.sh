#!/bin/bash
# Train G1 Motion Imitation with Isaac Lab
#
# Usage:
#   bash train_isaaclab.sh [run_name] [num_envs]
#
# Example:
#   bash train_isaaclab.sh teleop_v1 4096
#
# Prerequisites:
#   - Isaac Lab installed (conda activate isaaclab)
#   - RTX 5090 or compatible GPU

set -e

# Default values
RUN_NAME=${1:-"motion_mimic_$(date +%Y%m%d_%H%M%S)"}
NUM_ENVS=${2:-4096}
MOTION_FILE="motion_data_configs/teleop_dataset.yaml"
MAX_ITERATIONS=20000

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  G1 Motion Imitation Training (Isaac Lab)"
echo "============================================"
echo ""
echo "Run name:       $RUN_NAME"
echo "Num envs:       $NUM_ENVS"
echo "Motion file:    $MOTION_FILE"
echo "Max iterations: $MAX_ITERATIONS"
echo ""

# Check if isaaclab env is active
if [[ "$CONDA_DEFAULT_ENV" != "isaaclab" ]]; then
    echo "Warning: 'isaaclab' conda environment not active"
    echo "Run: conda activate isaaclab"
    echo ""
fi

# Install local packages if needed
pip install -e ./isaaclab_envs --quiet 2>/dev/null || true

# Run training
python scripts/train_isaaclab.py \
    --task Isaac-Motion-Mimic-G1-v0 \
    --motion_file "$MOTION_FILE" \
    --num_envs "$NUM_ENVS" \
    --run_name "$RUN_NAME" \
    --max_iterations "$MAX_ITERATIONS" \
    --headless

echo ""
echo "Training complete! Logs saved to: logs/isaaclab/$RUN_NAME"
