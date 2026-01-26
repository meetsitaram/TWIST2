#!/bin/bash

# Train policy on custom teleop motion data
#
# Usage: bash train_teleop.sh <experiment_id> <device>
#
# Example:
#   bash train_teleop.sh teleop_001 cuda:0
#   bash train_teleop.sh teleop_debug cuda:0 --debug
#
# Requirements:
#   - conda activate twist2
#   - Teleop motions converted: datasets/teleop_motions/*.pkl
#   - Config: motion_data_configs/teleop_dataset.yaml

set -e

SCRIPT_DIR=$(dirname $(realpath $0))
cd "${SCRIPT_DIR}/legged_gym/legged_gym/scripts"

# Parse arguments
exptid=${1:-"teleop_$(date +%m%d_%H%M)"}
device=${2:-"cuda:0"}
extra_args="${@:3}"

# Task configuration
task_name="g1_teleop"
proj_name="g1_teleop"

echo "=============================================="
echo "  TWIST2 Teleop Training"
echo "=============================================="
echo "  Task:       ${task_name}"
echo "  Experiment: ${exptid}"
echo "  Device:     ${device}"
echo "  Project:    ${proj_name}"
echo "  Extra args: ${extra_args}"
echo "=============================================="
echo ""

# Check if motion data exists
MOTION_CONFIG="${SCRIPT_DIR}/motion_data_configs/teleop_dataset.yaml"
if [ ! -f "${MOTION_CONFIG}" ]; then
    echo "ERROR: Motion config not found: ${MOTION_CONFIG}"
    echo "Run: cd deploy_real && python convert_episodes_to_motion.py --all --yaml teleop_dataset"
    exit 1
fi

echo "Using motion config: ${MOTION_CONFIG}"
echo ""

# Run training
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --no_wandb \
                ${extra_args}

echo ""
echo "Training complete!"
echo "Logs saved to: ${SCRIPT_DIR}/logs/${proj_name}/${exptid}"
