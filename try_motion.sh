#!/bin/bash

# Try different walking motions
# Usage: bash try_motion.sh 005
#
# NOTE: Run this with 'gmr' conda environment (has modern PyTorch with RTX 5090 support)
#       conda activate gmr && bash try_motion.sh 001

script_dir=$(dirname $(realpath $0))
motion_num=$1
device=${2:-cuda}  # Default to cuda (auto-fallback to cpu if GPU incompatible)

if [ -z "$motion_num" ]; then
    echo "Usage: bash try_motion.sh <motion_number> [device]"
    echo "Example: bash try_motion.sh 005"
    echo "Example: bash try_motion.sh 005 cpu"
    echo ""
    echo "Available motions: 001, 002, 003, 004, 005, 006, 007, 008, 009, 010"
    echo ""
    echo "NOTE: Use 'conda activate gmr' before running (for RTX 5090 CUDA support)"
    exit 1
fi

motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_${motion_num}.pkl"

if [ ! -f "$motion_file" ]; then
    echo "Error: Motion file not found: $motion_file"
    exit 1
fi

echo "Playing motion: $motion_file"
echo "Device: $device"

cd deploy_real

# Add pose module to Python path
export PYTHONPATH="${script_dir}/pose:${PYTHONPATH}"

redis_ip="localhost"

python server_motion_lib.py \
    --motion_file ${motion_file} \
    --robot unitree_g1_with_hands \
    --vis \
    --device ${device} \
    --redis_ip ${redis_ip}

