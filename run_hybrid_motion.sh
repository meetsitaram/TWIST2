#!/bin/bash

# Run hybrid motion: PKL body + camera arms
#
# Usage: 
#   bash run_hybrid_motion.sh 001          # Use walking motion 001
#   bash run_hybrid_motion.sh 005          # Use walking motion 005
#   bash run_hybrid_motion.sh /path/to.pkl # Use custom motion file
#
# NOTE: Run with 'gmr' conda environment for camera CUDA support
#       conda activate gmr && bash run_hybrid_motion.sh 001

script_dir=$(dirname $(realpath $0))
motion_input=$1

if [ -z "$motion_input" ]; then
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║  Hybrid Motion: PKL Body + Camera Arms                ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: bash run_hybrid_motion.sh <motion_number|path>"
    echo ""
    echo "Examples:"
    echo "  bash run_hybrid_motion.sh 001     # Use walking motion 001"
    echo "  bash run_hybrid_motion.sh 005     # Use walking motion 005"
    echo ""
    echo "Available walking motions: 001, 002, 003, 004, 005, 006, 007, 008, 009, 010"
    echo ""
    echo "Prerequisites:"
    echo "  1. Terminal 1: conda activate twist2 && bash sim2sim.sh"
    echo "  2. Terminal 2: conda activate gmr && bash run_hybrid_motion.sh 001"
    echo ""
    exit 1
fi

# Check if input is a number (walking motion) or a path
if [[ "$motion_input" =~ ^[0-9]+$ ]]; then
    # Pad to 3 digits
    motion_num=$(printf "%03d" $motion_input)
    motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_${motion_num}.pkl"
else
    motion_file="$motion_input"
fi

if [ ! -f "$motion_file" ]; then
    echo "Error: Motion file not found: $motion_file"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Hybrid Motion: PKL Body + Camera Arms                ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "Motion file: $motion_file"
echo "Cameras: 4, 6, 2"
echo ""

cd "${script_dir}/deploy_real"

python multicam_with_motion.py \
    --motion-file "${motion_file}" \
    --camera-ids 4,6,2 \
    --calibration ../calibration/calibration.toml \
    --target-fps 50

