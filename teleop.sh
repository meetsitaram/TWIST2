#!/bin/bash

# Teleoperation with Calibrated Joints
# Easy launch script for right hand teleoperation

set -e

cd "$(dirname "$0")/deploy_real"

# Parse arguments
CAMERA_ID=${1:-2}  # Default to camera 2
MODE=${2:-right_hand_only}  # Default to right hand only
CONFIG=${3:-../joint_mapping_config.json}  # Default config

echo "=================================================="
echo "🤖 TWIST2 TELEOPERATION (CALIBRATED)"
echo "=================================================="
echo "Camera ID: $CAMERA_ID"
echo "Mode: $MODE"
echo "Config: $CONFIG"
echo ""

# Check if config exists
if [ -f "$CONFIG" ]; then
    echo "✅ Calibration config found: $CONFIG"
else
    echo "⚠️  Calibration config not found: $CONFIG"
    echo "   Using default (uncalibrated) mappings"
    echo ""
    echo "   To create calibration:"
    echo "   cd .."
    echo "   ./auto_calibrate.sh $CAMERA_ID $MODE 60"
    echo ""
fi

echo "=================================================="
echo ""
echo "📋 Instructions:"
echo "  1. Make sure sim2sim.sh is running in another terminal"
echo "  2. Stand in front of camera"
echo "  3. Get ready to match robot's standing pose"
echo "  4. System will count down from 5"
echo "  5. Move your right arm naturally!"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate twist2

# Run teleoperation
python freemocap_to_twist2.py \
    --camera-id "$CAMERA_ID" \
    --tracking-mode "$MODE" \
    --joint-config "$CONFIG" \
    --startup-delay 5 \
    --flip-horizontal

echo ""
echo "Teleoperation ended."
