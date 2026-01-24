#!/usr/bin/env python3
"""
Annotate Charuco Board with Measurement Guidelines

This script overlays measurement annotations on a Charuco board image
to show where to measure square size and marker size.

Usage:
    conda activate gmr
    python annotate_charuco_board.py [--input BOARD_IMAGE] [--output OUTPUT_IMAGE]
"""

import cv2
import numpy as np
import os
import argparse
import yaml


def load_config(config_path):
    """Load board settings from camera_config.yaml"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('calibration', {}).get('board', {})
    return {}


def annotate_charuco_board(input_path, output_path, squares_x=5, squares_y=3):
    """Add measurement annotations to a Charuco board image."""
    
    # Load the charuco board
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Calculate approximate square positions (board has margins)
    margin = 75  # approximate margin used in generation
    board_w = w - 2 * margin
    board_h = h - 2 * margin
    
    # Square size in pixels
    square_w = board_w // squares_x
    square_h = board_h // squares_y
    
    # Draw annotation for square size measurement
    # Draw on first black square (top-left)
    x1 = margin
    y1 = margin
    x2 = margin + square_w
    y2 = margin + square_h
    
    # Draw red rectangle around first square
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 8)
    
    # Draw horizontal measurement line
    line_y = y1 + square_h // 2
    cv2.line(img, (x1, line_y), (x2, line_y), (0, 0, 255), 6)
    cv2.line(img, (x1, line_y - 30), (x1, line_y + 30), (0, 0, 255), 6)  # Left cap
    cv2.line(img, (x2, line_y - 30), (x2, line_y + 30), (0, 0, 255), 6)  # Right cap
    
    # Add label for square size
    cv2.putText(img, "SQUARE SIZE", (x1 + 20, line_y - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.putText(img, "(measure this!)", (x1 + 20, line_y + 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # Draw annotation for marker size (inside a white square with ArUco)
    # Second square (white with marker)
    mx1 = margin + square_w
    my1 = margin
    mx2 = margin + 2 * square_w
    my2 = margin + square_h
    
    # Approximate marker position (centered in square, ~75% of square size)
    marker_margin = int(square_w * 0.125)  # ~12.5% margin on each side
    mmx1 = mx1 + marker_margin
    mmy1 = my1 + marker_margin
    mmx2 = mx2 - marker_margin
    mmy2 = my2 - marker_margin
    
    # Draw blue rectangle around marker
    cv2.rectangle(img, (mmx1, mmy1), (mmx2, mmy2), (255, 100, 0), 6)
    
    # Draw measurement line for marker
    marker_line_y = mmy1 + (mmy2 - mmy1) // 2
    cv2.line(img, (mmx1, marker_line_y), (mmx2, marker_line_y), (255, 100, 0), 4)
    cv2.line(img, (mmx1, marker_line_y - 20), (mmx1, marker_line_y + 20), (255, 100, 0), 4)
    cv2.line(img, (mmx2, marker_line_y - 20), (mmx2, marker_line_y + 20), (255, 100, 0), 4)
    
    # Add label for marker size
    cv2.putText(img, "MARKER SIZE", (mmx1 - 30, mmy1 - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 0), 3)
    cv2.putText(img, "(optional)", (mmx1 - 30, mmy2 + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 3)
    
    # Add legend at bottom
    legend_y = h - 150
    cv2.putText(img, "RED = Square size (measure this!)", (50, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(img, "BLUE = Marker size (optional, ~75% of square)", (50, legend_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)
    
    # Save annotated image
    cv2.imwrite(output_path, img)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Annotate Charuco board with measurement guides")
    parser.add_argument("--input", type=str, default=None,
                       help="Input Charuco board image")
    parser.add_argument("--output", type=str, default=None,
                       help="Output annotated image path")
    parser.add_argument("--squares-x", type=int, default=None,
                       help="Number of squares in X direction")
    parser.add_argument("--squares-y", type=int, default=None,
                       help="Number of squares in Y direction")
    
    args = parser.parse_args()
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    calibration_dir = os.path.join(project_dir, "calibration")
    charuco_dir = os.path.join(calibration_dir, "charuco_boards")
    config_path = os.path.join(calibration_dir, "camera_config.yaml")
    
    # Load config
    board_config = load_config(config_path)
    squares_x = args.squares_x or board_config.get('squares_x', 5)
    squares_y = args.squares_y or board_config.get('squares_y', 3)
    
    # Find input image
    if args.input:
        input_path = args.input
    else:
        # Try to find a charuco board image in charuco_boards subdirectory
        for name in ["charuco_board_5x3_A4.png", "charuco_board_A3.png", "charuco_board.png"]:
            path = os.path.join(charuco_dir, name)
            if os.path.exists(path):
                input_path = path
                break
        else:
            print("Error: No Charuco board image found. Generate one first:")
            print("  python deploy_real/generate_charuco_board.py")
            return
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_annotated{ext}"
    
    print(f"Annotating: {input_path}")
    print(f"Board size: {squares_x}x{squares_y}")
    
    result = annotate_charuco_board(input_path, output_path, squares_x, squares_y)
    
    if result:
        print(f"Annotated board saved to: {output_path}")
        print(f"\nView with: xdg-open {output_path}")


if __name__ == "__main__":
    main()
