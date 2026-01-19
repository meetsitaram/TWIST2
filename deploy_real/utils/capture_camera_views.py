#!/usr/bin/env python3
"""
Capture single frame from each camera for viewing/debugging.

Usage:
    conda activate gmr
    python capture_camera_views.py

This helps identify camera positions (left, center, right) and verify
that all cameras are working correctly.
"""

import cv2
import os
import argparse
from datetime import datetime


def capture_views(camera_ids, output_dir, resolution=(1280, 720)):
    """Capture one frame from each camera and save to disk."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Capturing one frame from each camera...")
    print("=" * 50)
    
    captured = []
    
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            print(f"Camera {cam_id}: FAILED to open")
            continue
        
        # Set MJPEG mode for better FPS
        # MJPEG uses compression, reducing USB bandwidth and increasing FPS
        # Without this: ~10 FPS per camera (raw YUYV)
        # With this: ~26 FPS per camera (compressed MJPEG)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Warm up (discard first few frames)
        for _ in range(10):
            cap.read()
        
        # Capture frame
        ret, frame = cap.read()
        
        if ret:
            # Add camera ID label to image
            cv2.putText(frame, f"Camera {cam_id}", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (50, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save image
            output_path = os.path.join(output_dir, f"camera_{cam_id}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Camera {cam_id}: Saved to {output_path}")
            captured.append(cam_id)
        else:
            print(f"Camera {cam_id}: Failed to capture frame")
        
        cap.release()
    
    print("=" * 50)
    print(f"\nCaptured {len(captured)} images to: {output_dir}")
    
    return captured


def main():
    parser = argparse.ArgumentParser(description="Capture single frame from each camera")
    parser.add_argument("--cameras", type=int, nargs="+", default=[2, 4, 6],
                       help="Camera IDs to capture from (default: 2 4 6)")
    parser.add_argument("--output", type=str, 
                       default="../calibration/camera_views",
                       help="Output directory for images")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720],
                       help="Resolution WxH (default: 1280 720)")
    
    args = parser.parse_args()
    
    # Get absolute path for output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up from utils/ to TWIST2/
    output_dir = os.path.join(project_dir, "calibration", "camera_views")
    
    capture_views(args.cameras, output_dir, tuple(args.resolution))
    
    print("\nTo view images:")
    print(f"  xdg-open {output_dir}")


if __name__ == "__main__":
    main()
