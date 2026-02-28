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
import time
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
        
        # Warm up: discard frames while auto-exposure settles
        time.sleep(1.0)
        for _ in range(30):
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


def load_camera_ids_from_config():
    """Read camera IDs from calibration/camera_config.yaml."""
    import yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    config_path = os.path.join(project_dir, "calibration", "camera_config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ids = cfg.get("camera_ids", [])
        if ids:
            return ids
    return None


def main():
    parser = argparse.ArgumentParser(description="Capture single frame from each camera")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                       help="Camera IDs to capture from (default: read from camera_config.yaml)")
    parser.add_argument("--output", type=str, 
                       default="../calibration/camera_views",
                       help="Output directory for images")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720],
                       help="Resolution WxH (default: 1280 720)")
    
    args = parser.parse_args()
    
    camera_ids = args.cameras or load_camera_ids_from_config() or [0, 1, 2]
    print(f"Using camera IDs: {camera_ids}")
    
    # Get absolute path for output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up from utils/ to TWIST2/
    output_dir = os.path.join(project_dir, "calibration", "camera_views")
    
    capture_views(camera_ids, output_dir, tuple(args.resolution))
    
    print("\nTo view images:")
    print(f"  xdg-open {output_dir}")


if __name__ == "__main__":
    main()
