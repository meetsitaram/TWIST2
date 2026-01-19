#!/usr/bin/env python3
"""
Interactive Camera Setup Tool

This script helps you:
1. Detect available cameras
2. Capture test images from each
3. Assign camera positions (left, center, right)
4. Generate camera_config.yaml

Usage:
    conda activate gmr
    python setup_cameras.py
"""

import cv2
import os
import yaml
from datetime import datetime


def detect_cameras(max_cameras=10):
    """Detect all available cameras."""
    print("Detecting cameras...")
    available = []
    
    for cam_id in range(max_cameras):
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                available.append({'id': cam_id, 'resolution': f"{w}x{h}"})
            cap.release()
    
    return available


def test_camera_fps(cam_id, duration=2):
    """Test camera FPS with MJPEG mode."""
    import time
    
    cap = cv2.VideoCapture(cam_id)
    # MJPEG mode: ~26 FPS (vs ~10 FPS with raw YUYV) due to lower USB bandwidth
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Warm up
    for _ in range(5):
        cap.read()
    
    start = time.time()
    frames = 0
    while time.time() - start < duration:
        ret, _ = cap.read()
        if ret:
            frames += 1
    
    fps = frames / duration
    cap.release()
    return fps


def capture_test_images(cameras, output_dir):
    """Capture test image from each camera."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCapturing test images to: {output_dir}")
    
    for cam in cameras:
        cam_id = cam['id']
        cap = cv2.VideoCapture(cam_id)
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Warm up
        for _ in range(10):
            cap.read()
        
        ret, frame = cap.read()
        if ret:
            # Add label
            cv2.putText(frame, f"Camera {cam_id}", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            output_path = os.path.join(output_dir, f"camera_{cam_id}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"  Camera {cam_id}: {output_path}")
        
        cap.release()
    
    return output_dir


def generate_config(camera_mapping, output_path):
    """Generate camera_config.yaml file."""
    
    config = {
        'capture': {
            'resolution': [1280, 720],
            'fps': 30,
            'codec': 'MJPG'
        },
        'cameras': {},
        'camera_ids': [],
        'calibration': {
            'board': {
                'squares_x': 9,
                'squares_y': 6,
                'square_size_mm': 30.0,
                'marker_size_mm': 22.0,
                'dictionary': 'DICT_4X4_50'
            },
            'thresholds': {
                'max_reprojection_error': 1.0,
                'min_detected_frames': 50
            }
        }
    }
    
    # Add camera mappings
    for position, cam_id in camera_mapping.items():
        config['cameras'][position] = {
            'id': cam_id,
            'description': f"{position.capitalize()} camera"
        }
    
    # Ordered list: left, center, right
    order = ['left', 'center', 'right']
    config['camera_ids'] = [camera_mapping[pos] for pos in order if pos in camera_mapping]
    
    # Write YAML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Camera Configuration for Multi-Camera Motion Capture\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        f.write("# Regenerate with: python deploy_real/setup_cameras.py\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nConfiguration saved to: {output_path}")
    return config


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up from utils/ to TWIST2/
    calibration_dir = os.path.join(project_dir, "calibration")
    
    print("=" * 60)
    print("  Camera Setup Tool")
    print("=" * 60)
    
    # Step 1: Detect cameras
    cameras = detect_cameras()
    
    if not cameras:
        print("\nNo cameras detected!")
        print("Make sure cameras are connected and not in use by another application.")
        return
    
    print(f"\nFound {len(cameras)} camera(s):")
    for cam in cameras:
        fps = test_camera_fps(cam['id'])
        print(f"  Camera {cam['id']}: {cam['resolution']} @ {fps:.1f} FPS (MJPEG 720p)")
    
    # Step 2: Capture test images
    views_dir = os.path.join(calibration_dir, "camera_views")
    capture_test_images(cameras, views_dir)
    
    print(f"\nTest images saved. Please view them to identify camera positions:")
    print(f"  xdg-open {views_dir}")
    
    # Step 3: Get user input for camera positions
    print("\n" + "-" * 60)
    print("Assign camera positions (enter camera ID for each position)")
    print("Press Enter to skip a position if you have fewer than 3 cameras")
    print("-" * 60)
    
    camera_ids = [c['id'] for c in cameras]
    camera_mapping = {}
    
    for position in ['left', 'center', 'right']:
        while True:
            try:
                response = input(f"  {position.capitalize()} camera ID [{camera_ids}]: ").strip()
                
                if response == "":
                    print(f"    Skipping {position}")
                    break
                
                cam_id = int(response)
                if cam_id in camera_ids:
                    camera_mapping[position] = cam_id
                    print(f"    {position.capitalize()} = Camera {cam_id}")
                    break
                else:
                    print(f"    Invalid ID. Choose from: {camera_ids}")
            except ValueError:
                print(f"    Please enter a number from: {camera_ids}")
    
    if len(camera_mapping) < 2:
        print("\nWarning: At least 2 cameras are needed for triangulation.")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Setup cancelled.")
            return
    
    # Step 4: Generate config
    config_path = os.path.join(calibration_dir, "camera_config.yaml")
    generate_config(camera_mapping, config_path)
    
    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print(f"\nCamera configuration:")
    for position, cam_id in camera_mapping.items():
        print(f"  {position.capitalize()}: Camera {cam_id}")
    print(f"\nConfig file: {config_path}")
    print("\nNext step: Run camera calibration")
    print("  python deploy_real/calibrate_cameras.py")


if __name__ == "__main__":
    main()
