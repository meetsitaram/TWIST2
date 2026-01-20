#!/usr/bin/env python3
"""
Debug script for multi-camera pose streaming.
Tests each component step by step.
"""

import numpy as np
import cv2
import time
import sys

def test_cameras(camera_ids):
    """Test if cameras are accessible."""
    print("\n=== Testing Cameras ===")
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera {cam_id}: OK ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"  Camera {cam_id}: OPENED but can't read frame")
            cap.release()
        else:
            print(f"  Camera {cam_id}: FAILED to open")
    return True

def test_calibration(calibration_file):
    """Test if calibration file loads correctly."""
    print("\n=== Testing Calibration ===")
    import toml
    
    try:
        with open(calibration_file, 'r') as f:
            data = toml.load(f)
        
        cameras = data.get('cameras', {})
        print(f"  Calibration file: {calibration_file}")
        print(f"  Number of cameras: {len(cameras)}")
        
        for cam_id, cam_data in cameras.items():
            matrix = np.array(cam_data['matrix'])
            translation = np.array(cam_data['translation'])
            print(f"  Camera {cam_id}:")
            print(f"    Matrix: {matrix.shape}")
            print(f"    Translation: {translation}")
        
        return cameras
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def test_mediapipe():
    """Test if MediaPipe is working."""
    print("\n=== Testing MediaPipe ===")
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        print(f"  MediaPipe version: {mp.__version__}")
        print(f"  Holistic model: OK")
        holistic.close()
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_detection(camera_ids):
    """Test MediaPipe detection on live cameras."""
    print("\n=== Testing Live Detection ===")
    import mediapipe as mp
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"  Camera {cam_id}: FAILED to open")
            continue
        
        # Try 10 frames
        detected = 0
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                if results.pose_landmarks:
                    detected += 1
            time.sleep(0.1)
        
        cap.release()
        print(f"  Camera {cam_id}: Detected pose in {detected}/10 frames")
    
    holistic.close()
    return True

def test_triangulation(camera_ids, calibration_file):
    """Test triangulation with live data."""
    print("\n=== Testing Triangulation ===")
    
    from multicam_pose_streamer import MultiCamPoseStreamer
    
    streamer = MultiCamPoseStreamer(
        camera_ids=camera_ids,
        calibration_file=calibration_file,
        resolution=(1280, 720),
        enable_display=False
    )
    streamer.start()
    
    print("  Running for 5 seconds...")
    time.sleep(2)  # Let cameras warm up
    
    valid_frames = 0
    total_frames = 0
    reproj_errors = []
    
    for _ in range(50):  # 5 seconds at ~10Hz
        skeleton_3d, reproj_error = streamer.get_3d_skeleton()
        total_frames += 1
        
        if skeleton_3d is not None:
            # Check for NaN
            nan_count = np.isnan(skeleton_3d).sum()
            valid_count = 33 - (nan_count // 3)
            
            if nan_count == 0:
                valid_frames += 1
                reproj_errors.append(reproj_error)
            
            if total_frames % 10 == 0:
                print(f"    Frame {total_frames}: {valid_count}/33 landmarks valid, reproj={reproj_error:.1f}px")
        else:
            if total_frames % 10 == 0:
                print(f"    Frame {total_frames}: No skeleton detected")
        
        time.sleep(0.1)
    
    streamer.stop()
    
    print(f"\n  Results:")
    print(f"    Valid frames: {valid_frames}/{total_frames}")
    if reproj_errors:
        print(f"    Avg reprojection error: {np.mean(reproj_errors):.1f}px")
    
    return valid_frames > 0

def test_smplx_conversion(camera_ids, calibration_file):
    """Test SMPL-X conversion."""
    print("\n=== Testing SMPL-X Conversion ===")
    
    from multicam_pose_streamer import MultiCamPoseStreamer
    
    streamer = MultiCamPoseStreamer(
        camera_ids=camera_ids,
        calibration_file=calibration_file,
        resolution=(1280, 720),
        enable_display=False
    )
    streamer.start()
    
    print("  Running for 5 seconds...")
    time.sleep(2)  # Let cameras warm up
    
    valid_conversions = 0
    total_attempts = 0
    
    for _ in range(50):
        smplx_data, _, _, _, _ = streamer.get_current_frame()
        total_attempts += 1
        
        if smplx_data is not None:
            valid_conversions += 1
            if valid_conversions == 1:
                print(f"    First valid SMPL-X data:")
                print(f"      transl: {smplx_data['transl']}")
                print(f"      global_orient: {smplx_data['global_orient']}")
                print(f"      body_pose shape: {smplx_data['body_pose'].shape}")
        
        time.sleep(0.1)
    
    streamer.stop()
    
    print(f"\n  Results:")
    print(f"    Valid conversions: {valid_conversions}/{total_attempts}")
    
    return valid_conversions > 0

def main():
    # Configuration
    camera_ids = [4, 6, 2]
    calibration_file = "../calibration/calibration.toml"
    
    print("=" * 60)
    print("Multi-Camera Pose Streaming Diagnostic")
    print("=" * 60)
    
    # Run tests
    test_cameras(camera_ids)
    test_calibration(calibration_file)
    test_mediapipe()
    test_detection(camera_ids)
    test_triangulation(camera_ids, calibration_file)
    test_smplx_conversion(camera_ids, calibration_file)
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
