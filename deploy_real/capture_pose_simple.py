#!/usr/bin/env python3
"""
Simple Pose Capture - works reliably with display.

Usage:
    python capture_pose_simple.py
    
Controls:
    SPACE - Capture pose
    Q - Quit
"""

import numpy as np
import cv2
import time
import json
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy_real.multicam_pose_streamer import MultiCamPoseStreamer

CALIBRATION_DIR = PROJECT_ROOT / "calibration"
POSES_DIR = CALIBRATION_DIR / "captured_poses"


# MediaPipe landmark indices
class MP:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


def compute_angle_3d(p1, p2, p3):
    """Compute angle at p2 formed by vectors p1->p2 and p2->p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


def compute_joint_angles(skeleton):
    """Compute joint angles from MediaPipe skeleton."""
    angles = {}
    
    l_shoulder = skeleton[MP.LEFT_SHOULDER]
    r_shoulder = skeleton[MP.RIGHT_SHOULDER]
    l_elbow = skeleton[MP.LEFT_ELBOW]
    r_elbow = skeleton[MP.RIGHT_ELBOW]
    l_wrist = skeleton[MP.LEFT_WRIST]
    r_wrist = skeleton[MP.RIGHT_WRIST]
    l_hip = skeleton[MP.LEFT_HIP]
    r_hip = skeleton[MP.RIGHT_HIP]
    l_knee = skeleton[MP.LEFT_KNEE]
    r_knee = skeleton[MP.RIGHT_KNEE]
    l_ankle = skeleton[MP.LEFT_ANKLE]
    r_ankle = skeleton[MP.RIGHT_ANKLE]
    
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip = (l_hip + r_hip) / 2
    
    spine_dir = mid_shoulder - mid_hip
    spine_dir = spine_dir / (np.linalg.norm(spine_dir) + 1e-8)
    
    shoulder_dir = r_shoulder - l_shoulder
    shoulder_dir = shoulder_dir / (np.linalg.norm(shoulder_dir) + 1e-8)
    
    forward_dir = np.cross(spine_dir, shoulder_dir)
    forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)
    
    right_dir = shoulder_dir
    
    # Left arm
    l_upper_arm = l_elbow - l_shoulder
    l_upper_arm_norm = l_upper_arm / (np.linalg.norm(l_upper_arm) + 1e-8)
    angles["left_shoulder_pitch"] = np.degrees(np.arcsin(np.clip(np.dot(l_upper_arm_norm, forward_dir), -1, 1)))
    angles["left_shoulder_roll"] = np.degrees(np.arcsin(np.clip(np.dot(l_upper_arm_norm, spine_dir), -1, 1)))
    
    # Left shoulder yaw - computed from forearm orientation relative to upper arm plane
    l_forearm = l_wrist - l_elbow
    l_forearm_norm = l_forearm / (np.linalg.norm(l_forearm) + 1e-8)
    # Create a reference plane perpendicular to upper arm
    # Project forearm onto plane perpendicular to upper arm
    l_forearm_perp = l_forearm_norm - np.dot(l_forearm_norm, l_upper_arm_norm) * l_upper_arm_norm
    l_forearm_perp_norm = np.linalg.norm(l_forearm_perp)
    if l_forearm_perp_norm > 0.1:  # Only if elbow is bent enough
        l_forearm_perp = l_forearm_perp / l_forearm_perp_norm
        # Reference direction: perpendicular to arm and pointing "down" (toward body center)
        l_arm_forward = np.cross(l_upper_arm_norm, spine_dir)
        l_arm_forward = l_arm_forward / (np.linalg.norm(l_arm_forward) + 1e-8)
        l_arm_up = np.cross(l_arm_forward, l_upper_arm_norm)
        # Yaw is angle of forearm in the perpendicular plane
        yaw_cos = np.dot(l_forearm_perp, l_arm_up)
        yaw_sin = np.dot(l_forearm_perp, l_arm_forward)
        angles["left_shoulder_yaw"] = np.degrees(np.arctan2(yaw_sin, yaw_cos))
    else:
        angles["left_shoulder_yaw"] = 0.0
    
    # Right arm
    r_upper_arm = r_elbow - r_shoulder
    r_upper_arm_norm = r_upper_arm / (np.linalg.norm(r_upper_arm) + 1e-8)
    angles["right_shoulder_pitch"] = np.degrees(np.arcsin(np.clip(np.dot(r_upper_arm_norm, forward_dir), -1, 1)))
    angles["right_shoulder_roll"] = np.degrees(np.arcsin(np.clip(-np.dot(r_upper_arm_norm, spine_dir), -1, 1)))
    
    # Right shoulder yaw
    r_forearm = r_wrist - r_elbow
    r_forearm_norm = r_forearm / (np.linalg.norm(r_forearm) + 1e-8)
    r_forearm_perp = r_forearm_norm - np.dot(r_forearm_norm, r_upper_arm_norm) * r_upper_arm_norm
    r_forearm_perp_norm = np.linalg.norm(r_forearm_perp)
    if r_forearm_perp_norm > 0.1:
        r_forearm_perp = r_forearm_perp / r_forearm_perp_norm
        r_arm_forward = np.cross(r_upper_arm_norm, spine_dir)
        r_arm_forward = r_arm_forward / (np.linalg.norm(r_arm_forward) + 1e-8)
        r_arm_up = np.cross(r_arm_forward, r_upper_arm_norm)
        yaw_cos = np.dot(r_forearm_perp, r_arm_up)
        yaw_sin = np.dot(r_forearm_perp, r_arm_forward)
        angles["right_shoulder_yaw"] = -np.degrees(np.arctan2(yaw_sin, yaw_cos))  # Negate for right side
    else:
        angles["right_shoulder_yaw"] = 0.0
    
    # Elbows
    angles["left_elbow"] = 180 - compute_angle_3d(l_shoulder, l_elbow, l_wrist)
    angles["right_elbow"] = 180 - compute_angle_3d(r_shoulder, r_elbow, r_wrist)
    
    # Left leg
    l_thigh = l_knee - l_hip
    l_thigh_norm = l_thigh / (np.linalg.norm(l_thigh) + 1e-8)
    angles["left_hip_pitch"] = np.degrees(np.arcsin(np.clip(np.dot(l_thigh_norm, forward_dir), -1, 1)))
    angles["left_hip_roll"] = np.degrees(np.arcsin(np.clip(-np.dot(l_thigh_norm, right_dir), -1, 1)))
    
    # Right leg
    r_thigh = r_knee - r_hip
    r_thigh_norm = r_thigh / (np.linalg.norm(r_thigh) + 1e-8)
    angles["right_hip_pitch"] = np.degrees(np.arcsin(np.clip(np.dot(r_thigh_norm, forward_dir), -1, 1)))
    angles["right_hip_roll"] = np.degrees(np.arcsin(np.clip(np.dot(r_thigh_norm, right_dir), -1, 1)))
    
    # Knees
    angles["left_knee"] = 180 - compute_angle_3d(l_hip, l_knee, l_ankle)
    angles["right_knee"] = 180 - compute_angle_3d(r_hip, r_knee, r_ankle)
    
    return angles


def main():
    print("\n" + "="*60)
    print("  SIMPLE POSE CAPTURE")
    print("="*60)
    print("\nControls (click on camera window first!):")
    print("  SPACE - Start 5-second countdown and capture")
    print("  Q     - Quit")
    print("\nNOTE: Click on the camera window to give it focus for key input.")
    print("="*60)
    
    # Load config
    config_path = CALIBRATION_DIR / 'camera_config.yaml'
    calibration_file = CALIBRATION_DIR / 'calibration.toml'
    
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    camera_ids = config.get('camera_ids', [])
    print(f"\nCameras: {camera_ids}")
    
    # Load world frame correction if available
    world_pitch = config.get('world_pitch_correction_deg', 0.0)
    world_roll = config.get('world_roll_correction_deg', 0.0)
    leg_pitch = config.get('leg_pitch_correction_deg', 0.0)
    if abs(world_pitch) > 0.1 or abs(world_roll) > 0.1:
        print(f"World frame correction: pitch={world_pitch}°, roll={world_roll}°")
    if abs(leg_pitch) > 0.1:
        print(f"Leg pitch correction: {leg_pitch}°")
    
    # Initialize streamer (disable internal display, we'll handle it)
    print("Starting cameras...")
    streamer = MultiCamPoseStreamer(
        camera_ids=camera_ids,
        calibration_file=str(calibration_file),
        resolution=(1280, 720),
        enable_display=False,  # We handle display in main thread
        world_pitch_correction_deg=world_pitch,
        world_roll_correction_deg=world_roll,
        leg_pitch_correction_deg=leg_pitch,
    )
    streamer.start()
    
    # Create window in main thread for proper key handling
    # Use fixed size matching the streamer output (1280x720)
    window_name = "Pose Capture - SPACE to capture, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Show initial "waiting" frame so window is visible (match streamer output size)
    waiting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(waiting_frame, "Starting cameras...", (500, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imshow(window_name, waiting_frame)
    cv2.waitKey(100)  # Give window time to appear
    
    print("\nReady! Press SPACE to capture a pose.\n")
    
    POSES_DIR.mkdir(parents=True, exist_ok=True)
    pose_count = 0
    last_valid_frame = None  # Keep last valid frame to prevent flashing
    
    try:
        while True:
            # Get current skeleton
            skeleton_3d, reproj_error = streamer.get_3d_skeleton()
            
            # Get and display the camera frames
            display_frame = streamer.get_display_frame()
            if display_frame is not None:
                last_valid_frame = display_frame.copy()
                # Add instructions overlay
                cv2.putText(display_frame, "SPACE: Capture | Q: Quit", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow(window_name, display_frame)
            elif last_valid_frame is not None:
                # Use last valid frame to prevent flashing
                cv2.imshow(window_name, last_valid_frame)
            else:
                # Initial startup - show waiting message (match streamer output size: 1280x720)
                waiting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(waiting_frame, "Waiting for camera frames...", (450, 340),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(waiting_frame, "Press Q to quit", (550, 400),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow(window_name, waiting_frame)
            
            # Check for key press - window must be focused!
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord(' '):
                # Countdown
                print("\n" + "="*40)
                print("CAPTURING in 10 seconds - HOLD STILL!")
                for i in range(10, 0, -1):
                    print(f"  {i}...")
                    # Keep updating display during countdown
                    for _ in range(10):  # ~1 second with 100ms sleep
                        display_frame = streamer.get_display_frame()
                        if display_frame is not None:
                            # Add countdown overlay
                            overlay = display_frame.copy()
                            cv2.putText(overlay, str(i), (overlay.shape[1]//2 - 50, overlay.shape[0]//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                            cv2.imshow(window_name, overlay)
                        cv2.waitKey(100)
                
                print("CAPTURING...")
                
                # Capture for 2 seconds
                skeletons = []
                start = time.time()
                while time.time() - start < 2.0:
                    skel, _ = streamer.get_3d_skeleton()
                    if skel is not None:
                        skeletons.append(skel.copy())
                    # Keep updating display during capture
                    display_frame = streamer.get_display_frame()
                    if display_frame is not None:
                        # Add "CAPTURING" overlay
                        overlay = display_frame.copy()
                        cv2.putText(overlay, "CAPTURING", (20, overlay.shape[0]//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        cv2.imshow(window_name, overlay)
                    cv2.waitKey(30)
                
                if len(skeletons) < 5:
                    print(f"Only got {len(skeletons)} frames. Try again.")
                    continue
                
                # Average
                avg_skeleton = np.mean(skeletons, axis=0)
                std_skeleton = np.std(skeletons, axis=0)
                avg_std = np.mean(std_skeleton)
                
                print(f"Captured {len(skeletons)} frames (std: {avg_std*100:.1f}cm)")
                
                # Compute angles
                angles = compute_joint_angles(avg_skeleton)
                
                print("\nJoint Angles:")
                print("-"*40)
                for joint, angle in sorted(angles.items()):
                    print(f"  {joint:25s}: {angle:+7.1f}°")
                
                # Get name
                print("\n" + "-"*40)
                name = input("Name this pose (or 'skip'): ").strip()
                
                if name.lower() == 'skip' or not name:
                    print("Skipped.")
                    continue
                
                name = name.replace(" ", "_").lower()
                
                # Save
                pose_data = {
                    "name": name,
                    "timestamp": datetime.now().isoformat(),
                    "num_frames": len(skeletons),
                    "avg_std_cm": float(avg_std * 100),
                    "skeleton_3d": avg_skeleton.tolist(),
                    "joint_angles": {k: float(v) for k, v in angles.items()},
                }
                
                filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = POSES_DIR / filename
                
                with open(filepath, 'w') as f:
                    json.dump(pose_data, f, indent=2)
                
                pose_count += 1
                print(f"\nSaved: {filepath}")
                print(f"Total poses: {pose_count}")
                print("\nPress SPACE for next pose, Q to quit.")
    
    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
        print(f"\nDone. Captured {pose_count} poses.")


if __name__ == "__main__":
    main()
