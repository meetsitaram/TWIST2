#!/usr/bin/env python3
"""
Capture Default Pose

Stand in your preferred default pose, press SPACE to capture.
The joint angles will be saved to calibration/default_pose.yaml
and automatically loaded by stream_ik_teleop.py.

Usage:
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    conda activate gmr
    python capture_default_pose.py
"""

import argparse
import numpy as np
import yaml
import time
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
DEFAULT_POSE_FILE = CALIBRATION_DIR / "default_pose.yaml"
CAMERA_CONFIG_FILE = CALIBRATION_DIR / "camera_config.yaml"

# Joint order (matches the 29-DOF robot)
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def main():
    parser = argparse.ArgumentParser(description="Capture default pose from camera")
    parser.add_argument("--two-stage", "-2", action="store_true",
                       help="Use two-stage IK (whole body)")
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    from multicam_pose_streamer import MultiCamPoseStreamer
    from end_effector_ik_retarget import EndEffectorIKRetargeter
    import cv2
    
    # Load camera config
    with open(CAMERA_CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    
    camera_ids = config.get('camera_ids', [0, 1, 2])
    calibration_file = str(CALIBRATION_DIR / "calibration.toml")
    
    # Load corrections
    world_pitch = config.get('world_pitch_correction_deg', 0.0)
    world_roll = config.get('world_roll_correction_deg', 0.0)
    leg_pitch = config.get('leg_pitch_correction_deg', 0.0)
    
    print("\n" + "=" * 60)
    print("  CAPTURE DEFAULT POSE")
    print("=" * 60)
    print(f"\nCameras: {camera_ids}")
    if abs(leg_pitch) > 0.1:
        print(f"Leg pitch correction: {leg_pitch}°")
    print("\nInstructions:")
    print("  1. Stand in your preferred DEFAULT pose")
    print("     (e.g., arms at sides, standing straight)")
    print("  2. Press SPACE to capture")
    print("  3. Press S to save to config")
    print("  4. Press Q to quit")
    print("=" * 60 + "\n")
    
    # Initialize camera streamer
    print("Starting cameras...")
    streamer = MultiCamPoseStreamer(
        camera_ids=camera_ids,
        calibration_file=calibration_file,
        enable_display=False,
        world_pitch_correction_deg=world_pitch,
        world_roll_correction_deg=world_roll,
        leg_pitch_correction_deg=leg_pitch,
    )
    streamer.start()
    time.sleep(2.0)
    
    # Initialize IK retargeter
    print("Initializing IK retargeter...")
    retargeter = EndEffectorIKRetargeter(
        verbose=False,
        max_iterations=50,
        ground_clearance=0.08,
    )
    
    # Create window
    cv2.namedWindow("Capture Default Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture Default Pose", 1280, 720)
    
    captured_qpos = None
    captured_joint_angles = None
    
    try:
        while True:
            # Get skeleton
            skeleton, reproj_error = streamer.get_3d_skeleton()
            
            # Get display frame
            display_frame = streamer.get_display_frame()
            
            if display_frame is not None:
                h, w = display_frame.shape[:2]
                
                # Status
                if captured_qpos is not None:
                    status = "CAPTURED! Press S to save, SPACE to recapture"
                    color = (0, 255, 0)
                else:
                    status = "Stand in default pose, press SPACE to capture"
                    color = (0, 255, 255)
                
                cv2.putText(display_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                if skeleton is not None:
                    cv2.putText(display_frame, "Skeleton detected", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No skeleton - move into view", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow("Capture Default Pose", display_frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):
                # Start countdown
                print("\n" + "=" * 40)
                print("GET INTO POSITION!")
                print("Capturing in 10 seconds...")
                print("=" * 40)
                
                # 10 second countdown with visual feedback
                for countdown in range(10, 0, -1):
                    print(f"  {countdown}...")
                    
                    # Keep updating display during countdown
                    for _ in range(10):  # ~1 second with 100ms iterations
                        display_frame = streamer.get_display_frame()
                        if display_frame is not None:
                            # Add countdown overlay
                            overlay = display_frame.copy()
                            h, w = overlay.shape[:2]
                            
                            # Big countdown number
                            cv2.putText(overlay, str(countdown), (w//2 - 80, h//2 + 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                            cv2.putText(overlay, "GET READY!", (w//2 - 150, h//2 - 80),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                            
                            cv2.imshow("Capture Default Pose", overlay)
                        
                        # Check for cancel (Q key)
                        cancel_key = cv2.waitKey(100) & 0xFF
                        if cancel_key == ord('q'):
                            print("Cancelled")
                            break
                        time.sleep(0.0)  # Small sleep to not hog CPU
                    else:
                        continue
                    break  # Break outer loop if cancelled
                else:
                    # Countdown finished - now capture
                    print("\nCAPTURING NOW - HOLD STILL!")
                    
                    # Get fresh skeleton
                    skeleton, reproj_error = streamer.get_3d_skeleton()
                    
                    if skeleton is None:
                        print("No skeleton detected - cannot capture")
                        continue
                    
                    # Run IK
                    if args.two_stage:
                        result = retargeter.retarget_two_stage(skeleton)
                    else:
                        result = retargeter.retarget(skeleton, fixed_base=True)
                    
                    if result.get('valid', False):
                        captured_qpos = result['qpos']
                        captured_joint_angles = result['joint_angles_rad']
                        
                        print("\nCaptured joint angles (radians):")
                        print("-" * 40)
                        for joint_name, angle in captured_joint_angles.items():
                            print(f"  {joint_name}: {angle:.4f} ({np.degrees(angle):.1f}°)")
                        print("-" * 40)
                        print(f"Pelvis height: {captured_qpos[2]:.3f}m")
                        print("\nPress S to save, SPACE to recapture")
                    else:
                        print("IK failed - try again")
            
            elif key == ord('s'):
                # Save to config
                if captured_qpos is None:
                    print("No pose captured yet - press SPACE first")
                    continue
                
                # Build config dict
                default_pose = {
                    'pelvis_height': float(captured_qpos[2]),
                    'joint_angles_rad': {k: float(v) for k, v in captured_joint_angles.items()},
                }
                
                # Save to file
                with open(DEFAULT_POSE_FILE, 'w') as f:
                    yaml.dump({'default_pose': default_pose}, f, default_flow_style=False)
                
                print(f"\nSaved default pose to: {DEFAULT_POSE_FILE}")
                print("This will be loaded automatically by stream_ik_teleop.py")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        streamer.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
