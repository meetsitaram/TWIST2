#!/usr/bin/env python3
"""
Compare robot default pose with camera-captured pose.

Usage:
    1. Stand in robot's default standing position (arms slightly forward, legs slightly bent)
    2. Run this script
    3. Compare the values
"""

import numpy as np
import time
import json
import redis

# Import the default pose
import sys
sys.path.insert(0, '.')
from data_utils.params import DEFAULT_MIMIC_OBS

def get_robot_default_pose():
    """Get the robot's default standing pose."""
    default = DEFAULT_MIMIC_OBS["unitree_g1_with_hands"]
    
    print("=" * 70)
    print("ROBOT DEFAULT STANDING POSE (mimic_obs format, 35 dims)")
    print("=" * 70)
    
    print("\n--- Root State (6 dims) ---")
    print(f"  [0] root_vel_x:    {default[0]:.4f}")
    print(f"  [1] root_vel_y:    {default[1]:.4f}")
    print(f"  [2] root_z:        {default[2]:.4f} m  (height)")
    print(f"  [3] roll:          {default[3]:.4f} rad")
    print(f"  [4] pitch:         {default[4]:.4f} rad")
    print(f"  [5] yaw_vel:       {default[5]:.4f} rad/s")
    
    print("\n--- Left Leg (6 dims) ---")
    print(f"  [6]  hip_pitch:    {default[6]:.4f} rad  ({np.degrees(default[6]):.1f}°)")
    print(f"  [7]  hip_roll:     {default[7]:.4f} rad  ({np.degrees(default[7]):.1f}°)")
    print(f"  [8]  hip_yaw:      {default[8]:.4f} rad  ({np.degrees(default[8]):.1f}°)")
    print(f"  [9]  knee:         {default[9]:.4f} rad  ({np.degrees(default[9]):.1f}°)")
    print(f"  [10] ankle_pitch:  {default[10]:.4f} rad ({np.degrees(default[10]):.1f}°)")
    print(f"  [11] ankle_roll:   {default[11]:.4f} rad ({np.degrees(default[11]):.1f}°)")
    
    print("\n--- Right Leg (6 dims) ---")
    print(f"  [12] hip_pitch:    {default[12]:.4f} rad ({np.degrees(default[12]):.1f}°)")
    print(f"  [13] hip_roll:     {default[13]:.4f} rad ({np.degrees(default[13]):.1f}°)")
    print(f"  [14] hip_yaw:      {default[14]:.4f} rad ({np.degrees(default[14]):.1f}°)")
    print(f"  [15] knee:         {default[15]:.4f} rad ({np.degrees(default[15]):.1f}°)")
    print(f"  [16] ankle_pitch:  {default[16]:.4f} rad ({np.degrees(default[16]):.1f}°)")
    print(f"  [17] ankle_roll:   {default[17]:.4f} rad ({np.degrees(default[17]):.1f}°)")
    
    print("\n--- Waist/Torso (3 dims) ---")
    print(f"  [18] waist_yaw:    {default[18]:.4f} rad ({np.degrees(default[18]):.1f}°)")
    print(f"  [19] waist_pitch:  {default[19]:.4f} rad ({np.degrees(default[19]):.1f}°)")
    print(f"  [20] waist_roll:   {default[20]:.4f} rad ({np.degrees(default[20]):.1f}°)")
    
    print("\n--- Left Arm (7 dims) ---")
    print(f"  [21] shoulder_pitch: {default[21]:.4f} rad ({np.degrees(default[21]):.1f}°)")
    print(f"  [22] shoulder_roll:  {default[22]:.4f} rad ({np.degrees(default[22]):.1f}°)")
    print(f"  [23] shoulder_yaw:   {default[23]:.4f} rad ({np.degrees(default[23]):.1f}°)")
    print(f"  [24] elbow:          {default[24]:.4f} rad ({np.degrees(default[24]):.1f}°)")
    print(f"  [25] wrist_roll:     {default[25]:.4f} rad ({np.degrees(default[25]):.1f}°)")
    print(f"  [26] wrist_pitch:    {default[26]:.4f} rad ({np.degrees(default[26]):.1f}°)")
    print(f"  [27] wrist_yaw:      {default[27]:.4f} rad ({np.degrees(default[27]):.1f}°)")
    
    print("\n--- Right Arm (7 dims) ---")
    print(f"  [28] shoulder_pitch: {default[28]:.4f} rad ({np.degrees(default[28]):.1f}°)")
    print(f"  [29] shoulder_roll:  {default[29]:.4f} rad ({np.degrees(default[29]):.1f}°)")
    print(f"  [30] shoulder_yaw:   {default[30]:.4f} rad ({np.degrees(default[30]):.1f}°)")
    print(f"  [31] elbow:          {default[31]:.4f} rad ({np.degrees(default[31]):.1f}°)")
    print(f"  [32] wrist_roll:     {default[32]:.4f} rad ({np.degrees(default[32]):.1f}°)")
    print(f"  [33] wrist_pitch:    {default[33]:.4f} rad ({np.degrees(default[33]):.1f}°)")
    print(f"  [34] wrist_yaw:      {default[34]:.4f} rad ({np.degrees(default[34]):.1f}°)")
    
    return default


def capture_camera_pose():
    """Capture pose from cameras and convert to mimic_obs."""
    from multicam_pose_streamer import MultiCamPoseStreamer
    import cv2
    
    camera_ids = [4, 6, 2]
    calibration_file = "../calibration/calibration.toml"
    
    print("\n" + "=" * 70)
    print("CAPTURING YOUR POSE FROM CAMERAS")
    print("=" * 70)
    print("\nStand in the default robot pose:")
    print("  - Feet shoulder-width apart")
    print("  - Knees slightly bent (~23°)")
    print("  - Arms relaxed, elbows bent (~69°)")
    print("  - Face the cameras")
    
    # Show live preview first with skeleton overlay
    print("\n--- LIVE PREVIEW WITH SKELETON ---")
    print("Position yourself so you're visible in all cameras.")
    print("Press 'c' to capture, 'q' to quit")
    
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    # Open cameras for preview - use lower resolution to avoid USB bandwidth issues
    caps = {}
    detectors = {}
    print("  Opening cameras (using 640x480 to avoid USB bandwidth issues)...")
    
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        
        # Use MJPEG mode - critical for multi-camera USB bandwidth!
        # Without this: ~10 FPS per camera (raw YUYV saturates USB)
        # With this: ~26 FPS per camera (compressed MJPEG)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            # Warm up camera - read a few frames
            print(f"  Camera {cam_id}: warming up...", end=" ")
            for _ in range(5):
                ret, _ = cap.read()
                time.sleep(0.1)
            
            # Test if it's actually returning frames
            ret, frame = cap.read()
            if ret and frame is not None:
                caps[cam_id] = cap
                detectors[cam_id] = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=0,  # Faster model for preview
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print(f"OK ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"NO FRAMES (USB bandwidth issue?)")
                cap.release()
        else:
            print(f"  Camera {cam_id}: FAILED to open")
    
    print(f"  Total cameras working: {len(caps)}/{len(camera_ids)}")
    
    if len(caps) < 2:
        print("\nERROR: Need at least 2 working cameras for triangulation!")
        print("Try unplugging and replugging cameras, or use different USB ports.")
        for cap in caps.values():
            cap.release()
        return None
    
    cv2.namedWindow("Camera Preview - Press 'c' to capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Preview - Press 'c' to capture", 1920, 480)
    
    while True:
        frames = []
        for cam_id in camera_ids:
            if cam_id in caps:
                ret, frame = caps[cam_id].read()
                if ret:
                    # Run MediaPipe detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detectors[cam_id].process(frame_rgb)
                    
                    # Draw skeleton
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        status = "DETECTED"
                        color = (0, 255, 0)
                    else:
                        status = "NO POSE"
                        color = (0, 0, 255)
                    
                    # Add camera label and status
                    cv2.putText(frame, f"Cam {cam_id}: {status}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Resize for display
                    frame = cv2.resize(frame, (640, 360))
                    frames.append(frame)
                else:
                    # Create placeholder for failed read
                    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Cam {cam_id}: NO FRAME", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frames.append(placeholder)
            else:
                # Create placeholder for camera that didn't open
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Cam {cam_id}: NOT OPENED", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames.append(placeholder)
        
        if frames:
            # Combine frames horizontally
            combined = np.hstack(frames)
            cv2.imshow("Camera Preview - Press 'c' to capture", combined)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('c'):
            # Countdown with live preview so user can get in position
            print("\n--- GET INTO POSITION! ---")
            for countdown in range(5, 0, -1):
                # Keep showing preview during countdown
                frames = []
                for cam_id in camera_ids:
                    if cam_id in caps:
                        ret, frame = caps[cam_id].read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = detectors[cam_id].process(frame_rgb)
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame, results.pose_landmarks,
                                    mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                                )
                            # Big countdown text
                            cv2.putText(frame, f"CAPTURING IN {countdown}...", (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                            frame = cv2.resize(frame, (640, 360))
                            frames.append(frame)
                
                if frames:
                    combined = np.hstack(frames)
                    cv2.imshow("Camera Preview - Press 'c' to capture", combined)
                    cv2.waitKey(1)
                
                print(f"  Capturing in {countdown}...")
                time.sleep(1)
            
            print("\n*** HOLD STILL - CAPTURING NOW! ***")
            break
        elif key == ord('q'):
            print("\nCancelled.")
            for cap in caps.values():
                cap.release()
            for det in detectors.values():
                det.close()
            cv2.destroyAllWindows()
            return None
    
    # Close preview
    for cap in caps.values():
        cap.release()
    for det in detectors.values():
        det.close()
    cv2.destroyAllWindows()
    
    # Start streamer
    streamer = MultiCamPoseStreamer(
        camera_ids=camera_ids,
        calibration_file=calibration_file,
        resolution=(1280, 720),
        enable_display=False
    )
    streamer.start()
    time.sleep(2)  # Warm up
    
    # Capture multiple frames and average using direct G1 conversion
    captured_poses = []
    
    for _ in range(30):  # 3 seconds of data
        mimic_obs = streamer.get_mimic_obs()
        if mimic_obs is not None:
            captured_poses.append(mimic_obs)
        time.sleep(0.1)
    
    streamer.stop()
    
    if not captured_poses:
        print("\nERROR: No valid poses captured!")
        return None
    
    # Average the captured poses
    avg_pose = np.mean(captured_poses, axis=0)
    print(f"\nCaptured {len(captured_poses)} valid frames")
    
    return avg_pose


def compare_poses(robot_pose, camera_pose):
    """Compare robot default pose with camera-captured pose."""
    print("\n" + "=" * 70)
    print("COMPARISON: Robot Default vs Camera Captured")
    print("=" * 70)
    
    labels = [
        "root_vel_x", "root_vel_y", "root_z", "roll", "pitch", "yaw_vel",
        "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
        "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
        "waist_yaw", "waist_pitch", "waist_roll",
        "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow", "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
        "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow", "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
    ]
    
    print(f"\n{'Index':<6} {'Label':<20} {'Robot':<12} {'Camera':<12} {'Diff':<12} {'Status'}")
    print("-" * 75)
    
    large_diffs = []
    for i, (r, c, label) in enumerate(zip(robot_pose, camera_pose, labels)):
        diff = c - r
        
        # Threshold depends on type
        if i < 2:  # velocities
            threshold = 0.5
        elif i == 2:  # height
            threshold = 0.3
        elif i < 6:  # orientation
            threshold = 0.5
        else:  # joint angles
            threshold = 0.5  # ~30 degrees
        
        status = "OK" if abs(diff) < threshold else "LARGE DIFF"
        if status == "LARGE DIFF":
            large_diffs.append((i, label, r, c, diff))
        
        # Convert to degrees for joint angles
        if i >= 6:
            print(f"[{i:<3}]  {label:<20} {np.degrees(r):>8.1f}°   {np.degrees(c):>8.1f}°   {np.degrees(diff):>+8.1f}°  {status}")
        else:
            print(f"[{i:<3}]  {label:<20} {r:>8.4f}    {c:>8.4f}    {diff:>+8.4f}   {status}")
    
    if large_diffs:
        print("\n--- LARGE DIFFERENCES ---")
        for i, label, r, c, diff in large_diffs:
            if i >= 6:
                print(f"  [{i}] {label}: robot={np.degrees(r):.1f}°, camera={np.degrees(c):.1f}°, diff={np.degrees(diff):+.1f}°")
            else:
                print(f"  [{i}] {label}: robot={r:.4f}, camera={c:.4f}, diff={diff:+.4f}")
    
    return large_diffs


def send_to_redis(pose):
    """Send pose to Redis for robot to execute."""
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        r.set("action_body_unitree_g1_with_hands", json.dumps(pose.tolist()))
        r.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        r.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        r.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
        print("\nPose sent to Redis!")
        return True
    except Exception as e:
        print(f"\nFailed to send to Redis: {e}")
        return False


def main():
    # Show robot default pose
    robot_pose = get_robot_default_pose()
    
    print("\n" + "-" * 70)
    input("\nPress Enter to capture your pose from cameras...")
    
    # Capture camera pose
    camera_pose = capture_camera_pose()
    
    if camera_pose is not None:
        # Compare
        large_diffs = compare_poses(robot_pose, camera_pose)
        
        print("\n" + "=" * 70)
        print("OPTIONS")
        print("=" * 70)
        print("\n1. Send ROBOT default pose to Redis (safe)")
        print("2. Send CAMERA captured pose to Redis (test)")
        print("3. Exit")
        
        choice = input("\nChoice [1/2/3]: ")
        
        if choice == "1":
            send_to_redis(robot_pose)
        elif choice == "2":
            send_to_redis(camera_pose)
        else:
            print("Exiting.")


if __name__ == "__main__":
    main()
