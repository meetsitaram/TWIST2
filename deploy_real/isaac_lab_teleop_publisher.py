#!/usr/bin/env python3
"""
Isaac Lab Teleop Publisher

Publishes upper body joint targets from camera streaming to Redis
for consumption by play_isaaclab_teleop.py.

This bridges the existing multicam_pose_streamer.py to the Isaac Lab
teleop overlay deployment.

Usage:
    Terminal 1 (publish teleop targets):
        conda activate gmr
        cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
        python isaac_lab_teleop_publisher.py --display

    Terminal 2 (run policy with teleop):
        conda activate env_isaaclab
        cd ~/projects/g1-pick-n-place/TWIST2
        python scripts/play_isaaclab_teleop.py \
            --checkpoint logs/isaaclab/motion_mimic/model_17500.pt \
            --teleop redis

Architecture:
    Camera Capture → MediaPipe → Triangulation → IK Retarget → Redis → Isaac Lab Policy
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml
import cv2

# Add deploy_real to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Add TWIST2 root for imports
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)


def load_camera_config():
    """Load camera configuration."""
    project_dir = os.path.dirname(SCRIPT_DIR)
    calibration_dir = os.path.join(project_dir, "calibration")
    config_path = os.path.join(calibration_dir, "camera_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Publish teleop targets to Redis")
    
    # Camera settings
    parser.add_argument("--cameras", type=int, nargs='+', default=None,
                       help="Camera IDs (default: from camera_config.yaml)")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Path to calibration.toml")
    
    # Display
    parser.add_argument("--display", action="store_true",
                       help="Show visualization window")
    
    # Redis settings
    parser.add_argument("--redis_host", type=str, default="localhost",
                       help="Redis server host")
    parser.add_argument("--redis_port", type=int, default=6379,
                       help="Redis server port")
    parser.add_argument("--redis_key", type=str, default="teleop:mimic_obs",
                       help="Redis key for teleop targets")
    
    # IK settings
    parser.add_argument("--smoothing", type=str, default="one_euro",
                       choices=["none", "one_euro"],
                       help="Skeleton smoothing method")
    parser.add_argument("--responsive", action="store_true",
                       help="More responsive tracking (less smoothing, may be jittery)")
    parser.add_argument("--debug_timing", action="store_true",
                       help="Print timing breakdown to identify lag sources")
    parser.add_argument("--direct", action="store_true",
                       help="Use direct joint mapping instead of IK solver (faster)")
    
    # Countdown
    parser.add_argument("--countdown", type=int, default=10,
                       help="Countdown seconds before starting (default: 10)")
    
    # MuJoCo visualization
    parser.add_argument("--mujoco_viz", action="store_true",
                       help="Show MuJoCo viewer with the IK'd robot pose (input visualization)")
    
    args = parser.parse_args()
    
    # Import Redis
    try:
        import redis
    except ImportError:
        print("ERROR: redis-py not installed. Install with: pip install redis")
        return
    
    # Import components
    from multicam_pose_streamer import MultiCamPoseStreamer
    from end_effector_ik_retarget import EndEffectorIKRetargeter
    from mediapipe_to_g1_direct import MediaPipeToG1Direct
    
    # Load camera config
    config = load_camera_config()
    camera_ids = args.cameras or config['camera_ids']
    
    # Calibration path
    calibration_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "calibration")
    calibration_path = args.calibration or os.path.join(calibration_dir, "calibration.toml")
    
    print("=" * 60)
    print("  Isaac Lab Teleop Publisher")
    print("=" * 60)
    print(f"\nCameras: {camera_ids}")
    print(f"Calibration: {calibration_path}")
    print(f"Redis: {args.redis_host}:{args.redis_port}")
    print(f"Key: {args.redis_key}")
    print(f"Smoothing: {args.smoothing}")
    print(f"Display: {args.display}")
    print("-" * 60)
    
    # Connect to Redis
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port)
    try:
        redis_client.ping()
        print(f"[Redis] Connected to {args.redis_host}:{args.redis_port}")
    except redis.ConnectionError as e:
        print(f"ERROR: Could not connect to Redis: {e}")
        print("Make sure Redis is running: redis-server")
        return
    
    # Smoothing parameters - responsive mode reduces lag but may be jittery
    if args.responsive:
        min_cutoff = 3.0   # Higher = less smoothing at low speeds
        beta = 0.5         # Higher = faster response to speed changes
        print("[Streamer] Responsive mode: reduced smoothing for lower latency")
    else:
        min_cutoff = 1.0
        beta = 0.007
    
    # Create pose streamer (disable internal display - we handle it in main thread)
    streamer = MultiCamPoseStreamer(
        camera_ids=camera_ids,
        calibration_file=calibration_path,
        enable_hands=True,
        enable_display=False,  # We handle display in main thread
        skeleton_smoothing=args.smoothing,
        smoothing_min_cutoff=min_cutoff,
        smoothing_beta=beta,
    )
    
    # Create retargeter (IK-based or direct mapping)
    mujoco_model_path = os.path.join(TWIST2_ROOT, "assets/g1/g1_mocap_29dof.xml")
    
    if args.direct:
        print("[Retarget] Using DIRECT joint mapping (faster)")
        retargeter = MediaPipeToG1Direct()
        use_direct = True
    else:
        print("[Retarget] Using IK solver (mink)")
        retargeter = EndEffectorIKRetargeter(
            model_path=mujoco_model_path,
            verbose=False,
        )
        use_direct = False
    
    # MuJoCo viewer for input visualization
    mj_model = None
    mj_data = None
    mj_viewer = None
    
    if args.mujoco_viz:
        try:
            import mujoco
            import mujoco.viewer
            
            print("[MuJoCo] Loading model for visualization...")
            mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path)
            mj_data = mujoco.MjData(mj_model)
            
            # Launch passive viewer (non-blocking)
            mj_viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
            mj_viewer.cam.azimuth = 180
            mj_viewer.cam.elevation = -20
            mj_viewer.cam.distance = 3.0
            
            print("[MuJoCo] Viewer launched - showing IK'd robot pose")
        except ImportError:
            print("[MuJoCo] ERROR: mujoco not installed. Install with: pip install mujoco")
            args.mujoco_viz = False
        except Exception as e:
            print(f"[MuJoCo] ERROR: Could not create viewer: {e}")
            args.mujoco_viz = False
    
    # Start streaming
    streamer.start()
    print("[Streamer] Started multi-camera pose streaming")
    
    # Create display window if display enabled (vertical stack: 480x270 per camera)
    window_name = "Teleop Publisher"
    if args.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 810)  # 3 cameras stacked vertically
    
    # Wait for ENTER key, then countdown
    print("\n" + "=" * 60)
    print("  TELEOP PUBLISHER READY")
    print("=" * 60)
    print("  Press ENTER to start (10-second countdown)")
    print("  Press Ctrl+C to quit")
    print("=" * 60)
    
    input("\n[Teleop] Press ENTER to start countdown...")
    
    # Countdown
    countdown_seconds = args.countdown
    print(f"\n[Teleop] Starting in {countdown_seconds} seconds - GET INTO POSITION!")
    for remaining in range(countdown_seconds, 0, -1):
        print(f"  {remaining}...")
        # Update display during countdown
        if args.display:
            frames_dict, _ = streamer.get_latest_frames()
            if frames_dict:
                CELL_W, CELL_H = 480, 270
                sorted_cam_ids = sorted(frames_dict.keys())
                stacked_frames = []
                for cam_id in sorted_cam_ids[:3]:
                    frame = frames_dict[cam_id]
                    resized = cv2.resize(frame, (CELL_W, CELL_H))
                    # Add countdown overlay
                    cv2.putText(resized, str(remaining), (CELL_W//2 - 30, CELL_H//2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                    stacked_frames.append(resized)
                if stacked_frames:
                    overlay = np.vstack(stacked_frames)
                    cv2.imshow(window_name, overlay)
            cv2.waitKey(100)
        time.sleep(1)
    
    print("\n[Teleop] GO!\n")
    print("Press Ctrl+C or Q to stop\n")
    
    # Statistics
    frame_count = 0
    valid_count = 0
    start_time = time.time()
    last_print_time = start_time
    last_dof_pos = None
    last_ik_error = 0.0
    first_publish = True  # Track first successful Redis publish
    
    # Failure tracking
    fail_no_skeleton = 0
    fail_nan_skeleton = 0
    fail_ik_none = 0
    fail_ik_exception = 0
    
    # Timing stats
    timing_skel = []
    timing_ik = []
    timing_redis = []
    
    try:
        while streamer.is_running:
            frame_count += 1
            
            t0 = time.time()
            
            # Get 3D skeleton
            skeleton_3d, reproj_error = streamer.get_3d_skeleton()
            t_skel = time.time() - t0
            
            # Check skeleton validity and run retargeting
            skeleton_valid = False
            if skeleton_3d is None:
                fail_no_skeleton += 1
            elif not np.isfinite(skeleton_3d).all():
                fail_nan_skeleton += 1
            else:
                skeleton_valid = True
            
            # Run retargeting if skeleton is valid
            if skeleton_valid:
                t1 = time.time()
                try:
                    if use_direct:
                        # Direct mapping (faster)
                        result = retargeter.skeleton_to_robot(skeleton_3d)
                        t_ik = time.time() - t1
                        
                        if result is not None:
                            dof_pos = result['dof_pos']  # Already 29 DOF
                            last_dof_pos = dof_pos.copy()
                            last_ik_error = 0.0  # No IK error for direct mapping
                        else:
                            fail_ik_none += 1
                    else:
                        # IK-based retargeting (mink solver)
                        result = retargeter.retarget(skeleton_3d, fixed_base=True)
                        t_ik = time.time() - t1
                        
                        if result is not None:
                            # Extract joint positions (29 DOF)
                            dof_pos = result['qpos'][7:]  # Skip floating base
                            last_dof_pos = dof_pos.copy()
                            last_ik_error = result.get('error', 0.0)
                        else:
                            fail_ik_none += 1
                    
                    if result is not None:
                        # Update MuJoCo viewer if enabled
                        if args.mujoco_viz and mj_viewer is not None and mj_viewer.is_running():
                            # Set joint positions (qpos[7:] for joints after floating base)
                            # The model has 7 DOF floating base + 29 joint DOFs
                            mj_data.qpos[7:7+len(dof_pos)] = dof_pos
                            # Forward kinematics to update body positions
                            import mujoco
                            mujoco.mj_forward(mj_model, mj_data)
                            # Sync viewer
                            mj_viewer.sync()
                        
                        # Publish to Redis
                        t2 = time.time()
                        redis_client.set(
                            args.redis_key,
                            dof_pos.astype(np.float32).tobytes()
                        )
                        t_redis = time.time() - t2
                        
                        # Track timing
                        if args.debug_timing:
                            timing_skel.append(t_skel * 1000)
                            timing_ik.append(t_ik * 1000)
                            timing_redis.append(t_redis * 1000)
                        
                        # Log first successful publish
                        if first_publish:
                            print(f"\n[Redis] First publish to '{args.redis_key}' - {len(dof_pos)} DOFs")
                            print(f"[Redis] Sample values: L_shoulder={dof_pos[15]:.3f}, R_shoulder={dof_pos[22]:.3f}")
                            first_publish = False
                        
                        # Publish IK error
                        redis_client.set(
                            f"{args.redis_key}:ik_error",
                            str(last_ik_error).encode()
                        )
                        
                        valid_count += 1
                        
                except Exception as e:
                    # Retargeting failed - keep last valid pose
                    fail_ik_exception += 1
                    if fail_ik_exception <= 5:
                        print(f"\n[IK] Exception: {e}")
            
            # Display handling - vertical stack layout
            if args.display:
                # Get individual frames for vertical stacking
                frames_dict, _ = streamer.get_latest_frames()
                
                if frames_dict:
                    # Create vertical stack of camera views
                    CELL_W, CELL_H = 480, 270  # Smaller for vertical layout
                    sorted_cam_ids = sorted(frames_dict.keys())
                    
                    stacked_frames = []
                    for cam_id in sorted_cam_ids[:3]:  # Max 3 cameras
                        frame = frames_dict[cam_id]
                        resized = cv2.resize(frame, (CELL_W, CELL_H))
                        # Add camera label
                        cv2.putText(resized, f"Cam {cam_id}", (10, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        stacked_frames.append(resized)
                    
                    if stacked_frames:
                        overlay = np.vstack(stacked_frames)
                        cv2.imshow(window_name, overlay)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF if args.display else -1
            
            if key == ord('q'):
                print("\n[Teleop] Quit requested")
                break
            
            # Print status periodically
            now = time.time()
            if now - last_print_time >= 1.0:
                elapsed = now - start_time
                total_fps = frame_count / elapsed
                valid_fps = valid_count / elapsed
                
                status_str = f"\r[Teleop] FPS: {total_fps:.1f} | Valid: {valid_fps:.1f}/s | IK: {last_ik_error:.3f}"
                
                # Print timing breakdown
                if args.debug_timing and timing_ik:
                    avg_skel = sum(timing_skel[-100:]) / len(timing_skel[-100:]) if timing_skel else 0
                    avg_ik = sum(timing_ik[-100:]) / len(timing_ik[-100:]) if timing_ik else 0
                    avg_redis = sum(timing_redis[-100:]) / len(timing_redis[-100:]) if timing_redis else 0
                    status_str += f" | Skel:{avg_skel:.0f}ms IK:{avg_ik:.0f}ms Redis:{avg_redis:.1f}ms"
                
                print(status_str, end="", flush=True)
                last_print_time = now
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\n[Teleop] Stopped by user")
    
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
        # Close MuJoCo viewer
        if mj_viewer is not None:
            try:
                mj_viewer.close()
            except:
                pass
        print("[Teleop] Cleanup complete")
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Final Statistics")
    print(f"{'='*60}")
    print(f"  Total frames:    {frame_count}")
    print(f"  Valid frames:    {valid_count} ({100*valid_count/max(1,frame_count):.1f}%)")
    print(f"  Duration:        {elapsed:.1f}s")
    print(f"  Avg FPS:         {frame_count/elapsed:.1f}")
    print(f"{'='*60}")
    print(f"  Failure Breakdown:")
    print(f"    No skeleton:   {fail_no_skeleton} ({100*fail_no_skeleton/max(1,frame_count):.1f}%)")
    print(f"    NaN skeleton:  {fail_nan_skeleton} ({100*fail_nan_skeleton/max(1,frame_count):.1f}%)")
    print(f"    IK returned None: {fail_ik_none} ({100*fail_ik_none/max(1,frame_count):.1f}%)")
    print(f"    IK exception:  {fail_ik_exception} ({100*fail_ik_exception/max(1,frame_count):.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
