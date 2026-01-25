#!/usr/bin/env python3
"""
Live IK Teleoperation - Stream human pose to robot using End-Effector IK.

Real-time teleoperation using:
- Multi-camera 3D skeleton capture (MediaPipe + triangulation)
- End-effector IK retargeting (hands, elbows, and optionally feet)
- MuJoCo passive viewer (kinematics only, no physics)

IK Modes:
- Single-stage (default): Upper body only, fixed base at Z=0.75m
- Two-stage (--two-stage): Whole body - upper body first, then lower body

Usage:
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    conda activate gmr
    
    # Live teleoperation (upper body only)
    python stream_ik_teleop.py
    
    # Live teleoperation (whole body with two-stage IK)
    python stream_ik_teleop.py --two-stage
    
    # Record an episode with two-stage IK
    python stream_ik_teleop.py --record --name wholebody_001 --duration 30 --two-stage
    
    # List recorded episodes
    python stream_ik_teleop.py --list-episodes
    
    # Replay a recorded episode
    python stream_ik_teleop.py --replay baseline_001
    python stream_ik_teleop.py --replay baseline_001 --speed 0.5 --loop
    
Press 'Q' or Ctrl+C to quit.
"""

import argparse
import time
import numpy as np
import os
import sys
import mujoco
import toml
from pathlib import Path
from typing import Optional

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from deploy_real.multicam_pose_streamer import MultiCamPoseStreamer
from deploy_real.end_effector_ik_retarget import EndEffectorIKRetargeter, ROBOT_MODEL_PATH
from deploy_real.teleop_episode_recorder import TeleopEpisodeRecorder, EPISODES_DIR

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Default calibration file
DEFAULT_CALIBRATION = Path(project_root) / "calibration" / "calibration.toml"


class IKTeleopStreamer:
    """Live IK teleoperation with MuJoCo visualization."""
    
    def __init__(
        self,
        camera_ids: list,
        calibration_file: str,
        target_fps: int = 30,
        verbose: bool = False,
        record: bool = False,
        record_duration: float = 60.0,
        record_name: str = None,
        record_video: bool = False,
        skeleton_smoothing: str = "none",
        smoothing_min_cutoff: float = 1.0,
        smoothing_beta: float = 0.007,
        two_stage: bool = False,
    ):
        self.target_fps = target_fps
        self.verbose = verbose
        self.running = False
        self.two_stage = two_stage
        
        # Recording setup
        self.record = record
        self.record_duration = record_duration
        self.record_video = record_video
        self.recorder: Optional[TeleopEpisodeRecorder] = None
        if record:
            # Build smoothing params dict for metadata
            smoothing_params = {}
            if skeleton_smoothing == "one_euro":
                smoothing_params = {
                    "min_cutoff": smoothing_min_cutoff,
                    "beta": smoothing_beta,
                }
            
            self.recorder = TeleopEpisodeRecorder(
                name=record_name,
                fps=target_fps,
                smoothing=skeleton_smoothing,
                smoothing_params=smoothing_params,
                record_video=record_video,
                camera_ids=camera_ids if record_video else None,
            )
        
        # Initialize camera streamer
        print(f"[IK Teleop] Initializing cameras: {camera_ids}")
        smoothing_label = skeleton_smoothing if skeleton_smoothing != "none" else "disabled"
        print(f"[IK Teleop] Skeleton smoothing: {smoothing_label}")
        self.camera_streamer = MultiCamPoseStreamer(
            camera_ids=camera_ids,
            calibration_file=calibration_file,
            target_fps=target_fps,
            enable_display=True,  # Show camera feed
            skeleton_smoothing=skeleton_smoothing,
            smoothing_min_cutoff=smoothing_min_cutoff,
            smoothing_beta=smoothing_beta,
        )
        
        # Initialize IK retargeter
        mode_str = "two-stage (whole body)" if two_stage else "single-stage (upper body)"
        print(f"[IK Teleop] Initializing IK retargeter ({mode_str})...")
        self.retargeter = EndEffectorIKRetargeter(
            verbose=False,
            max_iterations=30,  # Fewer iterations for real-time
        )
        
        # Initialize MuJoCo model for visualization
        print(f"[IK Teleop] Loading robot model: {ROBOT_MODEL_PATH}")
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        
        # Set initial pose
        self._set_default_pose()
        
        # Stats
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # IK failure recovery - balanced settings
        self.ik_error_threshold = 0.5  # Relaxed - only reject very bad IK results
        self.ik_error_history = []
        self.ik_error_history_size = 10  # Track last N frames
        self.consecutive_failures = 0
        self.failure_recovery_threshold = 10  # Reset after 10 consecutive failures
        self.last_valid_qpos = None  # Track last known good state
        self.frames_since_last_reset = 0  # Prevent reset spam
        self.min_frames_between_resets = 60  # 2 seconds between resets
        
        # Joint velocity limiting for safety
        self.max_joint_velocity = 2.0  # rad/s max change per joint
        self.prev_qpos = None  # For velocity limiting
        
    def _set_default_pose(self):
        """Set robot to default standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.75  # Base height
        self.data.qpos[3] = 1.0   # Quaternion w (upright)
        mujoco.mj_forward(self.model, self.data)
        
    def _velocity_limit_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Apply velocity limiting to qpos (DISABLED for now - just pass through).
        """
        # Store for recording
        self.prev_qpos = qpos.copy()
        return qpos
    
    def _apply_to_visualizer(self, qpos_limited: np.ndarray):
        """
        Apply velocity-limited qpos to MuJoCo visualizer.
        
        Args:
            qpos_limited: Already velocity-limited qpos from _velocity_limit_qpos()
        """
        # Copy joint angles (index 7+)
        self.data.qpos[7:] = qpos_limited[7:]
        
        # Copy Z height (for crouching/standing) but keep X, Y fixed
        # qpos[0:2] = X, Y position - keep fixed
        # qpos[2] = Z height - copy from IK for crouching
        # qpos[3:7] = base orientation quaternion - keep fixed
        self.data.qpos[2] = qpos_limited[2]
        
        mujoco.mj_forward(self.model, self.data)
    
    def _send_to_robot(self, qpos_limited: np.ndarray):
        """
        Send velocity-limited qpos to the actual robot.
        
        This is a placeholder for future robot integration.
        Any real robot command should go through this method to ensure
        velocity limiting is always applied.
        
        Args:
            qpos_limited: Already velocity-limited qpos from _velocity_limit_qpos()
        """
        # TODO: Implement actual robot communication
        # Example: self.robot_client.send_joint_positions(qpos_limited[7:])
        pass
    
    def _apply_joint_angles(self, result: dict):
        """Apply IK result with velocity limiting to all outputs."""
        if not result.get('valid', False):
            return
            
        qpos = result.get('qpos')
        if qpos is not None:
            # Apply velocity limiting - this is the FINAL safety gate
            qpos_limited = self._velocity_limit_qpos(qpos)
            
            # Apply to visualizer
            self._apply_to_visualizer(qpos_limited)
            
            # Send to actual robot (if connected)
            self._send_to_robot(qpos_limited)
            
    def _wait_for_start(self, viewer):
        """Wait for ENTER key press in terminal, then countdown before starting."""
        
        mode_str = "TWO-STAGE (WHOLE BODY)" if self.two_stage else "SINGLE-STAGE (UPPER BODY)"
        print("\n" + "="*60)
        print(f"IK TELEOPERATION - {mode_str}")
        print("="*60)
        print("Controls:")
        print("  - Press ENTER in terminal to start (10s countdown)")
        print("  - Press Ctrl+C to quit")
        print("  - Close MuJoCo viewer to quit")
        print("="*60 + "\n")
        
        input("[IK Teleop] Press ENTER to start countdown...")
        
        if not viewer.is_running():
            return False
        
        # Countdown
        countdown_seconds = 10
        print(f"\n[IK Teleop] Starting in {countdown_seconds} seconds - GET INTO POSITION!")
        
        for remaining in range(countdown_seconds, 0, -1):
            print(f"  {remaining}...")
            time.sleep(1)
            viewer.sync()
            
            if not viewer.is_running():
                return False
        
        print("\n[IK Teleop] GO!\n")
        
        # Start recording if enabled
        if self.recorder:
            self.recorder.start()
            print(f"[IK Teleop] Recording for {self.record_duration}s...")
        
        return True
    
    def run(self):
        """Main loop: stream poses and visualize."""
        from mujoco.viewer import launch_passive
        
        # Start camera streaming (has its own display window)
        self.camera_streamer.start()
        self.running = True
        
        # Launch MuJoCo passive viewer
        viewer = launch_passive(
            self.model, 
            self.data, 
            show_left_ui=False, 
            show_right_ui=True
        )
        
        # Wait for enter and countdown
        if not self._wait_for_start(viewer):
            self.camera_streamer.stop()
            viewer.close()
            return
        
        record_start_time = time.time()
        
        try:
            while viewer.is_running() and self.running:
                loop_start = time.time()
                
                # Check recording duration
                if self.recorder and (time.time() - record_start_time) >= self.record_duration:
                    print(f"\n[IK Teleop] Recording complete ({self.record_duration}s)")
                    break
                
                # Get latest 3D skeleton from cameras
                skeleton_3d, reproj_error = self.camera_streamer.get_3d_skeleton()
                
                if skeleton_3d is None:
                    # Skeleton tracking lost - count as failure
                    self.consecutive_failures += 1
                    if self.consecutive_failures == 1:
                        print("\n[IK Teleop] Skeleton tracking lost...")
                
                if skeleton_3d is not None:
                    self.frames_since_last_reset += 1
                    
                    # Check if we need to recover due to previous failures
                    # Only recover if enough time has passed since last recovery (prevent spam)
                    should_recover = (self.consecutive_failures >= self.failure_recovery_threshold and 
                                     self.frames_since_last_reset >= self.min_frames_between_resets)
                    
                    if should_recover:
                        # Reset to default pose for clean recovery
                        print(f"\n[IK Teleop] Resetting to default pose (from {self.consecutive_failures} failures)")
                        self.consecutive_failures = 0
                        self.frames_since_last_reset = 0
                        # Keep prev_qpos as-is - next valid IK result will establish new baseline
                    
                    # Run IK retargeting
                    if self.two_stage:
                        # Two-stage IK: upper body first, then lower body
                        # Pelvis Z is derived from foot positions
                        result = self.retargeter.retarget_two_stage(
                            skeleton_3d, 
                            reset_to_default=should_recover,
                        )
                    else:
                        # Single-stage: upper body only with fixed base
                        # fixed_base=True keeps pelvis at constant height (0.75m)
                        result = self.retargeter.retarget(
                            skeleton_3d, 
                            reset_to_default=should_recover,
                            fixed_base=True,  # Fixed pelvis height (arm-only tracking)
                        )
                    
                    if result.get('valid', False):
                        ik_error = result['error']
                        
                        # Track IK error for logging
                        self.ik_error_history.append(ik_error)
                        if len(self.ik_error_history) > self.ik_error_history_size:
                            self.ik_error_history.pop(0)
                        
                        # Reset failure counter on valid result
                        self.consecutive_failures = 0
                        self.last_valid_qpos = result['qpos'].copy()
                        
                        # ALWAYS apply - velocity limiting provides safety
                        self._apply_joint_angles(result)
                        
                        # Record frame (velocity-limited qpos)
                        if self.recorder and self.prev_qpos is not None:
                            camera_frames = None
                            if self.record_video:
                                camera_frames, _ = self.camera_streamer.get_latest_frames()
                            
                            self.recorder.add_frame(
                                human_skeleton=skeleton_3d,
                                robot_qpos=self.prev_qpos,  # Velocity-limited
                                ik_error=result['error'],
                                camera_frames=camera_frames,
                            )
                        
                        if self.verbose:
                            print(f"[IK] error={result['error']:.4f}, iters={result['iterations']}")
                    else:
                        # Invalid result - count as failure
                        self.consecutive_failures += 1
                
                # Update viewer
                viewer.sync()
                
                # Update FPS counter
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now
                    print(f"[IK Teleop] FPS: {self.fps:.1f}")
                
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / self.target_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n[IK Teleop] Interrupted by user")
        finally:
            self.running = False
            
            # Save recording if active
            if self.recorder:
                self.recorder.stop()
                self.recorder.save()
            
            self.camera_streamer.stop()
            viewer.close()
            print("[IK Teleop] Stopped")


def list_episodes():
    """List all available episodes with info."""
    episodes = TeleopEpisodeRecorder.list_episodes()
    
    if not episodes:
        print(f"\nNo episodes found in: {EPISODES_DIR}")
        return
    
    print(f"\n{'='*70}")
    print(f"  Available Episodes ({len(episodes)})")
    print(f"{'='*70}")
    print(f"{'Name':<25} {'Frames':>8} {'Duration':>10} {'Smoothing':<15} {'Video'}")
    print("-" * 70)
    
    for filepath in episodes:
        try:
            episode = TeleopEpisodeRecorder.load(filepath)
            
            # Check for video files
            video_files = list(filepath.parent.glob(f"{episode.name}_cam*.mp4"))
            has_video = "Yes" if video_files else "No"
            
            print(f"{episode.name:<25} {episode.num_frames:>8} {episode.duration_sec:>9.1f}s {episode.smoothing:<15} {has_video}")
        except Exception as e:
            print(f"{filepath.stem:<25} [Error loading: {e}]")
    
    print()


def replay_episode(
    episode_name: str,
    speed: float = 1.0,
    loop: bool = False,
    show_video: bool = False,
    camera_id: int = None,
):
    """
    Replay an episode in MuJoCo viewer.
    
    Args:
        episode_name: Name of episode (without .npz)
        speed: Playback speed multiplier
        loop: Whether to loop playback
        show_video: Whether to show recorded video alongside
        camera_id: Specific camera ID to show (None = all cameras)
    """
    from mujoco.viewer import launch_passive
    
    # Load episode
    # Episodes are stored in subdirectories: EPISODES_DIR/episode_name/episode_name.npz
    filepath = EPISODES_DIR / episode_name / f"{episode_name}.npz"
    if not filepath.exists():
        # Try old format (flat directory) for backward compatibility
        filepath = EPISODES_DIR / f"{episode_name}.npz"
    if not filepath.exists():
        print(f"Episode not found: {episode_name}")
        print(f"Looking in: {EPISODES_DIR}")
        list_episodes()
        return
    
    print(f"Loading episode: {filepath}")
    episode = TeleopEpisodeRecorder.load(filepath)
    
    # Print info
    print(f"\n{'='*50}")
    print(f"  Episode: {episode.name}")
    print(f"{'='*50}")
    print(f"  Frames:     {episode.num_frames}")
    print(f"  Duration:   {episode.duration_sec:.1f} seconds")
    print(f"  FPS:        {episode.fps}")
    print(f"  Smoothing:  {episode.smoothing}")
    print(f"  Playback:   {speed}x speed, {'looping' if loop else 'once'}")
    print()
    
    # Load MuJoCo model
    print(f"Loading robot model...")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL_PATH))
    data = mujoco.MjData(model)
    
    # Launch viewer
    viewer = launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15
    
    # Video playback setup
    video_caps = {}
    if show_video and HAS_CV2:
        if camera_id is not None:
            # Load specific camera only
            video_file = filepath.parent / f"{episode.name}_cam{camera_id}.mp4"
            if video_file.exists():
                cap = cv2.VideoCapture(str(video_file))
                if cap.isOpened():
                    video_caps[video_file.stem] = cap
                    # Create resizable window
                    cv2.namedWindow(video_file.stem, cv2.WINDOW_NORMAL)
                    print(f"  Loaded video: {video_file.name}")
            else:
                print(f"  Warning: Camera {camera_id} video not found: {video_file.name}")
        else:
            # Load all cameras
            video_files = sorted(filepath.parent.glob(f"{episode.name}_cam*.mp4"))
            for vf in video_files:
                cap = cv2.VideoCapture(str(vf))
                if cap.isOpened():
                    video_caps[vf.stem] = cap
                    # Create resizable window
                    cv2.namedWindow(vf.stem, cv2.WINDOW_NORMAL)
                    print(f"  Loaded video: {vf.name}")
    elif show_video and not HAS_CV2:
        print("  Warning: OpenCV not available for video playback")
    
    # Playback parameters
    frame_dt = 1.0 / episode.fps / speed
    frame_idx = 0
    total_frames = episode.num_frames
    
    print("\nPlaying... (close viewer to stop)")
    print()
    
    try:
        while viewer.is_running():
            t_start = time.time()
            
            # Get current frame
            frame = episode.frames[frame_idx]
            qpos = frame.robot_qpos
            
            # Apply to MuJoCo (qpos has root + joints)
            if len(qpos) >= 36:
                # Set root position
                data.qpos[:3] = qpos[:3]
                # Set root orientation
                data.qpos[3:7] = qpos[3:7]
                # Set joint positions
                data.qpos[7:36] = qpos[7:36]
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Update camera to follow robot
            try:
                pelvis_id = model.body("pelvis").id
                pelvis_pos = data.xpos[pelvis_id]
                viewer.cam.lookat = pelvis_pos
            except:
                pass
            
            viewer.sync()
            
            # Show video frames
            if video_caps:
                for name, cap in video_caps.items():
                    ret, img = cap.read()
                    if ret:
                        cv2.imshow(name, img)
                cv2.waitKey(1)
            
            # Print progress
            current_time = frame.t_ms / 1000.0
            ik_error = frame.ik_error
            print(f"\rFrame {frame_idx+1:>5}/{total_frames} | "
                  f"Time: {current_time:>6.2f}s | "
                  f"IK Error: {ik_error:.4f}", end="")
            
            # Next frame
            frame_idx += 1
            if frame_idx >= total_frames:
                if loop:
                    frame_idx = 0
                    # Reset video captures
                    for cap in video_caps.values():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("\n[Looping...]")
                else:
                    print("\n\nPlayback complete!")
                    # Keep viewer open for inspection
                    print("Viewer still open - close window to exit")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    break
            
            # Maintain playback speed
            elapsed = time.time() - t_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Cleanup
        viewer.close()
        for cap in video_caps.values():
            cap.release()
        if video_caps:
            cv2.destroyAllWindows()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Live IK Teleoperation with record and replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Live teleoperation (upper body only)
    python stream_ik_teleop.py
    
    # Live teleoperation (whole body with two-stage IK)
    python stream_ik_teleop.py --two-stage
    
    # Record an episode (upper body)
    python stream_ik_teleop.py --record --name baseline_001 --duration 30
    
    # Record with two-stage IK (whole body)
    python stream_ik_teleop.py --record --name wholebody_001 --duration 30 --two-stage
    
    # Record with video
    python stream_ik_teleop.py --record --name baseline_001 --video
    
    # List recorded episodes
    python stream_ik_teleop.py --list-episodes
    
    # Replay an episode
    python stream_ik_teleop.py --replay baseline_001
    
    # Replay slowly with video
    python stream_ik_teleop.py --replay baseline_001 --speed 0.5 --video
    
    # Replay in loop
    python stream_ik_teleop.py --replay baseline_001 --loop
"""
    )
    # Replay/list options
    parser.add_argument("--replay", type=str, metavar="EPISODE",
                       help="Replay a recorded episode")
    parser.add_argument("--list-episodes", action="store_true",
                       help="List all recorded episodes")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                       help="Playback speed for replay (default: 1.0)")
    parser.add_argument("--loop", "-l", action="store_true",
                       help="Loop replay playback")
    parser.add_argument("--cam", type=int, default=None,
                       help="Specific camera ID to show during replay (default: all)")
    
    # Live streaming options
    parser.add_argument("--cameras", "-c", type=int, nargs="+", default=None,
                       help="Camera IDs (default: read from calibration file)")
    parser.add_argument("--calibration", type=str, 
                       default=str(DEFAULT_CALIBRATION),
                       help="Path to calibration file")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    # Recording options
    parser.add_argument("--record", "-r", action="store_true",
                       help="Record episode for jitter analysis")
    parser.add_argument("--duration", "-d", type=float, default=60.0,
                       help="Recording duration in seconds (default: 60)")
    parser.add_argument("--name", "-n", type=str, default=None,
                       help="Episode name (default: auto-generated)")
    parser.add_argument("--video", action="store_true",
                       help="Record/show video (for record or replay)")
    
    # Smoothing options
    parser.add_argument("--smoothing", type=str, default="none",
                       choices=["none", "one_euro"],
                       help="Skeleton smoothing method (default: none)")
    parser.add_argument("--smooth-cutoff", type=float, default=1.0,
                       help="One Euro min_cutoff - lower = smoother (default: 1.0)")
    parser.add_argument("--smooth-beta", type=float, default=0.007,
                       help="One Euro beta - higher = more responsive (default: 0.007)")
    
    # IK mode options
    parser.add_argument("--two-stage", "-2", action="store_true",
                       help="Use two-stage IK (whole body: upper body first, then lower body)")
    
    args = parser.parse_args()
    
    # Handle list-episodes
    if args.list_episodes:
        list_episodes()
        return
    
    # Handle replay
    if args.replay:
        replay_episode(
            episode_name=args.replay,
            speed=args.speed,
            loop=args.loop,
            show_video=args.video,
            camera_id=args.cam,
        )
        return
    
    # Live streaming mode - check calibration file exists
    if not os.path.exists(args.calibration):
        print(f"Error: Calibration file not found: {args.calibration}")
        print("Run calibration first or specify --calibration path")
        return
    
    # Read camera IDs from calibration file if not specified
    camera_ids = args.cameras
    if camera_ids is None:
        calib_data = toml.load(args.calibration)
        camera_ids = calib_data.get('metadata', {}).get('cameras', [0, 1, 2])
        print(f"[IK Teleop] Using cameras from calibration: {camera_ids}")
    
    streamer = IKTeleopStreamer(
        camera_ids=camera_ids,
        calibration_file=args.calibration,
        target_fps=args.fps,
        verbose=args.verbose,
        record=args.record,
        record_duration=args.duration,
        record_name=args.name,
        record_video=args.video,
        skeleton_smoothing=args.smoothing,
        smoothing_min_cutoff=args.smooth_cutoff,
        smoothing_beta=args.smooth_beta,
        two_stage=args.two_stage,
    )
    
    streamer.run()


if __name__ == "__main__":
    main()
