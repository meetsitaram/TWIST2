#!/usr/bin/env python3
"""
Live IK Teleoperation - Stream human pose to robot using End-Effector IK.

Real-time teleoperation using:
- Multi-camera 3D skeleton capture (MediaPipe + triangulation)
- End-effector IK retargeting (hands and feet)
- MuJoCo passive viewer (kinematics only, no physics)

Usage:
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    conda activate gmr
    
    # Live teleoperation
    python stream_ik_teleop.py
    
    # Record an episode
    python stream_ik_teleop.py --record --name baseline_001 --duration 30
    
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
    ):
        self.target_fps = target_fps
        self.verbose = verbose
        self.running = False
        
        # Recording setup
        self.record = record
        self.record_duration = record_duration
        self.record_video = record_video
        self.recorder: Optional[TeleopEpisodeRecorder] = None
        if record:
            self.recorder = TeleopEpisodeRecorder(
                name=record_name,
                fps=target_fps,
                smoothing="none",
                record_video=record_video,
                camera_ids=camera_ids if record_video else None,
            )
        
        # Initialize camera streamer
        print(f"[IK Teleop] Initializing cameras: {camera_ids}")
        self.camera_streamer = MultiCamPoseStreamer(
            camera_ids=camera_ids,
            calibration_file=calibration_file,
            target_fps=target_fps,
            enable_display=True,  # Show camera feed
        )
        
        # Initialize IK retargeter
        print(f"[IK Teleop] Initializing IK retargeter...")
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
        
    def _set_default_pose(self):
        """Set robot to default standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.75  # Base height
        self.data.qpos[3] = 1.0   # Quaternion w (upright)
        mujoco.mj_forward(self.model, self.data)
        
    def _apply_joint_angles(self, result: dict):
        """Apply IK result to MuJoCo data."""
        if not result.get('valid', False):
            return
            
        qpos = result.get('qpos')
        if qpos is not None:
            # Copy joint angles (index 7+)
            self.data.qpos[7:] = qpos[7:]
            
            # Copy Z height (for crouching/standing) but keep X, Y fixed
            # qpos[0:2] = X, Y position - keep fixed
            # qpos[2] = Z height - copy from IK for crouching
            # qpos[3:7] = base orientation quaternion - keep fixed
            self.data.qpos[2] = qpos[2]
            
            mujoco.mj_forward(self.model, self.data)
            
    def _wait_for_start(self, viewer):
        """Wait for ENTER key press in terminal, then countdown before starting."""
        
        print("\n" + "="*60)
        print("IK TELEOPERATION - KINEMATICS MODE")
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
                
                if skeleton_3d is not None:
                    # Run IK retargeting
                    # fixed_base=False allows pelvis height to adjust for crouching/standing
                    result = self.retargeter.retarget(
                        skeleton_3d, 
                        reset_to_default=False,  # Keep previous pose for smoothness
                        fixed_base=False,  # Allow dynamic pelvis height
                    )
                    
                    if result.get('valid', False):
                        self._apply_joint_angles(result)
                        
                        # Record frame if recording
                        if self.recorder and result.get('qpos') is not None:
                            # Get raw camera frames if recording video
                            camera_frames = None
                            if self.record_video:
                                camera_frames, _ = self.camera_streamer.get_latest_frames()
                            
                            self.recorder.add_frame(
                                human_skeleton=skeleton_3d,
                                robot_qpos=result['qpos'],
                                ik_error=result['error'],
                                camera_frames=camera_frames,
                            )
                        
                        if self.verbose:
                            print(f"[IK] error={result['error']:.4f}, "
                                  f"iters={result['iterations']}")
                
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
):
    """
    Replay an episode in MuJoCo viewer.
    
    Args:
        episode_name: Name of episode (without .npz)
        speed: Playback speed multiplier
        loop: Whether to loop playback
        show_video: Whether to show recorded video alongside
    """
    from mujoco.viewer import launch_passive
    
    # Load episode
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
        video_files = sorted(filepath.parent.glob(f"{episode.name}_cam*.mp4"))
        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            if cap.isOpened():
                video_caps[vf.stem] = cap
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
                        # Resize for display
                        h, w = img.shape[:2]
                        scale = 480 / h
                        img_small = cv2.resize(img, (int(w * scale), 480))
                        cv2.imshow(name, img_small)
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
    # Live teleoperation
    python stream_ik_teleop.py
    
    # Record an episode
    python stream_ik_teleop.py --record --name baseline_001 --duration 30
    
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
    )
    
    streamer.run()


if __name__ == "__main__":
    main()
